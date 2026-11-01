"""Local, streaming media adapter. GPL-2.0; originals are opened read-only.

PyAV carries each decoded presentation timestamp through the render. FFmpeg
remuxes original audio and metadata. Public export owns an isolated worker so a
cancel/deadline can stop native codec/CUDA calls as well as pipe waits.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

FFMPEG = os.environ.get("STUDIO_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("STUDIO_FFPROBE", "ffprobe")
_DEMUXERS = "mov,matroska,webm,avi,mxf,mpegts,mpeg,mpegvideo,asf,flv,ogg,nut"
_PROTOCOLS = ["-protocol_whitelist", "file,pipe", "-format_whitelist", _DEMUXERS]


def _regular(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("Media input must be a regular local file")
    return path


def _run(args, timeout=120):
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=timeout, check=False)
    if result.returncode:
        raise ValueError("Media command failed: " + result.stderr.decode(errors="replace")[-6000:])
    return result.stdout


def _probe(path):
    return json.loads(_run([FFPROBE, "-v", "error", *_PROTOCOLS,
                           "-show_streams", "-show_format", "-of", "json", str(path)], 60))


def _video(data):
    result = next((s for s in data["streams"] if s.get("codec_type") == "video"
                   and not s.get("disposition", {}).get("attached_pic")), None)
    if result is None:
        raise ValueError("Input has no decodable video stream")
    return result


def _fraction(value, default="0/1"):
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError, TypeError):
        return Fraction(default)


def _rotation(video):
    angle = next((item["rotation"] for item in video.get("side_data_list", [])
                  if "rotation" in item), video.get("tags", {}).get("rotate", 0))
    value = float(angle)
    if not math.isfinite(value) or abs(value / 90 - round(value / 90)) > 0.001:
        raise ValueError("Only multiples of 90 degrees of display rotation are supported")
    return int(round(value / 90)) * 90 % 360


def _depth(pixel_format, video=None):
    import av
    try:
        return max(c.bits for c in av.VideoFormat(pixel_format).components)
    except (ValueError, AttributeError):
        value = int((video or {}).get("bits_per_raw_sample") or 8)
        return value or 8


def _transfer(video):
    value = video.get("color_transfer", "unknown")
    return {"smpte2084": "pq", "arib-std-b67": "hlg", "iec61966-2-1": "srgb"}.get(value, "bt709")


def _matrix_fallback(video):
    return 1 if int(video["height"]) >= 720 else 5


def inspect_media(path):
    """Fast metadata inspection; fps_exact is rational, not a VFR conformance.

    Timestamp-preserving export also supports VFR. Average FPS alone cannot
    identify all VFR footage, so inspection does not claim a full timing scan.
    """
    path = _regular(path)
    data = _probe(path)
    video = _video(data)
    rate = _fraction(video.get("avg_frame_rate")) or _fraction(video.get("r_frame_rate"))
    if rate <= 0:
        raise ValueError("Video has no usable nominal frame rate")
    width, height = int(video["width"]), int(video["height"])
    if not 1 <= width <= 16384 or not 1 <= height <= 16384:
        raise ValueError("Video geometry exceeds the 16384-pixel safety bound")
    rotation = _rotation(video)
    sar = _fraction(video.get("sample_aspect_ratio", "1:1").replace(":", "/"), "1/1") or Fraction(1)
    duration = float(video.get("duration") or data.get("format", {}).get("duration") or 0)
    warnings = []
    if video.get("color_transfer", "unknown") in ("unknown", "unspecified", "reserved", None):
        warnings.append("Transfer is unspecified; effect processing assumes BT.709 without changing source color tags.")
    if sar != 1:
        warnings.append("Non-square source pixels are retained; lens coordinates use the source pixel grid.")
    if rotation:
        warnings.append("Display rotation is retained; effects run in the displayed orientation.")
    pixel = video.get("pix_fmt", "unknown")
    if video.get("color_space", "unknown") in ("unknown", "unspecified", None):
        warnings.append("YUV matrix is unspecified; RGB conversion assumes " +
                        ("BT.709 for HD." if height >= 720 else "BT.601 for SD."))
    if video.get("color_range", "unknown") in ("unknown", "unspecified", None):
        warnings.append("Color range is unspecified; RGB/YUVJ uses full range and other YUV uses limited range.")
    if any(s.get("codec_type") not in ("video", "audio") for s in data["streams"]):
        warnings.append("Subtitle and data tracks are not included in rendered exports.")
    if any("mastering" in str(item.get("side_data_type", "")).lower() or "content light" in str(item.get("side_data_type", "")).lower()
           for item in video.get("side_data_list", [])):
        warnings.append("HDR transfer/primaries/matrix are retained; mastering-display and content-light side data are not copied by this renderer.")
    if _transfer(video) in ("pq", "hlg"):
        warnings.append("HDR signal and color tags are retained; mastering-display, content-light and dynamic HDR metadata are not rebuilt after effects.")
    return {"name": path.name, "width": width, "height": height,
            "display_width": height if rotation in (90, 270) else width,
            "display_height": width if rotation in (90, 270) else height,
            "rotation": rotation, "sample_aspect_ratio": str(sar),
            "fps": float(rate), "fps_exact": str(rate), "duration": duration,
            "video_stream_index": int(video["index"]),
            "codec": video["codec_name"], "container": path.suffix.lower().lstrip("."),
            "pixel_format": pixel, "bit_depth": _depth(pixel, video),
            "audio_streams": sum(s.get("codec_type") == "audio" for s in data["streams"]),
            "transfer": _transfer(video), "color_transfer": video.get("color_transfer", "unknown"),
            "color_primaries": video.get("color_primaries", "unknown"),
            "color_space": video.get("color_space", "unknown"), "color_range": video.get("color_range", "unknown"),
            "processing_matrix": _matrix_fallback(video) if video.get("color_space", "unknown") in ("unknown", "unspecified", None) else video["color_space"],
            "warnings": warnings}


def _publish(staged, output):
    # Atomic new-name publication. Existing files, hardlinks and symlinks win.
    os.link(staged, output)


def _new_output(path):
    path = Path(path).absolute()
    if os.path.lexists(path):
        raise FileExistsError(f"Refusing existing output: {path}")
    if not path.parent.is_dir():
        raise ValueError("Output parent must already exist")
    return path


def make_proxy(source, outmp4, posterjpg):
    """Checked, browser-safe SDR H.264/AAC proxy; never changes the source."""
    source = _regular(source)
    info = inspect_media(source)
    selected_video = _video(_probe(source))["index"]
    output, poster = _new_output(outmp4), _new_output(posterjpg)
    if output == poster:
        raise ValueError("Proxy and poster paths must differ")
    with tempfile.TemporaryDirectory(prefix=".studio-proxy-", dir=output.parent) as folder:
        temporary = Path(folder)
        proxy, jpg = temporary / "proxy.mp4", temporary / "poster.jpg"
        filters = []
        if info["transfer"] in ("pq", "hlg"):
            filters += ["zscale=t=linear:npl=100", "format=gbrpf32le", "tonemap=tonemap=mobius:desat=0",
                        "zscale=p=bt709:t=bt709:m=bt709:r=tv"]
        filters += ["scale=w='min(1280,2*trunc(iw*sar/2))':h='max(2,2*trunc(ow/dar/2))':flags=lanczos",
                    "setsar=1", "format=yuv420p"]
        _run([FFMPEG, "-hide_banner", "-v", "error", "-xerror", "-n", *_PROTOCOLS,
              "-i", str(source), "-map", f"0:{selected_video}", "-map", "0:a:0?", "-vf", ",".join(filters),
              "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "160k",
              "-movflags", "+faststart", str(proxy)], 7200)
        _run([FFMPEG, "-v", "error", "-xerror", "-n", *_PROTOCOLS, "-i", str(proxy),
              "-frames:v", "1", "-update", "1", str(jpg)], 120)
        saved = _video(_probe(proxy))
        if saved["codec_name"] != "h264" or int(saved["width"]) > 1280 or not jpg.stat().st_size:
            raise ValueError("Proxy verification failed")
        _run([FFMPEG, "-v", "error", "-xerror", *_PROTOCOLS, "-i", str(proxy),
              "-f", "null", "-"], 7200)
        _publish(proxy, output)
        try:
            _publish(jpg, poster)
        except BaseException:
            if output.exists() and os.path.samefile(proxy, output):
                output.unlink()
            raise
    return {"proxy": str(output), "poster": str(poster), "warnings": info["warnings"]}


def _pixel_signature(pixel):
    import av
    fmt = av.VideoFormat(pixel)
    return (max(c.bits for c in fmt.components), fmt.is_rgb,
            fmt.chroma_width(128), fmt.chroma_height(128), len(fmt.components))


def _encoder_plan(info, mode, force_cpu=False):
    """Choose exact format or fail; match never silently changes the codec."""
    import av
    source_codec, pixel = info["codec"], info["pixel_format"]
    if _pixel_signature(pixel)[-1] == 4 and mode == "match":
        raise ValueError("Match-source alpha-channel processing is not supported; use a non-alpha export mode explicitly")
    codec = {"hevc": "hevc", "prores": "prores", "lossless": "ffv1", "preview": "h264"}.get(mode, source_codec)
    if mode not in ("match", "hevc", "prores", "lossless", "preview"):
        raise ValueError("Unknown export mode")
    if mode == "lossless":
        return {"encoder": "ffv1", "codec": "ffv1", "pixel": "gbrp16le", "options": {"level": "3", "coder": "1", "context": "1"}, "quality": "lossless RGB16"}
    if mode == "prores":
        pixel = "yuv422p10le"
    elif mode in ("preview", "hevc"):
        pixel = "yuv420p10le" if mode == "hevc" and info["bit_depth"] > 8 else "yuv420p"
    pixel = {"yuvj420p": "yuv420p", "yuvj422p": "yuv422p", "yuvj444p": "yuv444p"}.get(pixel, pixel)
    mapping = {"h264": "libx264", "hevc": "libx265", "prores": "prores_ks", "dnxhd": "dnxhd",
               "vp9": "libvpx-vp9", "vp8": "libvpx", "av1": "libsvtav1", "mpeg4": "mpeg4",
               "mpeg2video": "mpeg2video", "mjpeg": "mjpeg", "ffv1": "ffv1", "rawvideo": "rawvideo", "huffyuv": "huffyuv"}
    if codec not in mapping:
        raise ValueError(f"Match-source encoder is unavailable for {codec}; select another export mode explicitly")
    encoder = mapping[codec]
    options, quality = {}, "highest quality"
    width, height = info["width"], info["height"]
    nv_ok = not force_cpu and width % 2 == height % 2 == 0 and width >= 144 and height >= 128
    nv_pixel = {"yuv420p": "yuv420p", "yuv420p10le": "p010le", "yuv444p": "yuv444p", "yuv444p10le": "yuv444p16le"}.get(pixel)
    if codec in ("h264", "hevc") and nv_ok and nv_pixel and max(width, height) <= (4096 if codec == "h264" else 8192) and not (codec == "h264" and info["bit_depth"] > 8):
        encoder, pixel = codec + "_nvenc", nv_pixel
        if mode == "match":
            options, quality = {"preset": "p3", "tune": "lossless", "rc": "constqp", "qp": "0"}, "lossless codec"
        else:
            options = {"preset": "p5", "rc": "vbr", "cq": "16", "b": "0"}
    elif codec in ("h264", "hevc"):
        options, quality = {"preset": "medium", "crf": "0"}, "lossless codec"
        if codec == "hevc":
            options = {"preset": "medium", "x265-params": "lossless=1:log-level=error"}
    elif codec == "ffv1":
        options, quality = {"level": "3"}, "lossless codec"
    elif codec == "vp9":
        options, quality = {"lossless": "1", "b": "0"}, "lossless codec"
    elif codec == "av1":
        options = {"crf": "1", "preset": "6", "svtav1-params": "lp=4"}
    elif codec == "vp8":
        options = {"crf": "4", "b": "100M"}
    elif codec == "prores":
        options = {"profile": "4" if "444" in pixel else "3", "qscale": "1"}
    elif codec in ("mjpeg", "mpeg4", "mpeg2video"):
        options = {"qscale": "1"}
    elif codec == "dnxhd":
        options = {"profile": "dnxhr_hqx" if "10" in pixel else "dnxhr_hq"}
    if encoder not in av.codecs_available:
        raise ValueError(f"Installed PyAV does not provide {encoder}")
    formats = {f.name for f in av.CodecContext.create(encoder, "w").codec.video_formats or []}
    if pixel not in formats and formats:
        raise ValueError(f"{encoder} cannot preserve source pixel format {pixel}; choose an explicit conversion mode")
    return {"encoder": encoder, "codec": codec, "pixel": pixel, "options": options, "quality": quality}


def _active_bytes(frame):
    """Canonical RGB16 bytes, without decoder plane padding."""
    return frame.to_ndarray(format="rgb48le").tobytes()


def _audio_summary(path):
    """Payload integrity and timestamp extent, without retaining audio packets."""
    metadata = _probe(path)
    time_bases = {str(s["index"]): Fraction(s["time_base"]) for s in metadata["streams"] if s.get("codec_type") == "audio"}
    result = {}
    with tempfile.TemporaryFile() as error:
        proc = subprocess.Popen([FFPROBE, "-v", "error", *_PROTOCOLS, "-select_streams", "a",
            "-show_packets", "-show_data_hash", "sha256", "-show_entries", "packet=stream_index,data_hash,pts,duration",
            "-of", "compact=p=0:nk=0", str(path)], stdout=subprocess.PIPE, stderr=error, text=True)
        watchdog = threading.Timer(7200, _stop_tree, args=(proc,))
        watchdog.daemon = True
        watchdog.start()
        try:
            for line in proc.stdout:
                fields = dict(item.split("=", 1) for item in line.strip().split("|") if "=" in item)
                if "stream_index" in fields and "data_hash" in fields:
                    item = result.setdefault(fields["stream_index"], [hashlib.sha256(), 0, None, None])
                    item[0].update(fields["data_hash"].encode("ascii"))
                    item[1] += 1
                    if fields.get("pts", "N/A") != "N/A":
                        point = int(fields["pts"]) * time_bases[fields["stream_index"]]
                        end = point + int(fields.get("duration", "0").replace("N/A", "0")) * time_bases[fields["stream_index"]]
                        item[2] = min(item[2], point) if item[2] is not None else point
                        item[3] = max(item[3], end) if item[3] is not None else end
            if proc.wait(timeout=30):
                raise ValueError("Audio packet verification failed")
        finally:
            watchdog.cancel()
            proc.stdout.close()
            _stop_tree(proc)
    return [(value[0].hexdigest(), value[1], value[2], value[3]) for value in result.values()]


def _audio_hashes(path):
    return [item[:2] for item in _audio_summary(path)]


def _stop_tree(proc):
    if proc is None or proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
        except (OSError, subprocess.SubprocessError):
            proc.kill()
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            proc.kill()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=15)


def export_video(source, output_path, settings, mode="match", progress_callback=None,
                 cancel_event=None, preview=False):
    """Synchronous cancellable export, with exact VFR presentation timestamps.

    Outputs are published only after worker validation. A file racing the final
    publication is preserved. Match retains codec/container/raster/chroma/depth,
    original audio packets and source color tags; unsupported combinations fail.
    """
    source, output = _regular(source), _new_output(output_path)
    settings = dict(settings)
    info = inspect_media(source)
    actual_mode = "preview" if preview else mode
    suffix = {"hevc": ".mp4", "prores": ".mov", "lossless": ".mkv", "preview": ".mp4"}.get(actual_mode, source.suffix.lower())
    if output.suffix.lower() != suffix:
        raise ValueError(f"{actual_mode} export requires {suffix} output")
    timeout = float(settings.pop("_timeout_seconds", 7200))
    if not math.isfinite(timeout) or not 1 <= timeout <= 86400:
        raise ValueError("Export timeout must be 1..86400 seconds")
    callback = progress_callback or (lambda progress, message: None)
    cancel = cancel_event or threading.Event()
    if cancel.is_set():
        raise InterruptedError("Export canceled")
    with tempfile.TemporaryDirectory(prefix=".studio-export-", dir=output.parent) as folder:
        work = Path(folder)
        request = {"source": str(source), "settings": settings, "mode": actual_mode,
                   "work": str(work), "suffix": suffix, "ffmpeg": FFMPEG, "ffprobe": FFPROBE}
        (work / "request.json").write_text(json.dumps(request), encoding="utf-8")
        options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        with (work / "worker.log").open("wb") as log:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(work / "request.json")],
                                    stdout=log, stderr=log, **options)
            started, last = time.monotonic(), ""
            try:
                while proc.poll() is None:
                    if cancel.is_set():
                        raise InterruptedError("Export canceled; source and existing outputs are untouched")
                    if time.monotonic() - started > timeout:
                        raise TimeoutError("Export deadline reached")
                    try:
                        text = (work / "progress.json").read_text(encoding="utf-8")
                        if text != last:
                            progress = json.loads(text)
                            callback(progress["progress"], progress["message"])
                            last = text
                    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
                        pass
                    time.sleep(0.1)
                if proc.returncode:
                    detail = (work / "worker.log").read_text(errors="replace")[-6000:]
                    raise ValueError("Export failed: " + detail)
            finally:
                _stop_tree(proc)
        if cancel.is_set():
            raise InterruptedError("Export canceled before publication")
        result = json.loads((work / "result.json").read_text(encoding="utf-8"))
        staged = work / ("final" + suffix)
        _publish(staged, output)
        result.update(output=str(output), bytes=output.stat().st_size)
        callback(1.0, "Export verified and saved")
        return result


def _progress(work, value, message):
    temporary = work / "progress.tmp"
    temporary.write_text(json.dumps({"progress": min(0.99, max(0.0, value)), "message": message}), encoding="utf-8")
    os.replace(temporary, work / "progress.json")


def _worker(request):
    import av
    import numpy as np
    import torch
    from studio_engine import StudioRenderer, to_display_rgb
    global FFMPEG, FFPROBE
    FFMPEG, FFPROBE = request["ffmpeg"], request["ffprobe"]
    if not torch.cuda.is_available():
        raise ValueError("A working CUDA-enabled PyTorch GPU is required")
    source, work, mode = Path(request["source"]), Path(request["work"]), request["mode"]
    info, metadata = inspect_media(source), _probe(source)
    video = _video(metadata)
    warnings = list(info["warnings"])
    preview = mode == "preview"
    if preview:
        ratio = min(1.0, 1280 / max(info["width"], info["height"]))
        info["width"] = max(2, 2 * int(info["width"] * ratio / 2))
        info["height"] = max(2, 2 * int(info["height"] * ratio / 2))
    plan = _encoder_plan(info, mode, bool(request["settings"].get("_force_cpu_encoder", False)))
    renderer = StudioRenderer({k: v for k, v in request["settings"].items() if not k.startswith("_")}, transfer=info["transfer"])
    rate = Fraction(info["fps_exact"])
    temporary = work / "rendered.nut"
    expected = work / "timestamps.txt"
    sample_hash, frames, first_time, previous = hashlib.sha256(), 0, None, None
    # NUT preserves arbitrary rational time bases; Matroska intermediates would
    # quantize many source timestamps to milliseconds before final muxing.
    with av.open(str(source), options={"protocol_whitelist": "file,pipe", "format_whitelist": _DEMUXERS}) as input_container, \
            av.open(str(temporary), "w", format="nut") as output_container, expected.open("w", encoding="ascii") as timing:
        stream = input_container.streams[video["index"]]
        matrix = stream.codec_context.colorspace
        if matrix in (0, 2) and not av.VideoFormat(info["pixel_format"]).is_rgb:
            matrix = _matrix_fallback(video)
        source_range = stream.codec_context.color_range or (2 if info["pixel_format"].startswith("yuvj") or av.VideoFormat(info["pixel_format"]).is_rgb else 1)
        output_stream = output_container.add_stream(plan["encoder"], rate=rate)
        output_stream.width, output_stream.height = info["width"], info["height"]
        output_stream.pix_fmt, output_stream.options = plan["pixel"], plan["options"]
        output_stream.codec_context.max_b_frames = 0
        output_stream.codec_context.sample_aspect_ratio = Fraction(info["sample_aspect_ratio"])
        for attr in ("color_range", "color_trc", "color_primaries", "colorspace"):
            setattr(output_stream.codec_context, attr, getattr(stream.codec_context, attr))
        if mode == "lossless":
            output_stream.codec_context.color_range = 2
            output_stream.codec_context.colorspace = 0
        if preview:
            output_stream.codec_context.color_range = 1
            output_stream.codec_context.colorspace = 1
            output_stream.codec_context.color_primaries = 1
            output_stream.codec_context.color_trc = 13
        reformatter = av.video.reformatter.VideoReformatter()
        with torch.inference_mode():
            for decoded in input_container.decode(stream):
                if decoded.pts is None or decoded.time_base is None:
                    raise ValueError("Source has a frame without a presentation timestamp")
                timestamp = Fraction(decoded.pts) * decoded.time_base
                if first_time is None:
                    first_time = timestamp
                    output_stream.time_base = decoded.time_base
                    output_stream.codec_context.time_base = decoded.time_base
                relative = timestamp - first_time
                if previous is not None and timestamp <= previous:
                    raise ValueError("Source frame timestamps are not strictly increasing")
                previous = timestamp
                if preview and relative >= 3:
                    break
                rgb_frame = reformatter.reformat(decoded, width=info["width"], height=info["height"],
                    format="rgb48le", src_colorspace=matrix,
                    src_color_range=source_range, dst_color_range=2)
                rgb = rgb_frame.to_ndarray()
                tensor = torch.from_numpy(rgb.astype(np.float32)).to("cuda").permute(2, 0, 1) / 65535.0
                turns = info["rotation"] // 90
                if turns:
                    tensor = torch.rot90(tensor, turns, (1, 2))
                rendered = renderer.render(tensor, frames, float(relative))
                if preview:
                    rendered = to_display_rgb(rendered, info["transfer"], video.get("color_primaries", "unknown"))
                if turns:
                    rendered = torch.rot90(rendered, -turns, (1, 2))
                if rendered.shape != tensor.shape and not turns:
                    raise ValueError("Renderer changed frame dimensions")
                if not bool(torch.isfinite(rendered).all()):
                    raise ValueError("Renderer produced non-finite pixels")
                array = (rendered.clamp(0, 1).permute(1, 2, 0) * 65535).round().to(torch.int32).cpu().numpy().astype(np.uint16)
                if array.shape != (info["height"], info["width"], 3):
                    raise ValueError("Renderer changed frame dimensions")
                raw = av.VideoFrame.from_ndarray(array, format="rgb48le")
                if mode == "lossless":
                    sample_hash.update(array.tobytes())
                encoded = reformatter.reformat(raw, format=plan["pixel"], src_color_range=2,
                    dst_color_range=output_stream.codec_context.color_range or source_range,
                    dst_colorspace=output_stream.codec_context.colorspace if mode == "lossless" or preview else matrix)
                encoded.pts, encoded.time_base = decoded.pts - int(first_time / decoded.time_base), decoded.time_base
                encoded.duration = decoded.duration
                for packet in output_stream.encode(encoded):
                    output_container.mux(packet)
                timing.write(f"{relative.numerator}/{relative.denominator}\n")
                frames += 1
                if frames == 1 or frames % 10 == 0:
                    _progress(work, 0.8 * float(relative) / max(0.01, min(info["duration"], 3) if preview else info["duration"]), f"Rendering frame {frames}")
        for packet in output_stream.encode():
            output_container.mux(packet)
    if not frames:
        raise ValueError("Source decoded no frames")
    _progress(work, 0.82, "Restoring source audio and metadata")
    staged = work / ("final" + request["suffix"])
    audio = [s for s in metadata["streams"] if s.get("codec_type") == "audio"]
    audio_copy = bool(audio) and mode not in ("preview",)
    incompatible_mp4 = request["suffix"] == ".mp4" and any(s["codec_name"].startswith("pcm_") or s["codec_name"] in ("vorbis", "flac") for s in audio)
    if mode == "hevc" and incompatible_mp4:
        audio_copy = False
        warnings.append("MP4 cannot carry this source audio through this adapter; all audio tracks are converted to AAC.")
    args = [FFMPEG, "-hide_banner", "-v", "error", "-xerror", "-n", "-copyts", *_PROTOCOLS]
    if info["rotation"]:
        args += ["-display_rotation:v:0", str(info["rotation"])]
    args += ["-i", str(temporary), "-itsoffset", f"{-float(first_time):.12f}", *_PROTOCOLS, "-i", str(source),
            "-map", "0:v:0", "-map", "1:a?", "-map_metadata", "1", "-map_metadata:s:v:0", f"1:s:{video['index']}", "-map_chapters", "1", "-c:v", "copy",
            "-c:a", "copy" if audio_copy else "aac", "-avoid_negative_ts", "disabled"]
    if preview:
        args += ["-t", "3", "-map_metadata", "-1", "-map_chapters", "-1"]
    for key, option in (("color_range", "-color_range"), ("color_space", "-colorspace"),
                        ("color_transfer", "-color_trc"), ("color_primaries", "-color_primaries")):
        if not preview and mode != "lossless" and video.get(key) not in (None, "unknown", "unspecified"):
            args += [option, str(video[key])]
    if preview:
        args += ["-color_range", "tv", "-colorspace", "bt709", "-color_trc", "iec61966-2-1", "-color_primaries", "bt709"]
    elif mode == "lossless":
        args += ["-color_range", "pc", "-colorspace", "rgb"]
    if plan["codec"] == "hevc" and request["suffix"] in (".mp4", ".mov"):
        args += ["-tag:v", "hvc1"]
    if request["suffix"] in (".mp4", ".mov"):
        args += ["-movflags", "+faststart", "-video_track_timescale", str(Fraction(video["time_base"]).denominator)]
    _run(args + [str(staged)], 7200)
    _progress(work, 0.9, "Verifying timestamps, pixels and audio")
    saved_data = _probe(staged)
    saved = _video(saved_data)
    if saved["codec_name"] != plan["codec"] or (int(saved["width"]), int(saved["height"])) != (info["width"], info["height"]):
        raise ValueError("Encoded geometry or codec differs from the requested result")
    if mode == "match" and _pixel_signature(saved["pix_fmt"]) != _pixel_signature(info["pixel_format"]):
        raise ValueError("Match-source export changed source bit depth/chroma")
    if mode not in ("lossless", "preview"):
        for field in ("color_range", "color_space", "color_transfer", "color_primaries"):
            if video.get(field) not in (None, "unknown", "unspecified") and saved.get(field) != video[field]:
                raise ValueError(f"Export changed source {field}")
    if _rotation(saved) != info["rotation"]:
        raise ValueError("Final container did not retain source display rotation")
    if mode == "match" and _fraction(saved.get("sample_aspect_ratio", "1:1").replace(":", "/"), "1/1") != Fraction(info["sample_aspect_ratio"]):
        raise ValueError("Final container did not retain source pixel aspect ratio")
    actual_hash = hashlib.sha256()
    with av.open(str(staged), options={"protocol_whitelist": "file,pipe", "format_whitelist": _DEMUXERS}) as check, expected.open(encoding="ascii") as timing:
        checked = 0
        tolerance = Fraction(saved["time_base"]) / 2 + Fraction(1, 1000000)
        for frame, stamp in zip(check.decode(video=0), timing, strict=True):
            if frame.pts is None or abs(Fraction(frame.pts) * frame.time_base - Fraction(stamp.strip())) > tolerance:
                raise ValueError("Output presentation timing changed")
            if mode == "lossless":
                actual_hash.update(_active_bytes(frame))
            checked += 1
        if checked != frames:
            raise ValueError("Decoded output frame count differs from rendered frame count")
    if mode == "lossless" and actual_hash.digest() != sample_hash.digest():
        raise ValueError("Lossless RGB16 sample verification failed")
    saved_audio = [s for s in saved_data["streams"] if s.get("codec_type") == "audio"]
    if len(saved_audio) != len(audio):
        raise ValueError("Export dropped an audio track")
    if audio_copy:
        original_audio, final_audio = _audio_summary(source), _audio_summary(staged)
        if [item[:2] for item in original_audio] != [item[:2] for item in final_audio]:
            raise ValueError("Copied audio packet payloads changed")
        for original, final, source_stream, saved_stream in zip(original_audio, final_audio, audio, saved_audio, strict=True):
            tolerance = max(Fraction(source_stream["time_base"]), Fraction(saved_stream["time_base"]))
            if original[2] is not None and final[2] is not None and (
                abs(original[2] - first_time - final[2]) > tolerance or abs(original[3] - first_time - final[3]) > tolerance):
                raise ValueError("Copied audio alignment or duration changed")
    _run([FFMPEG, "-v", "error", "-xerror", *_PROTOCOLS, "-i", str(staged),
          "-map", "0:v:0", "-map", "0:a?", "-f", "null", "-"], 7200)
    if plan["quality"] == "lossless codec" and not plan["pixel"].startswith(("gbr", "rgb")):
        warnings.append("Codec encoding is lossless for processed target YUV samples; RGB-to-source chroma/bit-depth conversion is not mathematically lossless RGB.")
    result = {"width": info["width"], "height": info["height"], "frames": frames, "fps_exact": info["fps_exact"],
              "timestamps": "source presentation timestamps retained", "codec": plan["codec"], "encoder": plan["encoder"],
              "pixel_format": saved["pix_fmt"], "bit_depth": _depth(saved["pix_fmt"]), "container": request["suffix"].lstrip("."),
              "audio_streams": len(audio), "audio": "packet copy verified" if audio_copy else ("AAC" if audio else "none"),
              "encode_quality": plan["quality"], "transfer": info["transfer"], "warnings": warnings}
    (work / "result.json").write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("This module is a local application adapter, not a standalone CLI")
    try:
        _worker(json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")))
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
