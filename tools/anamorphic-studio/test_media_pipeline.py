"""Portable adapter contracts and actual CPU-render/codec integration tests.

python test_media_pipeline.py -v
Round trips run on CPU with software encoders, including on machines without a GPU.
"""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import media_pipeline as media


class Contracts(unittest.TestCase):
    def test_rotation_is_display_matrix_convention(self):
        self.assertEqual(media._rotation({"side_data_list": [{"rotation": -90}]}), 270)
        self.assertEqual(media._rotation({"tags": {"rotate": "180"}}), 180)
        with self.assertRaises(ValueError):
            media._rotation({"tags": {"rotate": "22"}})

    def test_new_file_publication_preserves_racing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "stage", Path(directory) / "output"
            a.write_bytes(b"new")
            b.write_bytes(b"foreign")
            with self.assertRaises(FileExistsError):
                media._publish(a, b)
            self.assertEqual(b.read_bytes(), b"foreign")

    def test_existing_hardlink_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "stage", Path(directory) / "output"
            a.write_bytes(b"original")
            os.link(a, b)
            with self.assertRaises(FileExistsError):
                media._new_output(b)
            self.assertEqual(a.read_bytes(), b"original")

    def test_encoder_match_preserves_odd_geometry_with_cpu(self):
        info = dict(codec="h264", pixel_format="yuv444p", width=321, height=181, bit_depth=8)
        plan = media._encoder_plan(info, "match", force_cpu=True, device_type="cpu")
        self.assertEqual(plan["encoder"], "libx264")
        self.assertEqual(plan["pixel"], "yuv444p")
        self.assertEqual(plan["quality"], "lossless codec")

    def test_unsupported_match_does_not_silently_change_codec(self):
        info = dict(codec="not_a_codec", pixel_format="yuv420p", width=320, height=180, bit_depth=8)
        with self.assertRaisesRegex(ValueError, "explicitly"):
            media._encoder_plan(info, "match", force_cpu=True, device_type="cpu")

    def test_alpha_match_fails_before_dropping_alpha(self):
        info = dict(codec="prores", pixel_format="yuva444p10le", width=320, height=180, bit_depth=10)
        with self.assertRaisesRegex(ValueError, "alpha"):
            media._encoder_plan(info, "match", force_cpu=True, device_type="cpu")

    def test_hevc422_uses_exact_format_cpu_fallback(self):
        info = dict(codec="hevc", pixel_format="yuv422p10le", width=320, height=180, bit_depth=10)
        plan = media._encoder_plan(info, "match", force_cpu=True, device_type="cpu")
        self.assertEqual((plan["encoder"], plan["pixel"]), ("libx265", "yuv422p10le"))

    def test_lossless_is_rgb16_not_subsampled_yuv(self):
        info = dict(codec="h264", pixel_format="yuv420p", width=320, height=180, bit_depth=8)
        plan = media._encoder_plan(info, "lossless", force_cpu=True, device_type="cpu")
        self.assertEqual((plan["codec"], plan["pixel"]), ("ffv1", "gbrp16le"))

    def test_apple_match_never_selects_unproven_lossless_hardware(self):
        info = dict(codec="hevc", pixel_format="yuv420p10le", width=320, height=180, bit_depth=10)
        with mock.patch.object(media.platform, "system", return_value="Darwin"), \
                mock.patch.object(media, "_hardware_works") as probe:
            plan = media._encoder_plan(info, "match", device_type="mps")
        self.assertEqual((plan["encoder"], plan["pixel"], plan["quality"]),
                         ("libx265", "yuv420p10le", "lossless codec"))
        self.assertIn("lossless=1", plan["options"]["x265-params"])
        probe.assert_not_called()

    def test_apple_delivery_uses_probed_10bit_videotoolbox(self):
        import av
        info = dict(codec="hevc", pixel_format="yuv420p10le", width=320, height=180, bit_depth=10)
        with mock.patch.object(media.platform, "system", return_value="Darwin"), \
                mock.patch.object(media, "_hardware_works", return_value=True) as probe, \
                mock.patch.object(media, "_encoder_formats", return_value={"p010le"}), \
                mock.patch.object(av, "codecs_available", av.codecs_available | {"hevc_videotoolbox"}):
            plan = media._encoder_plan(info, "hevc", device_type="mps")
        self.assertEqual((plan["encoder"], plan["pixel"], plan["quality"]),
                         ("hevc_videotoolbox", "p010le", "highest quality"))
        self.assertEqual(plan["options"]["global_quality"], "75")
        self.assertEqual(plan["options"]["allow_sw"], "0")
        probe.assert_called_once()

    def test_apple_unavailable_hardware_keeps_10bit_software_delivery(self):
        info = dict(codec="hevc", pixel_format="yuv420p10le", width=320, height=180, bit_depth=10)
        with mock.patch.object(media.platform, "system", return_value="Darwin"), \
                mock.patch.object(media, "_hardware_works", return_value=False):
            plan = media._encoder_plan(info, "hevc", device_type="mps")
        self.assertEqual((plan["encoder"], plan["pixel"]), ("libx265", "yuv420p10le"))
        self.assertTrue(any("VideoToolbox" in warning for warning in plan["warnings"]))

    def test_nvidia_lossless_options_unchanged(self):
        import av
        info = dict(codec="h264", pixel_format="yuv420p", width=320, height=180, bit_depth=8)
        with mock.patch.object(media, "_hardware_works", return_value=True), \
                mock.patch.object(media, "_encoder_formats", return_value={"yuv420p"}), \
                mock.patch.object(av, "codecs_available", av.codecs_available | {"h264_nvenc"}):
            plan = media._encoder_plan(info, "match", device_type="cuda")
        self.assertEqual(plan["encoder"], "h264_nvenc")
        self.assertEqual(plan["options"], {"preset": "p3", "tune": "lossless", "rc": "constqp", "qp": "0"})

    def test_cpu_does_not_attempt_nvidia_codec_present_in_wheel(self):
        info = dict(codec="h264", pixel_format="yuv420p", width=320, height=180, bit_depth=8)
        with mock.patch.object(media.platform, "system", return_value="Linux"), \
                mock.patch.object(media, "_hardware_works") as probe:
            plan = media._encoder_plan(info, "match", device_type="cpu")
        self.assertEqual(plan["encoder"], "libx264")
        probe.assert_not_called()

    def test_ffprobe_cleanup_does_not_kill_inherited_group(self):
        process = mock.Mock(pid=123456)
        process.poll.return_value = None
        posix = SimpleNamespace(name="posix", getpgid=mock.Mock(return_value=888), killpg=mock.Mock())
        with mock.patch.object(media, "os", posix):
            media._stop_tree(process)
        process.kill.assert_called_once()
        posix.killpg.assert_not_called()

    def test_owned_session_worker_cleanup_kills_its_group(self):
        process = mock.Mock(pid=123456)
        process.poll.return_value = None
        posix = SimpleNamespace(name="posix", getpgid=mock.Mock(return_value=123456), killpg=mock.Mock())
        with mock.patch.object(media, "os", posix), mock.patch.object(media, "signal", SimpleNamespace(SIGKILL=9)):
            media._stop_tree(process)
        posix.killpg.assert_called_once_with(123456, 9)
        process.kill.assert_not_called()


@unittest.skipUnless(shutil.which(media.FFMPEG) and shutil.which(media.FFPROBE), "FFmpeg required")
class RealMedia(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.environment = mock.patch.dict(os.environ, {"STUDIO_DEVICE": "cpu"})
        cls.environment.start()
        cls.temporary = tempfile.TemporaryDirectory(prefix="studio-media-test-")
        cls.folder = Path(cls.temporary.name)
        cls.source = cls.folder / "source.mov"
        subprocess.run([media.FFMPEG, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30000/1001",
                        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=48000",
                        "-t", "0.2", "-map", "0:v", "-map", "1:a", "-map", "2:a",
                        # Set frame properties and encoded SPS explicitly. The
                        # Homebrew FFmpeg 9 fixture lost generic output-only
                        # color options, unlike the Windows FFmpeg 7 build.
                        "-vf", "setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-color_trc", "bt709", "-color_primaries", "bt709", "-colorspace", "bt709",
                        "-bsf:v", "h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1",
                        "-c:a", "pcm_s16le", str(cls.source)], check=True)
        cls.source_hash = hashlib.sha256(cls.source.read_bytes()).hexdigest()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        cls.environment.stop()

    def export_cpu(self, source, output, settings, **kwargs):
        return media.export_video(source, output, {**settings, "_force_cpu_encoder": True}, **kwargs)

    def output(self, name):
        return self.folder / (self.id().split(".")[-1] + name)

    def test_inspect_retains_rational_rate_and_all_audio_count(self):
        info = media.inspect_media(self.source)
        self.assertEqual(info["fps_exact"], "30000/1001")
        self.assertEqual((info["width"], info["height"], info["bit_depth"]), (320, 180, 8))
        self.assertEqual(info["audio_streams"], 2)
        self.assertEqual(info["color_primaries"], "bt709")

    def test_fixture_encoded_bitstream_contains_bt709_tags(self):
        """Check SPS tags independently of inspect_media and MOV color atoms."""
        import av
        bitstream = self.output("-fixture.h264")
        subprocess.run([media.FFMPEG, "-v", "error", "-n", "-i", str(self.source),
                        "-map", "0:v:0", "-c:v", "copy", "-bsf:v", "h264_mp4toannexb",
                        str(bitstream)], check=True, timeout=30)
        with av.open(str(bitstream), format="h264") as container:
            frame = next(container.decode(video=0))
            self.assertEqual((frame.color_primaries, frame.color_trc, frame.colorspace), (1, 1, 1))

    def test_proxy_and_poster_decode(self):
        output, poster = self.output(".mp4"), self.output(".jpg")
        result = media.make_proxy(self.source, output, poster)
        self.assertEqual(media.inspect_media(output)["codec"], "h264")
        self.assertGreater(poster.stat().st_size, 0)
        self.assertEqual(result["proxy"], str(output))
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.source_hash)

    def test_copy_audio_hash_is_payload_sensitive(self):
        hashes = media._audio_hashes(self.source)
        self.assertEqual(len(hashes), 2)
        self.assertNotEqual(hashes[0][0], hashes[1][0])
        self.assertEqual(hashes, media._audio_hashes(self.source))

    def test_playlist_demuxers_are_refused(self):
        manifest = self.output(".mov")
        manifest.write_text("#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\nsource.mov\n#EXT-X-ENDLIST\n", encoding="ascii")
        with self.assertRaises(ValueError):
            media.inspect_media(manifest)

    def test_common_container_inspection(self):
        for suffix, codec in ((".mp4", "libx264"), (".mkv", "ffv1"), (".avi", "mpeg4"), (".webm", "libvpx-vp9")):
            with self.subTest(container=suffix):
                source = self.output(suffix)
                subprocess.run([media.FFMPEG, "-v", "error", "-i", str(self.source), "-map", "0:v:0", "-frames:v", "2",
                                "-an", "-c:v", codec, str(source)], check=True)
                info = media.inspect_media(source)
                self.assertEqual((info["width"], info["height"]), (320, 180))

    def test_canceled_before_worker_has_no_output(self):
        event = threading.Event()
        event.set()
        output = self.output(".mov")
        with self.assertRaises(InterruptedError):
            self.export_cpu(self.source, output, {}, cancel_event=event)
        self.assertFalse(output.exists())

    def test_match_container_change_requires_explicit_mode(self):
        output = self.output(".mp4")
        with self.assertRaisesRegex(ValueError, "requires .mov"):
            self.export_cpu(self.source, output, {}, mode="match")
        self.assertFalse(output.exists())

    def test_worker_failure_never_publishes(self):
        output = self.output(".mov")
        with self.assertRaises(ValueError):
            self.export_cpu(self.source, output, {"invalid_setting": True})
        self.assertFalse(output.exists())

    def test_running_worker_cancel_stops_owned_process(self):
        original = subprocess.Popen
        owned = []
        def sleeping_worker(*args, **kwargs):
            if args[0][0] != sys.executable:
                return original(*args, **kwargs)
            process = original([sys.executable, "-c", "import time;time.sleep(60)"], **kwargs)
            owned.append(process)
            return process
        event = threading.Event()
        timer = threading.Timer(0.4, event.set)
        output = self.output(".mov")
        with mock.patch.object(media.subprocess, "Popen", side_effect=sleeping_worker), \
             mock.patch.object(media, "inspect_media", return_value={}):
            timer.start()
            try:
                with self.assertRaises(InterruptedError):
                    self.export_cpu(self.source, output, {}, cancel_event=event)
            finally:
                timer.cancel()
        self.assertIsNotNone(owned[0].poll())
        self.assertFalse(output.exists())

    def test_worker_deadline_stops_owned_process(self):
        original = subprocess.Popen
        owned = []
        def sleeping_worker(*args, **kwargs):
            if args[0][0] != sys.executable:
                return original(*args, **kwargs)
            process = original([sys.executable, "-c", "import time;time.sleep(60)"], **kwargs)
            owned.append(process)
            return process
        output = self.output(".mov")
        with mock.patch.object(media.subprocess, "Popen", side_effect=sleeping_worker), \
             mock.patch.object(media, "inspect_media", return_value={}):
            with self.assertRaises(TimeoutError):
                self.export_cpu(self.source, output, {"_timeout_seconds": 1})
        self.assertIsNotNone(owned[0].poll())
        self.assertFalse(output.exists())

    def test_export_rotation_is_retained(self):
        rotated = self.output("-rotated.mov")
        subprocess.run([media.FFMPEG, "-v", "error", "-display_rotation:v:0", "90", "-i", str(self.source), "-map", "0", "-c", "copy",
                        str(rotated)], check=True)
        before = media.inspect_media(rotated)
        self.assertEqual(before["rotation"], 90)
        output = self.output("-rotated-result.mov")
        self.export_cpu(rotated, output, {"amount": 0.2})
        after = media.inspect_media(output)
        self.assertEqual((after["rotation"], after["width"], after["height"]), (90, 320, 180))

    def test_export_hdr10_transfer_primaries_matrix_depth_retained(self):
        import av
        import numpy as np
        from fractions import Fraction
        source = self.output("-hdr.mov")
        time_base = Fraction(1, 30)
        packet_times = []
        with av.open(str(source), "w") as container:
            stream = container.add_stream("libx265", rate=30)
            stream.width, stream.height, stream.pix_fmt = 320, 180, "yuv420p10le"
            # A two-frame color fixture needs no frame reordering. Leaving
            # libx265's B-frame defaults/timestamps inferred produced sporadic
            # huge negative DTS values, including on the Windows build.
            stream.time_base = stream.codec_context.time_base = time_base
            stream.codec_context.max_b_frames = 0
            stream.options = {"x265-params": "log-level=error:bframes=0"}
            stream.codec_context.color_trc = 16
            stream.codec_context.color_primaries = 9
            stream.codec_context.colorspace = 9
            stream.codec_context.color_range = 1
            def mux_checked(packet):
                self.assertIsNotNone(packet.pts)
                self.assertIsNotNone(packet.dts)
                times = (packet.pts * packet.time_base, packet.dts * packet.time_base)
                point = len(packet_times) * time_base
                self.assertEqual(times, (point, point), "HDR fixture must have explicit ordered PTS/DTS")
                packet_times.append(times)
                container.mux(packet)
            for index in range(2):
                frame = av.VideoFrame.from_ndarray(np.full((180, 320, 3), index * 20000, dtype=np.uint16), format="rgb48le")
                frame.pts, frame.time_base, frame.duration = index, time_base, 1
                for packet in stream.encode(frame):
                    mux_checked(packet)
            for packet in stream.encode():
                mux_checked(packet)
        self.assertEqual(len(packet_times), 2)
        with av.open(str(source)) as container:
            self.assertEqual([frame.pts * frame.time_base for frame in container.decode(video=0)], [0, time_base])
        output = self.output("-hdr-result.mov")
        self.export_cpu(source, output, {"amount": 0.2})
        video = media._video(media._probe(output))
        self.assertEqual((video["color_transfer"], video["color_primaries"], video["color_space"]),
                         ("smpte2084", "bt2020", "bt2020nc"))
        self.assertEqual(media.inspect_media(output)["bit_depth"], 10)

    def test_export_match_two_audio_tracks_and_source_unchanged(self):
        output = self.output(".mov")
        result = self.export_cpu(self.source, output, {"amount": 0}, mode="match")
        self.assertEqual(result["frames"], 6)
        self.assertEqual(result["audio"], "packet copy verified")
        self.assertEqual(result["encode_quality"], "lossless codec")
        self.assertEqual(result["backend"]["device"], "cpu")
        self.assertEqual(result["encoder"], "libx264")
        self.assertEqual(media._audio_hashes(output), media._audio_hashes(self.source))
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.source_hash)

    def test_export_rgb16_lossless_verifies_every_processed_sample(self):
        result = self.export_cpu(self.source, self.output(".mkv"), {"amount": 0.3}, mode="lossless")
        self.assertEqual((result["pixel_format"], result["bit_depth"]), ("gbrp16le", 16))
        self.assertEqual(result["encode_quality"], "lossless RGB16")

    def test_export_hevc_conversion_reports_audio_conversion(self):
        result = self.export_cpu(self.source, self.output(".mp4"), {"amount": 0}, mode="hevc")
        self.assertEqual(result["audio_streams"], 2)
        self.assertEqual(result["audio"], "AAC")
        self.assertTrue(any("AAC" in message for message in result["warnings"]))

    @unittest.skipUnless(sys.platform == "darwin", "Actual VideoToolbox integration requires macOS")
    def test_actual_mac_videotoolbox_hevc_8_and_10bit(self):
        """Exercise real VT only when this PyAV build and Mac can encode it."""
        import av
        from fractions import Fraction
        original_info = media.inspect_media(self.source)
        plans = {}
        for bits in (8, 10):
            info = {**original_info, "bit_depth": bits,
                    "pixel_format": "yuv420p10le" if bits == 10 else "yuv420p"}
            plans[bits] = media._encoder_plan(info, "hevc", device_type="cpu")
        if not any(plan["encoder"] == "hevc_videotoolbox" for plan in plans.values()):
            self.skipTest("Installed PyAV/macOS hardware cannot provide HEVC VideoToolbox at 320x180 in either 8 or 10 bit")
        for bits in (8, 10):
            with self.subTest(bit_depth=bits):
                if plans[bits]["encoder"] != "hevc_videotoolbox":
                    self.skipTest(f"HEVC VideoToolbox {bits}-bit hardware session is unavailable")
                source = self.source
                if bits == 10:
                    source = self.output("-vt-source-10bit.mov")
                    subprocess.run([media.FFMPEG, "-v", "error", "-n", "-i", str(self.source),
                                    "-map", "0", "-c:v", "libx265", "-pix_fmt", "yuv420p10le",
                                    "-x265-params", "log-level=error:pools=2", "-c:a", "copy", str(source)], check=True)
                before_hash = hashlib.sha256(source.read_bytes()).hexdigest()
                output = self.output(f"-vt-{bits}bit.mp4")
                # Deliberately bypass export_cpu: effects stay on CPU through
                # STUDIO_DEVICE, but delivery must select real Apple hardware.
                result = media.export_video(source, output, {"amount": 0.2}, mode="hevc")
                self.assertEqual(result["encoder"], "hevc_videotoolbox")
                self.assertEqual(result["backend"]["device"], "cpu")
                self.assertEqual((result["width"], result["height"], result["frames"], result["bit_depth"]),
                                 (320, 180, 6, bits))
                self.assertEqual(result["audio_streams"], 2)
                self.assertEqual(media.inspect_media(output)["codec"], "hevc")
                def decoded_times(path):
                    with av.open(str(path)) as container:
                        return [Fraction(frame.pts) * frame.time_base for frame in container.decode(video=0)]
                self.assertEqual(decoded_times(source), decoded_times(output))
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before_hash)

    @unittest.skipUnless(sys.platform == "darwin", "Actual Metal end-to-end integration requires macOS")
    def test_actual_mac_metal_video_export(self):
        import av
        import torch
        from fractions import Fraction
        if not torch.backends.mps.is_available():
            self.skipTest("This Mac runner has no available Metal MPS device")
        output = self.output("-metal.mp4")
        # Do not force a software encoder: this is the normal Mac pipeline,
        # Metal effects plus probed VideoToolbox or its software fallback.
        with mock.patch.dict(os.environ, {"STUDIO_DEVICE": "mps"}):
            result = media.export_video(self.source, output, {"amount": 0.2}, mode="hevc")
        self.assertEqual(result["backend"]["device"], "mps")
        self.assertTrue(result["backend"]["accelerated"])
        self.assertIn(result["encoder"], ("hevc_videotoolbox", "libx265"))
        self.assertEqual((result["codec"], result["width"], result["height"], result["frames"], result["bit_depth"]),
                         ("hevc", 320, 180, 6, 8))
        self.assertEqual(result["fps_exact"], "30000/1001")
        self.assertEqual((result["audio_streams"], result["audio"]), (2, "AAC"))
        def decoded_times(path):
            with av.open(str(path)) as container:
                return [Fraction(frame.pts) * frame.time_base for frame in container.decode(video=0)]
        self.assertEqual(decoded_times(self.source), decoded_times(output))
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.source_hash)

    def test_export_vfr_keeps_presentation_times(self):
        import av
        import numpy as np
        from fractions import Fraction
        source = self.output("-vfr.mov")
        stamps = [0, 33, 91, 125, 201]
        with av.open(str(source), "w") as container:
            stream = container.add_stream("libx264", rate=30)
            stream.width, stream.height, stream.pix_fmt = 320, 180, "yuv420p"
            stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
            for index, pts in enumerate(stamps):
                frame = av.VideoFrame.from_ndarray(np.full((180, 320, 3), index * 30, dtype=np.uint8), format="rgb24")
                frame.pts, frame.time_base = pts, Fraction(1, 1000)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        output = self.output("-vfr-output.mov")
        result = self.export_cpu(source, output, {"amount": 0}, mode="match")
        def times(path):
            with av.open(str(path)) as container:
                return [Fraction(f.pts) * f.time_base for f in container.decode(video=0)]
        self.assertEqual(times(source), times(output))
        self.assertEqual(result["frames"], 5)


if __name__ == "__main__":
    unittest.main()
