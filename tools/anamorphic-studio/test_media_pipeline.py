"""CPU adapter contracts plus opt-in actual CUDA/codec integration tests.

python test_media_pipeline.py -v
STUDIO_GPU_TESTS=1 enables small, real source-timing/audio/RGB16 round trips.
"""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
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
        plan = media._encoder_plan(info, "match")
        self.assertEqual(plan["encoder"], "libx264")
        self.assertEqual(plan["pixel"], "yuv444p")
        self.assertEqual(plan["quality"], "lossless codec")

    def test_unsupported_match_does_not_silently_change_codec(self):
        info = dict(codec="not_a_codec", pixel_format="yuv420p", width=320, height=180, bit_depth=8)
        with self.assertRaisesRegex(ValueError, "explicitly"):
            media._encoder_plan(info, "match")

    def test_alpha_match_fails_before_dropping_alpha(self):
        info = dict(codec="prores", pixel_format="yuva444p10le", width=320, height=180, bit_depth=10)
        with self.assertRaisesRegex(ValueError, "alpha"):
            media._encoder_plan(info, "match")

    def test_hevc422_uses_exact_format_cpu_fallback(self):
        info = dict(codec="hevc", pixel_format="yuv422p10le", width=320, height=180, bit_depth=10)
        plan = media._encoder_plan(info, "match")
        self.assertEqual((plan["encoder"], plan["pixel"]), ("libx265", "yuv422p10le"))

    def test_lossless_is_rgb16_not_subsampled_yuv(self):
        info = dict(codec="h264", pixel_format="yuv420p", width=320, height=180, bit_depth=8)
        plan = media._encoder_plan(info, "lossless")
        self.assertEqual((plan["codec"], plan["pixel"]), ("ffv1", "gbrp16le"))


@unittest.skipUnless(shutil.which(media.FFMPEG) and shutil.which(media.FFPROBE), "FFmpeg required")
class RealMedia(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="studio-media-test-")
        cls.folder = Path(cls.temporary.name)
        cls.source = cls.folder / "source.mov"
        subprocess.run([media.FFMPEG, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30000/1001",
                        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=48000",
                        "-t", "0.2", "-map", "0:v", "-map", "1:a", "-map", "2:a", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-color_trc", "bt709", "-color_primaries", "bt709", "-colorspace", "bt709", "-c:a", "pcm_s16le", str(cls.source)], check=True)
        cls.source_hash = hashlib.sha256(cls.source.read_bytes()).hexdigest()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def output(self, name):
        return self.folder / (self.id().split(".")[-1] + name)

    def test_inspect_retains_rational_rate_and_all_audio_count(self):
        info = media.inspect_media(self.source)
        self.assertEqual(info["fps_exact"], "30000/1001")
        self.assertEqual((info["width"], info["height"], info["bit_depth"]), (320, 180, 8))
        self.assertEqual(info["audio_streams"], 2)
        self.assertEqual(info["color_primaries"], "bt709")

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
            media.export_video(self.source, output, {}, cancel_event=event)
        self.assertFalse(output.exists())

    def test_match_container_change_requires_explicit_mode(self):
        output = self.output(".mp4")
        with self.assertRaisesRegex(ValueError, "requires .mov"):
            media.export_video(self.source, output, {}, mode="match")
        self.assertFalse(output.exists())

    def test_worker_failure_never_publishes(self):
        output = self.output(".mov")
        with self.assertRaises(ValueError):
            media.export_video(self.source, output, {"invalid_setting": True})
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
                    media.export_video(self.source, output, {}, cancel_event=event)
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
                media.export_video(self.source, output, {"_timeout_seconds": 1})
        self.assertIsNotNone(owned[0].poll())
        self.assertFalse(output.exists())

    @unittest.skipUnless(os.environ.get("STUDIO_GPU_TESTS") == "1", "Set STUDIO_GPU_TESTS=1 for CUDA round trips")
    def test_gpu_rotation_is_retained(self):
        rotated = self.output("-rotated.mov")
        subprocess.run([media.FFMPEG, "-v", "error", "-display_rotation:v:0", "90", "-i", str(self.source), "-map", "0", "-c", "copy",
                        str(rotated)], check=True)
        before = media.inspect_media(rotated)
        self.assertEqual(before["rotation"], 90)
        output = self.output("-rotated-result.mov")
        media.export_video(rotated, output, {"amount": 0.2})
        after = media.inspect_media(output)
        self.assertEqual((after["rotation"], after["width"], after["height"]), (90, 320, 180))

    @unittest.skipUnless(os.environ.get("STUDIO_GPU_TESTS") == "1", "Set STUDIO_GPU_TESTS=1 for CUDA round trips")
    def test_gpu_hdr10_transfer_primaries_matrix_depth_retained(self):
        import av
        import numpy as np
        source = self.output("-hdr.mov")
        with av.open(str(source), "w") as container:
            stream = container.add_stream("libx265", rate=30)
            stream.width, stream.height, stream.pix_fmt = 320, 180, "yuv420p10le"
            stream.options = {"x265-params": "log-level=error"}
            stream.codec_context.color_trc = 16
            stream.codec_context.color_primaries = 9
            stream.codec_context.colorspace = 9
            stream.codec_context.color_range = 1
            for index in range(2):
                frame = av.VideoFrame.from_ndarray(np.full((180, 320, 3), index * 20000, dtype=np.uint16), format="rgb48le")
                frame.pts = index
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        output = self.output("-hdr-result.mov")
        media.export_video(source, output, {"amount": 0.2})
        video = media._video(media._probe(output))
        self.assertEqual((video["color_transfer"], video["color_primaries"], video["color_space"]),
                         ("smpte2084", "bt2020", "bt2020nc"))
        self.assertEqual(media.inspect_media(output)["bit_depth"], 10)

    @unittest.skipUnless(os.environ.get("STUDIO_GPU_TESTS") == "1", "Set STUDIO_GPU_TESTS=1 for CUDA round trips")
    def test_gpu_match_two_audio_tracks_and_source_unchanged(self):
        output = self.output(".mov")
        result = media.export_video(self.source, output, {"amount": 0}, mode="match")
        self.assertEqual(result["frames"], 6)
        self.assertEqual(result["audio"], "packet copy verified")
        self.assertEqual(result["encode_quality"], "lossless codec")
        self.assertEqual(media._audio_hashes(output), media._audio_hashes(self.source))
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.source_hash)

    @unittest.skipUnless(os.environ.get("STUDIO_GPU_TESTS") == "1", "Set STUDIO_GPU_TESTS=1 for CUDA round trips")
    def test_gpu_rgb16_lossless_verifies_every_processed_sample(self):
        result = media.export_video(self.source, self.output(".mkv"), {"amount": 0.3}, mode="lossless")
        self.assertEqual((result["pixel_format"], result["bit_depth"]), ("gbrp16le", 16))
        self.assertEqual(result["encode_quality"], "lossless RGB16")

    @unittest.skipUnless(os.environ.get("STUDIO_GPU_TESTS") == "1", "Set STUDIO_GPU_TESTS=1 for CUDA round trips")
    def test_gpu_hevc_conversion_reports_audio_conversion(self):
        result = media.export_video(self.source, self.output(".mp4"), {"amount": 0}, mode="hevc")
        self.assertEqual(result["audio_streams"], 2)
        self.assertEqual(result["audio"], "AAC")
        self.assertTrue(any("AAC" in message for message in result["warnings"]))

    @unittest.skipUnless(os.environ.get("STUDIO_GPU_TESTS") == "1", "Set STUDIO_GPU_TESTS=1 for CUDA round trips")
    def test_gpu_vfr_keeps_presentation_times(self):
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
        result = media.export_video(source, output, {"amount": 0}, mode="match")
        def times(path):
            with av.open(str(path)) as container:
                return [Fraction(f.pts) * f.time_base for f in container.decode(video=0)]
        self.assertEqual(times(source), times(output))
        self.assertEqual(result["frames"], 5)


if __name__ == "__main__":
    unittest.main()
