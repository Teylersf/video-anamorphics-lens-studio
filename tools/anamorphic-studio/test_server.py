"""CPU-only API/ownership tests; no codec, GPU, original media or network access.

Run with .venv-studio/Scripts/python.exe tools/anamorphic-studio/test_server.py.
The real routes, Pydantic validation and temporary file publication are exercised;
only media inspection/proxy generation and the background executor are mocked.
"""
from __future__ import annotations

from concurrent.futures import Future
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
import server

CLIENT_HEADER = {"X-Studio-Client": "anamorphic-studio"}


class PendingExecutor:
    def __init__(self):
        self.pending = []
        self.closed = False

    def submit(self, function, *args):
        future = Future()
        self.pending.append((function, args, future))
        return future

    def shutdown(self, **kwargs):
        self.closed = True


class FakePipeline:
    def __init__(self):
        self.inspected = []
        self.proxies = []
        self.exports = []
        self.inspect_error = None
        self.geometry = (640, 360)

    def inspect_media(self, path):
        self.inspected.append((Path(path), Path(path).read_bytes()))
        if self.inspect_error:
            raise ValueError(self.inspect_error)
        return {"width": self.geometry[0], "height": self.geometry[1],
                "display_width": self.geometry[0], "display_height": self.geometry[1],
                "duration": 3., "fps": 30., "fps_exact": "30", "transfer": "bt709",
                "color_primaries": "unknown", "codec": "h264", "bit_depth": 8}

    def make_proxy(self, original, proxy, poster):
        self.proxies.append((Path(original), Path(proxy), Path(poster)))
        Path(proxy).write_bytes(b"owned-proxy")
        Path(poster).write_bytes(b"owned-poster")

    def export_video(self, *args, **kwargs):
        self.exports.append((args, kwargs))
        raise AssertionError("No actual export may run in CPU API tests")


class ReleaseGateLock:
    """Run a competing completion at the first outermost lock release.

    A real completion needs STATE_LOCK too. Its earliest legal point is exactly
    after this release. If cancel holds the transaction lock, its event has been
    set by then; if it only copied status under lock, completion wins first.
    """
    def __init__(self, callback):
        self.lock = threading.RLock()
        self.depth = 0
        self.armed = True
        self.callback = callback

    def __enter__(self):
        self.lock.acquire()
        self.depth += 1
        return self

    def __exit__(self, *args):
        self.depth -= 1
        outermost = self.depth == 0
        self.lock.release()
        if outermost and self.armed:
            self.armed = False
            self.callback()


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="studio-api-test-")
        self.store = Path(self.temp.name)
        self.pipeline = FakePipeline()
        self.executor = PendingExecutor()
        self.patches = [patch.object(server, "STORE", self.store),
                        patch.object(server, "MEDIA", {}), patch.object(server, "JOBS", {}),
                        patch.object(server, "CANCEL", {}),
                        patch.object(server, "GPU_LOCK", threading.Lock()),
                        patch.object(server, "STATE_LOCK", threading.RLock()),
                        patch.object(server, "EXECUTOR", self.executor),
                        patch.object(server, "SERVER", None),
                        patch.object(server, "pipeline", return_value=self.pipeline)]
        for item in self.patches:
            item.start()
        self.client = TestClient(server.app, raise_server_exceptions=False)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def upload(self, name="clip.mov", content=b"synthetic-container-bytes"):
        response = self.client.post("/api/media", headers=CLIENT_HEADER,
                                    files={"file": (name, content, "video/quicktime")})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def queue(self, media=None, **options):
        media = media or self.upload()
        response = self.client.post("/api/jobs", headers=CLIENT_HEADER,
            json={"media_id": media["id"], "settings": {}, **options})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_status_cpu_ready_and_explicit_unavailable_device(self):
        import runtime_backend
        with patch.dict('os.environ', {'STUDIO_DEVICE': 'cpu'}), patch.object(server.shutil, 'which', return_value='/tool'):
            response = self.client.get('/api/status')
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()['ready'])
            self.assertEqual(response.json()['device'], 'cpu')
            self.assertFalse(response.json()['accelerated'])
        with patch.object(runtime_backend, 'backend_info', side_effect=RuntimeError('Metal unavailable')):
            response = self.client.get('/api/status')
            self.assertEqual(response.status_code, 200)
            self.assertFalse(response.json()['ready'])
            self.assertEqual(response.json()['reason'], 'Metal unavailable')

    def test_actual_cpu_preview_resizes_and_publishes_jpegs(self):
        self._actual_preview('cpu')

    def test_actual_mps_preview_resizes_and_publishes_jpegs(self):
        import torch
        if not torch.backends.mps.is_available():
            self.skipTest('Actual Metal hardware is unavailable')
        self._actual_preview('mps')

    def _actual_preview(self, device):
        import numpy as np
        from PIL import Image
        media = self.upload()
        pixels = np.zeros((360, 640, 3), dtype=np.uint16)
        pixels[100:150, 250:350] = 65535
        with patch.dict('os.environ', {'STUDIO_DEVICE': device}), patch.object(
                server, 'frame_at', return_value=(pixels, .5)):
            response = self.client.post('/api/preview', headers=CLIENT_HEADER,
                json={'media_id': media['id'], 'time': .5, 'width': 320})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result['backend']['device'], device)
        self.assertEqual((result['width'], result['height']), (320, 180))
        for name in ('original_url', 'processed_url'):
            asset = self.client.get(result[name])
            self.assertEqual(asset.status_code, 200)
            import io
            with Image.open(io.BytesIO(asset.content)) as image:
                self.assertEqual(image.size, (320, 180))
        self.assertFalse(server.GPU_LOCK.locked())

    def test_preview_device_error_releases_renderer(self):
        import runtime_backend
        media = self.upload()
        with patch.object(runtime_backend, 'select_device', side_effect=RuntimeError('Metal unavailable')):
            response = self.client.post('/api/preview', headers=CLIENT_HEADER,
                json={'media_id': media['id']})
        self.assertEqual(response.status_code, 422, response.text)
        self.assertIn('Metal unavailable', response.text)
        self.assertFalse(server.GPU_LOCK.locked())

    def test_post_header_protection_and_browser_origin(self):
        for path, body in (("/api/media", None), ("/api/preview", {}),
                           ("/api/jobs", {}), ("/api/jobs/missing/cancel", {}),
                           ("/api/shutdown", {})):
            for headers in ({}, {"X-Studio-Client": "wrong"}):
                response = self.client.post(path, headers=headers, json=body)
                self.assertEqual(response.status_code, 403, (path, response.text))
        self.assertEqual(self.pipeline.inspected, [])
        self.assertEqual(server.MEDIA, {})
        denied = self.client.options("/api/jobs", headers={"Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "X-Studio-Client"})
        self.assertEqual(denied.status_code, 400)
        allowed = self.client.options("/api/jobs", headers={"Origin": "http://localhost:8766",
            "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "X-Studio-Client"})
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(allowed.headers["access-control-allow-origin"], "http://localhost:8766")
        self.assertEqual(self.client.get("/api/config", headers={"Host": "external.example"}).status_code, 400)

    def test_upload_basename_new_owned_bytes_and_duplicate_names(self):
        sentinel = self.store / "keep-personal-file.mov"
        sentinel.write_bytes(b"must-remain-identical")
        digest = hashlib.sha256(sentinel.read_bytes()).hexdigest()
        first = self.upload(r"C:\\private\\..\\clip.MOV", b"first-input")
        second = self.upload("../../clip.MOV", b"second-input")
        self.assertEqual(first["name"], "clip.MOV")
        self.assertEqual(second["name"], "clip.MOV")
        self.assertNotEqual(first["id"], second["id"])
        for media, expected in ((first, b"first-input"), (second, b"second-input")):
            folder = self.store / "media" / media["id"]
            self.assertEqual(len(media["id"]), 32)
            self.assertEqual((folder / "source.mov").read_bytes(), expected)
            self.assertEqual(media["bytes"], len(expected))
            self.assertEqual(json.loads((folder / "media.json").read_text())["name"], "clip.MOV")
            self.assertEqual(self.client.get(media["preview_url"]).content, b"owned-proxy")
            self.assertEqual(self.client.get(media["poster_url"]).content, b"owned-poster")
        self.assertEqual(hashlib.sha256(sentinel.read_bytes()).hexdigest(), digest)

    def test_upload_validation_and_owned_cleanup(self):
        media_root = self.store / "media"
        sentinel = media_root / "unrelated.txt"; sentinel.write_bytes(b"keep")
        for filename, data, expected in (("payload.exe", b"x", 415), ("empty.mov", b"", 422)):
            response = self.client.post("/api/media", headers=CLIENT_HEADER,
                files={"file": (filename, data, "application/octet-stream")})
            self.assertEqual(response.status_code, expected, response.text)
        self.pipeline.inspect_error = "Invalid video stream"
        self.assertEqual(self.client.post("/api/media", headers=CLIENT_HEADER,
            files={"file": ("bad.mov", b"bad", "video/quicktime")}).status_code, 422)
        self.pipeline.inspect_error = None
        self.pipeline.geometry = (8193, 1080)
        self.assertEqual(self.client.post("/api/media", headers=CLIENT_HEADER,
            files={"file": ("too-large.mov", b"data", "video/quicktime")}).status_code, 422)
        self.pipeline.geometry = (640, 360)
        with patch.object(server, "MAX_UPLOAD", 3):
            self.assertEqual(self.client.post("/api/media", headers=CLIENT_HEADER,
                files={"file": ("over.mov", b"1234", "video/quicktime")}).status_code, 422)
        with patch.object(server.shutil, "disk_usage", return_value=SimpleNamespace(free=1)):
            self.assertEqual(self.client.post("/api/media", headers=CLIENT_HEADER,
                files={"file": ("full.mov", b"1234", "video/quicktime")}).status_code, 422)
        self.assertEqual(sorted(p.name for p in media_root.iterdir()), ["unrelated.txt"])
        self.assertEqual(sentinel.read_bytes(), b"keep")
        self.assertEqual(server.MEDIA, {})
        self.assertEqual(self.pipeline.proxies, [])

    def test_asset_allowlist_no_original_or_arbitrary_file_access(self):
        media = self.upload()
        media_id = media["id"]
        secret = self.store / "private.txt"; secret.write_bytes(b"secret-sentinel")
        for path in (f"/api/media/{media_id}/source.mov", f"/api/media/{media_id}/media.json",
                     f"/api/media/{media_id}/private.txt", "/api/media/nonexistent/proxy.mp4",
                     f"/api/frames/{media_id}/../../private.txt",
                     f"/api/frames/{media_id}/%2e%2e%2fprivate.txt",
                     f"/api/frames/{media_id}/private-before.jpg"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 404, (path, response.text))
            self.assertNotIn(b"secret-sentinel", response.content)
        frames = self.store / "media" / media_id / "frames"; frames.mkdir()
        name = "a" * 24 + "-before.jpg"
        (frames / name).write_bytes(b"owned-preview")
        response = self.client.get(f"/api/frames/{media_id}/{name}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"owned-preview")
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_invalid_settings_rejected_before_gpu_or_queue(self):
        media = self.upload()
        for settings in ({"amount": 2}, {"invented": 1}, {"flare_length": 0},
                         {"amount": True}, {"amount": "0.5"}):
            for endpoint in ("/api/preview", "/api/jobs"):
                response = self.client.post(endpoint, headers=CLIENT_HEADER,
                    json={"media_id": media["id"], "settings": settings})
                self.assertEqual(response.status_code, 422, (endpoint, response.text))
        invalid_time = self.client.post("/api/preview", headers={**CLIENT_HEADER, "Content-Type": "application/json"},
            content=json.dumps({"media_id": media["id"], "time": float("nan")}))
        self.assertEqual(invalid_time.status_code, 422, invalid_time.text)
        self.assertEqual(self.client.post("/api/jobs", headers=CLIENT_HEADER,
            json={"media_id": media["id"], "mode": "untrusted-codec"}).status_code, 422)
        self.assertEqual(self.executor.pending, [])
        self.assertEqual(server.JOBS, {})
        self.assertFalse(server.GPU_LOCK.locked())

    def test_cached_preview_available_while_gpu_busy(self):
        media = self.upload()
        settings = server.engine().normalize_settings({})
        key = hashlib.sha256(json.dumps([0., settings, 1280], sort_keys=True).encode()).hexdigest()[:24]
        frames = self.store / "media" / media["id"] / "frames"; frames.mkdir()
        (frames / (key + "-before.jpg")).write_bytes(b"before")
        (frames / (key + "-after.jpg")).write_bytes(b"after")
        info = {"time": 0., "width": 640, "height": 360, "cached_test": True}
        (frames / (key + ".json")).write_text(json.dumps(info))
        server.GPU_LOCK.acquire()
        try:
            response = self.client.post("/api/preview", headers=CLIENT_HEADER,
                json={"media_id": media["id"], "time": 0., "settings": {}})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json(), info)
            response = self.client.post("/api/preview", headers=CLIENT_HEADER,
                json={"media_id": media["id"], "settings": {"amount": .1}})
            self.assertEqual(response.status_code, 409)
        finally:
            server.GPU_LOCK.release()

    def test_preview_selects_real_video_not_cover_art(self):
        import numpy as np
        audio = SimpleNamespace(index=0, type="audio", disposition=0)
        cover = SimpleNamespace(index=1, type="video", disposition=1024,
                                start_time=300, time_base=Fraction(1, 1000))
        main = SimpleNamespace(index=2, type="video", disposition=0,
                               start_time=300, time_base=Fraction(1, 1000))
        alternate = SimpleNamespace(index=3, type="video", disposition=0,
                                    start_time=300, time_base=Fraction(1, 1000))
        class Streams(list):
            @property
            def video(self):
                return [item for item in self if item.type == "video"]
        class Container:
            def __init__(self):
                self.streams = Streams([audio, cover, main, alternate])
                self.selected = None
                self.seeked = None
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def seek(self, timestamp, *, stream, backward):
                self.seeked = (timestamp, stream.index, backward)
            def decode(self, stream):
                self.selected = stream.index
                pixels = np.full((2, 3, 3), stream.index * 1000, dtype=np.uint16)
                yield SimpleNamespace(pts=800, time_base=Fraction(1, 1000),
                    colorspace=2, color_range=0, height=2, format=SimpleNamespace(name="yuv420p"),
                    to_ndarray=lambda: pixels)
        for metadata, expected in (({}, 2), ({"video_stream_index": 2}, 2),
                                    ({"video_stream_index": 3}, 3)):
            container = Container()
            conversion = []
            def reformat(frame, **options):
                conversion.append(options)
                return frame
            reformatter = SimpleNamespace(reformat=reformat)
            with patch.object(server.av, "open", return_value=container), patch.object(
                    server.av.video.reformatter, "VideoReformatter", return_value=reformatter):
                pixels, timestamp = server.frame_at(Path("unused-owned-source.mov"), .5, metadata)
            self.assertEqual(container.selected, expected)
            self.assertEqual(container.seeked, (800, expected, True))
            self.assertEqual(int(pixels[0, 0, 0]), expected * 1000)
            self.assertAlmostEqual(timestamp, .5)
            self.assertEqual(conversion, [{"format": "rgb48le", "src_colorspace": 5,
                                           "src_color_range": 1, "dst_color_range": 2}])

    def test_queue_cancel_event_and_worker_claim(self):
        job = self.queue()
        self.assertEqual(job["status"], "queued")
        self.assertEqual(len(self.executor.pending), 1)
        function, args, future = self.executor.pending[0]
        self.assertIs(function, server.execute_job)
        response = self.client.post(f'/api/jobs/{job["id"]}/cancel', headers=CLIENT_HEADER)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(server.CANCEL[job["id"]].is_set())
        self.assertEqual(response.json()["status"], "cancelling")
        function(*args)
        self.assertEqual(server.JOBS[job["id"]]["status"], "cancelled")
        self.assertEqual(self.pipeline.exports, [])
        self.assertFalse(server.GPU_LOCK.locked())

    def test_cancel_completion_transaction(self):
        job = self.queue()
        identifier = job["id"]
        completed_at_release = []
        def competing_completion():
            if not server.CANCEL[identifier].is_set():
                completed_at_release.append(True)
                server.update_job(identifier, status="completed", progress=1,
                                  output_name="verified.mov", output_url="ready")
        lock = ReleaseGateLock(competing_completion)
        with patch.object(server, "STATE_LOCK", lock):
            response = self.client.post(f"/api/jobs/{identifier}/cancel", headers=CLIENT_HEADER)
        self.assertEqual(response.status_code, 200, response.text)
        if completed_at_release:
            self.assertEqual(server.JOBS[identifier]["status"], "completed",
                             "A verified completion was overwritten by a stale cancel snapshot")
        else:
            self.assertTrue(server.CANCEL[identifier].is_set())
            self.assertEqual(server.JOBS[identifier]["status"], "cancelling")

    def test_download_completed_only_and_private_fields_hidden(self):
        job = self.queue()
        identifier = job["id"]
        self.assertEqual(self.client.get(f"/api/jobs/{identifier}/download").status_code, 409)
        self.assertEqual(self.client.get("/api/jobs/missing/download").status_code, 404)
        output = self.store / "jobs" / identifier / "verified-anamorphic.mov"
        output.write_bytes(b"verified-output")
        server.update_job(identifier, status="completed", output_name=output.name, progress=1,
                          _private_marker="not-public")
        status = self.client.get(f"/api/jobs/{identifier}")
        self.assertNotIn("_private_marker", status.json())
        download = self.client.get(f"/api/jobs/{identifier}/download")
        self.assertEqual(download.status_code, 200)
        self.assertEqual(download.content, b"verified-output")
        self.assertIn("verified-anamorphic.mov", download.headers["content-disposition"])
        response = self.client.post(f"/api/jobs/{identifier}/cancel", headers=CLIENT_HEADER)
        self.assertEqual(response.json()["status"], "completed")
        self.assertFalse(server.CANCEL[identifier].is_set())

    def test_queue_bound_shutdown_and_public_config(self):
        media = self.upload()
        jobs = [self.queue(media) for _ in range(4)]
        response = self.client.post("/api/jobs", headers=CLIENT_HEADER,
            json={"media_id": media["id"]})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(len(self.executor.pending), 4)
        config = self.client.get("/api/config")
        self.assertEqual(config.status_code, 200)
        self.assertEqual(set(config.json()["profiles"]), {"modern", "vintage", "clean", "off"})
        target = SimpleNamespace(should_exit=False)
        with patch.object(server, "SERVER", target):
            self.assertEqual(self.client.post("/api/shutdown", headers=CLIENT_HEADER).status_code, 200)
        self.assertTrue(target.should_exit)
        self.assertTrue(all(server.CANCEL[job["id"]].is_set() for job in jobs))


if __name__ == "__main__":
    unittest.main(verbosity=2)
