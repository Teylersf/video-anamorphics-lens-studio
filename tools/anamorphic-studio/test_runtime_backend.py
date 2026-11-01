"""Device policy tests use mocks, so CPU CI covers unavailable GPU behavior."""
from __future__ import annotations

import json
import os
from unittest.mock import patch
import unittest

import torch

import runtime_backend as runtime


class BackendTests(unittest.TestCase):
    def test_auto_priority_and_environment_override(self):
        for cuda, mps, expected in ((True, True, "cuda"), (False, True, "mps"),
                                     (False, False, "cpu")):
            with patch.object(torch.cuda, "is_available", return_value=cuda), patch.object(
                    torch.backends.mps, "is_available", return_value=mps), patch.dict(
                    os.environ, {"STUDIO_DEVICE": "auto"}):
                self.assertEqual(runtime.select_device(), torch.device(expected))
                self.assertEqual(runtime.select_device("auto"), torch.device(expected))
                with patch.dict(os.environ, {"STUDIO_DEVICE": " CPU "}):
                    self.assertEqual(runtime.select_device(), torch.device("cpu"))
                    self.assertEqual(runtime.select_device("auto"), torch.device(expected))

    def test_cpu_selection_does_not_probe_accelerators(self):
        with patch.object(torch.cuda, "is_available", side_effect=AssertionError) as cuda, patch.object(
                torch.backends.mps, "is_available", side_effect=AssertionError) as mps:
            self.assertEqual(runtime.select_device("cpu"), torch.device("cpu"))
            self.assertEqual(runtime.backend_info("cpu")["accelerated"], False)
            cuda.assert_not_called(); mps.assert_not_called()

    def test_explicit_unavailable_is_helpful_never_silent_fallback(self):
        with patch.object(torch.cuda, "is_available", return_value=False), patch.object(
                torch.backends.mps, "is_available", return_value=False), patch.object(
                torch.backends.mps, "is_built", return_value=False):
            for choice in ("cuda", "mps"):
                with self.assertRaisesRegex(RuntimeError, "requested.*unavailable|requested.*no Metal"):
                    runtime.select_device(choice)
                with self.assertRaises(RuntimeError):
                    runtime.backend_info(torch.device(choice))
            with self.assertRaisesRegex(RuntimeError, "PyTorch build has no Metal"):
                runtime.select_device("mps")
            with patch.object(torch.backends.mps, "is_built", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "hardware/macOS"):
                    runtime.select_device("mps")

    def test_invalid_choices(self):
        for choice in ("", "gpu", "metal", "cuda:0", "mps:0", "xpu", "cpu;command"):
            with self.assertRaises(ValueError):
                runtime.select_device(choice)
        for choice in (True, 0, [], torch.device("cpu")):
            with self.assertRaises(TypeError):
                runtime.select_device(choice)
        with patch.dict(os.environ, {"STUDIO_DEVICE": "invalid"}):
            with self.assertRaises(ValueError):
                runtime.select_device()
        for value in ("meta", "xpu"):
            with self.assertRaises(ValueError):
                runtime.backend_info(value)

    def test_info_json_safe_and_matching_selected_device(self):
        cpu = runtime.backend_info(torch.device("cpu"))
        self.assertEqual(set(cpu), {"device", "gpu", "accelerated", "reason"})
        self.assertEqual(cpu["device"], "cpu")
        self.assertFalse(cpu["accelerated"])
        json.dumps(cpu)
        with patch.object(torch.cuda, "is_available", return_value=True), patch.object(
                torch.cuda, "get_device_name", return_value="Mock NVIDIA"):
            info = runtime.backend_info(torch.device("cuda:0"))
            self.assertEqual(info["gpu"], "Mock NVIDIA")
            self.assertTrue(info["accelerated"])
            self.assertEqual(info["device"], "cuda:0")
        with patch.object(torch.backends.mps, "is_available", return_value=True), patch.object(
                torch.backends.mps, "get_name", return_value="Mock Metal", create=True):
            info = runtime.backend_info("mps")
            self.assertEqual(info["device"], "mps")
            self.assertEqual(info["gpu"], "Mock Metal")
            self.assertTrue(info["accelerated"])
        with patch.object(torch.backends.mps, "is_available", return_value=True), patch.object(
                torch.backends.mps, "get_name", side_effect=RuntimeError("driver"), create=True):
            self.assertEqual(runtime.backend_info("mps")["gpu"], "Metal GPU (MPS)")

    def test_cache_cleanup_is_backend_specific_and_cpu_noop(self):
        with patch.object(torch.cuda, "empty_cache") as cuda, patch.object(torch.mps, "empty_cache") as mps:
            runtime.empty_cache("cpu")
            cuda.assert_not_called(); mps.assert_not_called()
            with patch.object(torch.cuda, "is_available", return_value=False), patch.object(
                    torch.backends.mps, "is_available", return_value=False):
                runtime.empty_cache("cuda"); runtime.empty_cache(torch.device("mps"))
            cuda.assert_not_called(); mps.assert_not_called()
            with patch.object(torch.cuda, "is_available", return_value=True), patch.object(torch.cuda, "device") as context:
                runtime.empty_cache(torch.device("cuda:0"))
                context.assert_called_once_with(torch.device("cuda:0"))
                cuda.assert_called_once(); mps.assert_not_called()
            with patch.object(torch.backends.mps, "is_available", return_value=True):
                runtime.empty_cache("mps")
                mps.assert_called_once()

    def test_driver_probe_error_auto_uses_cpu(self):
        with patch.object(torch.cuda, "is_available", side_effect=RuntimeError("driver unavailable")), patch.object(
                torch.backends.mps, "is_available", side_effect=OSError("driver unavailable")):
            self.assertEqual(runtime.select_device("auto"), torch.device("cpu"))
            with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
                runtime.select_device("cuda")


if __name__ == "__main__":
    unittest.main(verbosity=2)
