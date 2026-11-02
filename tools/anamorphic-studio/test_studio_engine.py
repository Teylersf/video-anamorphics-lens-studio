"""Focused CPU/CUDA/MPS behavior tests; opt in to full 8K with --8k.

Run with python test_studio_engine.py [--8k]. Stress timing is
effect-only host/GPU processing, never sensor FPS or end-to-end export speed.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import time
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from studio_engine import (DEFAULTS, PROFILES, StudioRenderer, normalize_settings,
                           to_display_rgb, _decode, _encode, _spline, _PREVIEW_MATRICES,
                           _adaptive_area, _area_resize, _grain_noise)

STRESS_8K = "--8k" in sys.argv
if STRESS_8K:
    sys.argv.remove("--8k")


def neutral(**changes):
    settings = dict(DEFAULTS)
    settings.update(amount=1., flare=0., bloom=0., halation=0., ghosts=0.,
                    chromatic=0., distortion=0., vignette=0., grain=0., bokeh=0.)
    settings.update(changes)
    return settings


class SettingsTests(unittest.TestCase):
    def test_preview_primary_matrices_from_chromaticities(self):
        # Independent CPU derivation from CIE xy primary/white definitions;
        # verifies all nine constants, including DCI->D65 adaptation direction.
        import numpy as np
        d65, dci = (.3127, .3290), (.3140, .3510)
        chromaticities = {
            "bt709": ((.640, .330), (.300, .600), (.150, .060)),
            "bt470bg": ((.640, .330), (.290, .600), (.150, .060)),
            "smpte170m": ((.630, .340), (.310, .595), (.155, .070)),
            "bt2020": ((.708, .292), (.170, .797), (.131, .046)),
            "smpte432": ((.680, .320), (.265, .690), (.150, .060)),
            "smpte431": ((.680, .320), (.265, .690), (.150, .060)),
        }
        def xyz(white):
            x, y = white
            return np.array((x / y, 1., (1 - x - y) / y))
        def rgb_xyz(primaries, white):
            base = np.array([(x / y, 1., (1 - x - y) / y) for x, y in primaries]).T
            return base @ np.diag(np.linalg.solve(base, xyz(white)))
        bradford = np.array(((.8951, .2664, -.1614), (-.7502, 1.7135, .0367), (.0389, -.0685, 1.0296)))
        target = rgb_xyz(chromaticities["bt709"], d65)
        for name, primary in chromaticities.items():
            if name == "bt709":
                continue
            white = dci if name == "smpte431" else d65
            adaptation = np.linalg.solve(bradford, np.diag((bradford @ xyz(d65)) / (bradford @ xyz(white))) @ bradford)
            expected = np.linalg.solve(target, adaptation @ rgb_xyz(primary, white))
            actual = np.array(_PREVIEW_MATRICES[name])
            self.assertTrue(np.allclose(actual, expected, atol=1e-9, rtol=0), name)
            self.assertTrue(np.allclose(actual @ np.ones(3), np.ones(3), atol=2e-9, rtol=0), name)
        self.assertEqual(_PREVIEW_MATRICES["smpte170m"], _PREVIEW_MATRICES["smpte240m"])

    def test_normalization_snapshot_profiles(self):
        self.assertEqual(set(PROFILES), {"modern", "vintage", "clean", "off"})
        for profile in PROFILES.values():
            self.assertEqual(set(profile), set(DEFAULTS))
            self.assertEqual(normalize_settings(profile), profile)
        source = {"flare": 1}
        normalized = normalize_settings(source)
        source["flare"] = 0
        self.assertEqual(normalized["flare"], 1.)
        self.assertEqual(normalize_settings({}), DEFAULTS)
        for bad in (None, [], 1):
            with self.assertRaises(TypeError):
                normalize_settings(bad)
        for bad in ({"absent": 0}, {"flare": -1}, {"amount": 1.01},
                    {"flare_length": 0}, {"ovality": 4}, {"contrast": 0},
                    {"grain": math.inf}, {"grain": math.nan}):
            with self.assertRaises(ValueError):
                normalize_settings(bad)
        for bad in ({"grain": True}, {"tint": "blue"}):
            with self.assertRaises(TypeError):
                normalize_settings(bad)
        with self.assertRaises(ValueError):
            StudioRenderer({}, "not-a-transfer")


class CpuTests(unittest.TestCase):
    device = "cpu"

    def test_decoder_style_noncontiguous_rgb(self):
        frame = torch.linspace(0, 1, 31 * 47 * 3, device=self.device).reshape(31, 47, 3).permute(2, 0, 1)
        self.assertFalse(frame.is_contiguous())
        original = frame.clone()
        renderer = StudioRenderer({**PROFILES["vintage"], "bokeh": .3})
        actual = renderer(frame, 5)
        expected = renderer(frame.contiguous(), 5)
        self.assertEqual(tuple(actual.shape), (3, 31, 47))
        self.assertEqual(actual.device, frame.device)
        self.assertTrue(actual.is_contiguous())
        self.assertLess(float((actual - expected).abs().max()), 3e-4)
        self.assertTrue(torch.equal(frame, original))

    def test_nondivisible_area_and_real_downsample_shape(self):
        import torch.nn.functional as functional
        for shape, size in (((3, 7, 4097), (3, 1366)), ((3, 53, 79), (17, 23)),
                            ((3, 1, 17), (1, 6)), ((3, 19, 1), (7, 1))):
            source = torch.linspace(0, 1, math.prod(shape), dtype=torch.float32).reshape(shape)
            reference = functional.interpolate(source[None], size=size, mode="area")[0]
            actual = _adaptive_area(source.to(self.device), size).cpu()
            self.assertLess(float((actual - reference).abs().max()), 2e-4)
            self.assertEqual(tuple(actual.shape), (3, *size))
            routed = _area_resize(source.to(self.device), size).cpu()
            self.assertLess(float((routed - reference).abs().max()), 2e-4)
        frame = torch.full((3, 7, 4097), .18, device=self.device)
        frame[:, 2:5, 2300:2350] = .95
        original = frame.clone()
        out = StudioRenderer({**PROFILES["vintage"], "bokeh": .4})(frame, 9)
        self.assertEqual(out.device, frame.device)
        self.assertEqual(tuple(out.shape), tuple(frame.shape))
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertTrue(torch.equal(frame, original))
        self.assertGreaterEqual(float(out.min()), 0)
        self.assertLessEqual(float(out.max()), 1)

    def test_grain_scalar_reference_and_device_noise(self):
        # Scalar Python integer reference: no CUDA/MPS bitwise implementation
        # can change the established hash stream, including rows crossing128.
        seed, cell, start, rows, width = 0xFFFFFFFF, 3, 127, 7, 29
        expected = []
        for y in range(start, start + rows):
            line = []
            for x in range(width):
                value = ((x // cell) * 374761393 + (y // cell) * 668265263 + seed) & 0xFFFFFFFF
                value = ((value ^ (value >> 13)) * 1274126177) & 0xFFFFFFFF
                value ^= value >> 16
                line.append((value & 65535) / 32768 - 1)
            expected.append(line)
        expected = torch.tensor(expected, dtype=torch.float32)
        noise_device = "cpu" if self.device == "mps" else self.device
        nx = torch.arange(width, device=noise_device, dtype=torch.int64).div_(cell, rounding_mode="floor")[None]
        actual = _grain_noise(nx, start, rows, cell, seed)
        self.assertTrue(torch.equal(actual.cpu(), expected))
        if self.device == "mps":
            self.assertTrue(torch.equal(actual.to("mps").cpu(), expected))
    def test_off_exact_all_transfers_no_input_change(self):
        frame = torch.linspace(0, 1, 3 * 19 * 33, device=self.device).reshape(3, 19, 33)
        original = frame.clone()
        frame.requires_grad_(True)
        for transfer in ("bt709", "srgb", "pq", "hlg"):
            out = StudioRenderer(PROFILES["off"], transfer).render(frame, 7, .3)
            self.assertTrue(torch.equal(out, original))
            self.assertFalse(out.requires_grad)
            self.assertNotEqual(out.data_ptr(), frame.data_ptr())
        self.assertTrue(torch.equal(frame, original))

    def test_transfer_roundtrip_and_hdr_preview(self):
        x = torch.linspace(0, 1, 3 * 47 * 59, device=self.device).reshape(3, 47, 59)
        for transfer, tolerance in (("srgb", 5e-7), ("bt709", 3e-4),
                                    ("pq", 3e-5), ("hlg", 5e-7)):
            roundtrip = _encode(_decode(x, transfer), transfer)
            self.assertLessEqual(float((roundtrip - x).abs().max()), tolerance)
            out = StudioRenderer(neutral(), transfer).render(x)
            self.assertLessEqual(float((out - x).abs().max()), tolerance)
            preview = to_display_rgb(x, transfer, "bt2020")
            self.assertTrue(bool(torch.isfinite(preview).all()))
            self.assertGreaterEqual(float(preview.min()), 0)
            self.assertLessEqual(float(preview.max()), 1)
        # Equal-energy white remains neutral through the primary matrix.
        white = torch.full((3, 3, 5), .6, device=self.device)
        p = to_display_rgb(white, "pq", "bt2020")
        self.assertLess(float((p[0] - p[2]).abs().max()), 1e-5)
        with self.assertRaises(ValueError):
            to_display_rgb(x, "bt709", "made-up")

    def test_common_sd_p3_preview_primaries(self):
        white = torch.full((3, 3, 7), .55, device=self.device)
        colored = torch.tensor((.23, .48, .72), device=self.device)[:, None, None].expand(3, 3, 7).contiguous()
        for primaries in _PREVIEW_MATRICES:
            for transfer in ("bt709", "srgb", "pq", "hlg"):
                out = to_display_rgb(white, transfer, primaries)
                self.assertLess(float((out[0] - out[2]).abs().max()), 1e-5, primaries)
                self.assertTrue(bool(torch.isfinite(out).all()))
                self.assertEqual(out.dtype, torch.float32)
                self.assertEqual(tuple(out.shape), tuple(white.shape))
            out = to_display_rgb(colored, "srgb", primaries)
            linear = _decode(colored, "srgb")
            matrix = torch.tensor(_PREVIEW_MATRICES[primaries], device=self.device, dtype=torch.float32)
            expected = _encode((matrix @ linear[:, 0, 0]).clamp_min(0)[:, None, None], "srgb")
            self.assertLess(float((out - expected).abs().max()), 1e-6, primaries)
            self.assertGreaterEqual(float(out.min()), 0)
            self.assertLessEqual(float(out.max()), 1)

    def test_normalized_smooth_psf_no_rectangular_plateau(self):
        point = torch.zeros((1, 17, 513), device=self.device)
        point[0, 8, 256] = 1
        result = _spline(point, 75, -1)
        line = result[0, 8]
        self.assertAlmostEqual(float(line.sum()), 1., places=5)
        self.assertAlmostEqual(float((line - line.flip(0)).abs().max()), 0., places=6)
        tail = line[257:330]
        self.assertTrue(bool((tail[:-1] >= tail[1:] - 1e-7).all()))
        self.assertGreater(float(tail[3]), float(tail[25]))
        self.assertGreater(float(tail[25]), float(tail[55]))
        uniform = torch.full((3, 5, 7), .4, device=self.device)
        self.assertLess(float((_spline(uniform, 20, -1) - uniform).abs().max()), 1e-6)

    def test_flare_highlight_isolation_direction_and_tint(self):
        dark = torch.full((3, 129, 257), .1, device=self.device)
        settings = neutral(flare=1., threshold=.8, flare_length=.22)
        render = StudioRenderer(settings)
        self.assertLess(float((render(dark) - dark).abs().max()), 1e-6)
        scene = dark.clone()
        scene[:, 62:67, 126:131] = 1
        blue = render(scene)
        amber = StudioRenderer({**settings, "tint": 1.})(scene)
        self.assertGreater(float(blue[2, 64, 155]), float(blue[2, 95, 128]) + .002)
        self.assertGreater(float(blue[2, 64, 155]), float(blue[0, 64, 155]))
        self.assertGreater(float(amber[0, 64, 155]), float(amber[2, 64, 155]))
        self.assertTrue(torch.equal(scene[:, 62:67, 126:131], torch.ones_like(scene[:, 62:67, 126:131])))

    def test_grain_repeatability_no_global_rng_and_cache(self):
        scene = torch.full((3, 37, 65), .4, device=self.device)
        renderer = StudioRenderer(neutral(grain=.6))
        get_state = torch.cuda.get_rng_state if self.device == "cuda" else torch.mps.get_rng_state if self.device == "mps" else torch.get_rng_state
        state = get_state().clone()
        a, b, c = renderer(scene, 3), renderer(scene, 3), renderer(scene, 4)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, c))
        self.assertTrue(torch.equal(get_state(), state))
        renderer(torch.zeros((3, 1, 1), device=self.device))
        self.assertEqual(renderer._geometry_key[1:], (1, 1))
        self.assertEqual(sum(v.numel() for v in renderer._geometry.values()), 2)
        for shape in ((1, 17), (13, 1), (1, 1)):
            out = StudioRenderer(PROFILES["vintage"])(torch.full((3, *shape), .4, device=self.device))
            self.assertEqual(tuple(out.shape), (3, *shape))
            self.assertTrue(bool(torch.isfinite(out).all()))

    def test_focus_band_and_edge_clamp(self):
        h, w = 129, 257
        yy = torch.arange(h, device=self.device)[:, None]
        xx = torch.arange(w, device=self.device)[None]
        checker = ((xx + yy) % 2).float() * .5 + .2
        scene = checker.expand(3, -1, -1).contiguous()
        settings = neutral(bokeh=1., focus_y=.5, focus_width=.4, ovality=3.)
        result = StudioRenderer(settings)(scene)
        self.assertLess(float((result[:, 50:80] - scene[:, 50:80]).abs().max()), 1e-6)
        self.assertLess(float(result[:, :20].var()), float(scene[:, :20].var()) * .2)
        full_band = StudioRenderer({**settings, "focus_width": 1.})(scene)
        self.assertLess(float((full_band - scene).abs().max()), 1e-6)
        constant = torch.full((3, 67, 131), .35, device=self.device)
        warp = StudioRenderer(neutral(distortion=-1., chromatic=1.))(constant)
        self.assertLess(float((warp - constant).abs().max()), 1e-6)

    def test_ghosts_and_all_settings_extremes(self):
        frame = torch.zeros((3, 77, 131), device=self.device)
        frame[:, 12:22, 95:105] = 1
        ghost = StudioRenderer(neutral(ghosts=1., threshold=.5))(frame)
        self.assertGreater(float(ghost[:, 45:70, 5:55].max()), .001)
        extreme = neutral(amount=1., flare=2., flare_length=.5, threshold=0.,
            tint=1., bloom=1., halation=1., ghosts=1., chromatic=1., distortion=1.,
            vignette=1., grain=1., bokeh=1., ovality=3., focus_y=0., focus_width=0.,
            exposure=3., contrast=1.5, saturation=2.)
        for profile in [*PROFILES.values(), extreme, {**extreme, "exposure": -3., "contrast": .5, "distortion": -1.}]:
            out = StudioRenderer(profile)(frame, 2)
            self.assertTrue(bool(torch.isfinite(out).all()))
            self.assertGreaterEqual(float(out.min()), 0)
            self.assertLessEqual(float(out.max()), 1)

    def test_validation(self):
        renderer = StudioRenderer({})
        valid = torch.zeros((3, 9, 11), device=self.device)
        # MPS cannot construct float64 at all; float16 still exercises our
        # float32-only contract without failing in the test fixture first.
        wrong_dtype = valid.half() if self.device == "mps" else valid.double()
        for frame in (torch.empty((3, 9, 11), device="meta"), wrong_dtype, torch.zeros((2, 9, 11), device=self.device),
                      torch.empty((3, 0, 11), device=self.device)):
            with self.assertRaises((TypeError, ValueError)):
                renderer(frame)
        for value in (-.01, 1.01, math.nan, math.inf):
            frame = valid.clone(); frame[0, 0, 0] = value
            with self.assertRaises(ValueError):
                renderer(frame)
        for index in (-1, True, 1.):
            with self.assertRaises(ValueError):
                renderer(valid, index)
        for value in (-1, True, math.inf, math.nan, "0"):
            with self.assertRaises(ValueError):
                renderer(valid, 0, value)
        too_large = torch.as_strided(valid, (3, 8192, 8192), (0, 0, 0))
        with self.assertRaises(ValueError):
            renderer(too_large)

    @unittest.skipUnless(STRESS_8K, "use --8k during an exclusive GPU interval")
    def test_full_8k_accelerator_memory(self):
        if self.device not in {"cuda", "mps"}:
            self.skipTest("Opt-in 8K stress requires CUDA or Metal MPS")
        synchronize = torch.cuda.synchronize if self.device == "cuda" else torch.mps.synchronize
        renderer = StudioRenderer({**PROFILES["vintage"], "bokeh": .35})
        renderer(torch.full((3, 96, 160), .3, device=self.device))
        synchronize()
        frame = torch.full((3, 4320, 7680), .18, device=self.device)
        frame[:, 1800:2100, 4500:4750] = .98
        synchronize()
        if self.device == "cuda":
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        else:
            allocated_before = torch.mps.current_allocated_memory()
            driver_before = torch.mps.driver_allocated_memory()
            recommended = torch.mps.recommended_max_memory()
        wall = time.perf_counter()
        if self.device == "cuda":
            start.record()
        result = renderer(frame, 11, 11 / 30)
        if self.device == "cuda":
            end.record()
        synchronize()
        elapsed = time.perf_counter() - wall
        if self.device == "cuda":
            peak = torch.cuda.max_memory_allocated()
        else:
            allocated_after = torch.mps.current_allocated_memory()
            driver_after = torch.mps.driver_allocated_memory()
        self.assertEqual(tuple(result.shape), (3, 4320, 7680))
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.device, frame.device)
        self.assertTrue(bool(torch.isfinite(result).all()))
        self.assertGreaterEqual(float(result.min()), 0)
        self.assertLessEqual(float(result.max()), 1)
        if self.device == "cuda":
            self.assertLess(peak, 22 * 1024**3)
        else:
            for value in (allocated_before, allocated_after, driver_before, driver_after, recommended):
                self.assertGreaterEqual(value, 0)
        self.assertEqual(renderer._geometry_key[1:], (4320, 7680))
        self.assertEqual(sum(v.numel() for v in renderer._geometry.values()), 12000)
        # Five exact constant-region checks cover every input pixel, without
        # retaining an additional 380MiB full-RGB clone during the stress render.
        for region in (frame[:, :1800], frame[:, 2100:], frame[:, 1800:2100, :4500],
                       frame[:, 1800:2100, 4750:]):
            self.assertTrue(bool((region == .18).all()), "Renderer changed the input background")
        self.assertTrue(bool((frame[:, 1800:2100, 4500:4750] == .98).all()),
                        "Renderer changed the input highlight")
        if self.device == "cuda":
            print(f"\n8K effect-only: CUDA {start.elapsed_time(end):.2f}ms, wall {elapsed:.3f}s, "
                  f"peak total {peak / 1024**3:.3f}GiB, incremental {(peak-baseline) / 1024**3:.3f}GiB")
        else:
            print(f"\n8K effect-only: MPS wall {elapsed:.3f}s, input unchanged; "
                  f"current allocated before/after {allocated_before / 1024**3:.3f}/{allocated_after / 1024**3:.3f}GiB, "
                  f"driver before/after {driver_before / 1024**3:.3f}/{driver_after / 1024**3:.3f}GiB, "
                  f"recommended {recommended / 1024**3:.3f}GiB. "
                  "MPS figures are endpoint observations, not a measured peak.")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class CudaTests(CpuTests):
    device = "cuda"


@unittest.skipUnless(torch.backends.mps.is_available(), "Metal MPS unavailable")
class MpsTests(CpuTests):
    device = "mps"


class BackendComparisonTests(unittest.TestCase):
    def test_available_accelerators_against_cpu(self):
        devices = []
        if torch.cuda.is_available():
            devices.append("cuda")
        if torch.backends.mps.is_available():
            devices.append("mps")
        if not devices:
            self.skipTest("No CUDA/MPS hardware; CPU behavior suite still runs")
        frame = torch.linspace(0, 1, 3 * 39 * 67).reshape(3, 39, 67)
        settings = {**PROFILES["vintage"], "bokeh": .55, "focus_width": .35}
        for transfer in ("bt709", "srgb", "pq", "hlg"):
            expected = StudioRenderer(settings, transfer)(frame, 6)
            for device in devices:
                actual = StudioRenderer(settings, transfer)(frame.to(device), 6).cpu()
                error = (actual - expected).abs()
                tolerance = 2e-4 if device == "cuda" else 3e-3
                self.assertLess(float(error.max()), tolerance, (device, transfer))
                self.assertLess(float(error.mean()), tolerance / 10, (device, transfer))


if __name__ == "__main__":
    unittest.main(verbosity=2)
