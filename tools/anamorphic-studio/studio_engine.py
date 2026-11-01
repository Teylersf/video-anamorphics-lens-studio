"""Original single-frame CUDA lens/film interpretation for Anamorphic Studio.

Spatial light is extracted/composited in relative linear light. Smooth,
unit-energy three-box spline PSFs avoid a rectangular streak footprint. The
working light canvas is at most 1536 pixels per axis; distortion is sampled in
128-row strips, so no full-resolution sampling grid or frame history is kept.
Only one geometry's 1-D coordinates are cached. Calls must be serialized.

This is an artistic approximation, not an optical lens calibration, depth
reconstruction, extra field of view, desqueeze, sensor mode or HDR restoration.
The optional oval blur protects a user-selected horizontal focus band; it does
not estimate depth or isolate subjects. Input/output RGB primaries are unchanged.
PQ uses ST2084 (linear 1 = 10000 nits). HLG uses inverse/forward ARIB-B67 OETF
without a display-dependent OOTF. Neither path tone maps the exported signal.
The separate preview helper explicitly performs approximate HDR-to-SDR mapping.

GPL-2.0-only, consistent with the surrounding project. No external models/assets.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real

import torch
import torch.nn.functional as F


DEFAULTS = {
    "amount": .65, "flare": .55, "flare_length": .18, "threshold": .8,
    "tint": 0., "bloom": .15, "halation": .1, "ghosts": .08,
    "chromatic": .12, "distortion": .05, "vignette": .15, "grain": .08,
    "bokeh": 0., "ovality": 1.8, "focus_y": .5, "focus_width": .6,
    "exposure": 0., "contrast": 1., "saturation": 1.,
}
_RANGES = {key: (0., 1.) for key in DEFAULTS}
_RANGES.update(flare=(0., 2.), flare_length=(.02, .5), distortion=(-1., 1.),
               ovality=(1., 3.), exposure=(-3., 3.), contrast=(.5, 1.5),
               saturation=(0., 2.))
_TRANSFERS = {"bt709", "srgb", "pq", "hlg"}
_MAX_PIXELS = 35_700_000
_TILE_ROWS = 128
_LUMA = (.2126, .7152, .0722)
# Derived RGB->XYZ->BT709 matrices from the standardized CIE xy coordinates in
# FFmpeg libavutil/csp.c (https://github.com/FFmpeg/FFmpeg/blob/master/libavutil/csp.c).
# SMPTE431's DCI white is Bradford-adapted to D65, as in FFmpeg's colorspace
# filter (https://github.com/FFmpeg/FFmpeg/blob/master/libavfilter/vf_colorspace.c).
# These numerical color definitions are not copies of codec/filter code.
_PREVIEW_MATRICES = {
    "bt470bg": ((1.044043209, -.044043209, 0.), (0., 1., 0.), (0., .011793378, .988206622)),
    "smpte170m": ((.939542064, .050181357, .010276579), (.017772223, .965792862, .016434914),
                   (-.001621600, -.004369750, 1.005991350)),
    "bt2020": ((1.660491002, -.587641139, -.072849863), (-.124550475, 1.132899897, -.008349423),
                (-.018150763, -.100578898, 1.118729661)),
    "smpte432": ((1.224940176, -.224940176, 0.), (-.042056955, 1.042056955, 0.),
                  (-.019637555, -.078636046, 1.098273600)),
    "smpte431": ((1.157516406, -.154962378, -.002554028), (-.041500072, 1.045567923, -.004067852),
                  (-.018050039, -.078578273, 1.096628312)),
}
_PREVIEW_MATRICES["smpte240m"] = _PREVIEW_MATRICES["smpte170m"]


def normalize_settings(mapping: Mapping) -> dict:
    """Return a complete independent settings snapshot; reject ambiguous input."""
    if not isinstance(mapping, Mapping):
        raise TypeError("settings must be a mapping")
    if any(key not in DEFAULTS for key in mapping):
        raise ValueError("unknown setting: " + ", ".join(sorted(str(k) for k in mapping if k not in DEFAULTS)))
    result = dict(DEFAULTS)
    for key, value in mapping.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{key} must be a number")
        value = float(value)
        lo, hi = _RANGES[key]
        if not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f"{key} must be finite and within {lo}..{hi}")
        result[key] = value
    return result


PROFILES = {
    "modern": normalize_settings({}),
    "vintage": normalize_settings({"amount": .8, "flare": .8, "tint": .8,
        "bloom": .24, "halation": .3, "ghosts": .2, "chromatic": .28,
        "distortion": .22, "vignette": .3, "grain": .18, "contrast": .96,
        "saturation": .92}),
    "clean": normalize_settings({"amount": .5, "flare": .25, "bloom": .08,
        "halation": .025, "ghosts": .025, "chromatic": .035,
        "distortion": 0., "vignette": .08, "grain": 0.}),
    "off": normalize_settings({"amount": 0.}),
}


def _transfer_name(transfer):
    if not isinstance(transfer, str) or transfer not in _TRANSFERS:
        raise ValueError("transfer must be bt709, srgb, pq or hlg")
    return transfer


def _validate_frame(frame):
    if not isinstance(frame, torch.Tensor) or frame.dtype != torch.float32:
        raise TypeError("frame must be a float32 torch.Tensor")
    if frame.device.type != "cuda":
        raise ValueError("frame must be on CUDA")
    if frame.ndim != 3 or frame.shape[0] != 3 or min(frame.shape[1:]) < 1:
        raise ValueError("frame must be 3xHxW with positive dimensions")
    h, w = frame.shape[1:]
    if max(h, w) > 8192 or h * w > _MAX_PIXELS:
        raise ValueError("frame exceeds 8192 per axis or 35.7 megapixels")
    lo, hi = torch.aminmax(frame)
    low, high = lo.item(), hi.item()
    if not math.isfinite(low) or not math.isfinite(high) or low < 0 or high > 1:
        raise ValueError("encoded RGB must be finite and within 0..1")


def _decode(x, transfer):
    if transfer == "srgb":
        return torch.where(x <= .04045, x / 12.92, ((x + .055) / 1.055).pow(2.4))
    if transfer == "bt709":
        return torch.where(x < .081, x / 4.5, ((x + .099) / 1.099).pow(1 / .45))
    if transfer == "pq":
        p = x.pow(1 / (2523 / 32))
        return ((p - 3424 / 4096).clamp_min(0) /
                (2413 / 128 - (2392 / 128) * p).clamp_min(1e-7)).pow(1 / (2610 / 16384))
    # Clamp inactive branch arguments too: torch.where evaluates both branches.
    return torch.where(x <= .5, x.square() / 3,
        (((x - .55991073) / .17883277).exp() + .28466892) / 12)


def _encode(x, transfer):
    x = x.clamp(0, 1)
    if transfer == "srgb":
        return torch.where(x <= .0031308, x * 12.92, 1.055 * x.pow(1 / 2.4) - .055).clamp_(0, 1)
    if transfer == "bt709":
        return torch.where(x < .018, x * 4.5, 1.099 * x.pow(.45) - .099).clamp_(0, 1)
    if transfer == "pq":
        p = x.pow(2610 / 16384)
        return ((3424 / 4096 + (2413 / 128) * p) /
                (1 + (2392 / 128) * p)).pow(2523 / 32).clamp_(0, 1)
    return torch.where(x <= 1 / 12, (3 * x).sqrt(),
        .17883277 * (12 * x - .28466892).clamp_min(1e-7).log() + .55991073).clamp_(0, 1)


def _luma(image):
    return image[0] * _LUMA[0] + image[1] * _LUMA[1] + image[2] * _LUMA[2]


def _smoothstep(x):
    x = x.clamp(0, 1)
    return x.square() * (3 - 2 * x)


def _box(image, radius, axis):
    if radius < 1:
        return image
    pad = (radius, radius, 0, 0) if axis == -1 else (0, 0, radius, radius)
    padded = F.pad(image[None], pad, mode="replicate")[0]
    prefix = torch.cumsum(padded, dim=axis)
    length = 2 * radius + 1
    if axis == -1:
        prefix = F.pad(prefix, (1, 0))
        result = prefix[..., length:] - prefix[..., :-length]
    else:
        prefix = F.pad(prefix, (0, 0, 1, 0))
        result = prefix[..., length:, :] - prefix[..., :-length, :]
    return result.mul_(1 / length).clamp_min_(0)


def _spline(image, support, axis):
    """Three normalized boxes: continuous piecewise-quadratic smooth PSF."""
    if support <= 0:
        return image
    radius = max(1, round(support / 3))
    for _ in range(3):
        image = _box(image, radius, axis)
    return image


@torch.inference_mode()
def to_display_rgb(encodedRGB, transfer="bt709", primaries="bt709"):
    """Preview only: source transfer -> approximate SDR -> sRGB display signal.

    PQ/HLG previews use a fixed photographic shoulder, not HDR mastering or an
    export transform. BT2020, BT470BG, SMPTE170M/240M, Display P3 (SMPTE432)
    and DCI P3 (SMPTE431, Bradford white adaptation) convert to BT709 in linear
    light. Unspecified/unknown/None use the caller's BT709 fallback. This does
    not infer a Log curve, camera gamut, grading intent or display calibration;
    other explicitly named primaries are rejected instead of silently relabeled.
    """
    _validate_frame(encodedRGB)
    transfer = _transfer_name(transfer)
    if primaries not in {*_PREVIEW_MATRICES, "bt709", "unspecified", "unknown", None}:
        raise ValueError("preview primaries are unsupported; use a known BT709/2020, SD or P3 definition")
    linear = _decode(encodedRGB, transfer)
    if primaries in _PREVIEW_MATRICES:
        matrix = _PREVIEW_MATRICES[primaries]
        r, g, b = linear.unbind(0)
        linear = torch.stack(tuple(row[0]*r + row[1]*g + row[2]*b for row in matrix)).clamp_min_(0)
    if transfer in {"pq", "hlg"}:
        linear = linear * 100 if transfer == "pq" else linear.pow(1.2) * 10
        # Hue-preserving luminance shoulder; no per-channel highlight clipping.
        y = _luma(linear).clamp_min_(1e-7)
        mapped = y / (1 + y)
        linear = linear * (mapped / y)[None]
    return _encode(linear, "srgb").contiguous()


class StudioRenderer:
    def __init__(self, settings: dict, transfer: str = "bt709"):
        self.settings = normalize_settings(settings)
        self.transfer = _transfer_name(transfer)
        self._geometry_key = None
        self._geometry = {}

    def _prepare(self, image):
        h, w = image.shape[1:]
        key = (image.device, h, w)
        if key != self._geometry_key:
            def coords(n):
                return (torch.linspace(-1, 1, n, device=image.device, dtype=torch.float32)
                        if n > 1 else torch.zeros(1, device=image.device, dtype=torch.float32))
            self._geometry = {"x": coords(w), "y": coords(h)}
            self._geometry_key = key

    def _warp(self, image):
        s, amount = self.settings, self.settings["amount"]
        distortion, ca = s["distortion"] * amount * .08, s["chromatic"] * amount * .0025
        if distortion == 0 and ca == 0:
            return image
        h, w = image.shape[1:]
        x, y = self._geometry["x"], self._geometry["y"]
        out = torch.empty_like(image)
        color = image.new_tensor((1., 0., -1.))[:, None, None]
        source = image[:, None]
        for start in range(0, h, _TILE_ROWS):
            ys = y[start:start + _TILE_ROWS][:, None]
            radius = x[None].square() + ys.square()
            scale = 1 + radius[None] * (distortion + color * ca)
            grid = torch.stack((x[None, None] * scale, ys[None] * scale), dim=-1)
            out[:, start:start + _TILE_ROWS] = F.grid_sample(
                source, grid, mode="bilinear", padding_mode="border", align_corners=True)[:, 0]
        return out

    def _grade(self, image):
        s, a = self.settings, self.settings["amount"]
        if s["exposure"] or s["contrast"] != 1:
            image = image * (2 ** (s["exposure"] * a))
            if s["contrast"] != 1:
                image = .18 * (image / .18).clamp_min(0).pow(1 + (s["contrast"] - 1) * a)
            # Preserve sub-shoulder values; smoothly roll newly overdriven light.
            image = torch.where(image <= .85, image,
                .85 + .15 * (-((image - .85).clamp_min(0) / .15).neg().expm1()))
        if s["saturation"] != 1:
            y = _luma(image)
            image = y[None] + (image - y[None]) * (1 + (s["saturation"] - 1) * a)
        return image.clamp(0, 1)

    def _light(self, image):
        s, a = self.settings, self.settings["amount"]
        h, w = image.shape[1:]
        factor = max(1, math.ceil(max(h, w) / 1536))
        size = (max(1, math.ceil(h / factor)), max(1, math.ceil(w / factor)))
        # Threshold is displayed in encoded source units; extraction is linear.
        threshold = float(_decode(image.new_tensor(s["threshold"]), self.transfer).item())
        knee = max(.015, (1 - threshold) * .3)
        brightness = image.amax(dim=0)
        over = (brightness - threshold + knee).clamp_min_(0)
        mask = _smoothstep(over / (2 * knee)) * (over / (1 - threshold + knee))
        seed = F.interpolate((image * mask[None])[None], size=size, mode="area")[0]
        result = torch.zeros_like(seed)
        tint_values = tuple(blue * (1 - s["tint"]) + amber * s["tint"]
                            for blue, amber in zip((.18, .48, 1.), (1., .52, .16)))
        tint = image.new_tensor(tint_values)
        work_w = size[1]
        if s["flare"]:
            radius = max(1., s["flare_length"] * work_w)
            for c, dispersion in enumerate((-.12, 0., .12)):
                channel = seed[c:c + 1]
                streak = torch.zeros_like(channel)
                for scale, weight in ((.14, .48), (.45, .32), (1., .2)):
                    streak.add_(_spline(channel, radius * scale * (1 + dispersion * s["chromatic"]), -1), alpha=weight)
                streak = _spline(streak, max(1., work_w * .0014), -2)
                result[c:c + 1].add_(streak, alpha=tint_values[c] * a * s["flare"] * 3)
        if s["bloom"]:
            glow = _spline(_spline(seed, work_w * .012, -1), work_w * .0035, -2)
            result.add_(glow, alpha=a * s["bloom"] * 1.8)
        if s["halation"]:
            halo = _spline(_spline(seed.mean(dim=0, keepdim=True), work_w * .005, -1), work_w * .003, -2)
            result.add_(halo * image.new_tensor((1., .18, .045))[:, None, None], alpha=a * s["halation"])
        if s["ghosts"]:
            wh, ww = size
            gx = torch.linspace(-1, 1, ww, device=image.device, dtype=torch.float32) if ww > 1 else image.new_zeros(1)
            gy = torch.linspace(-1, 1, wh, device=image.device, dtype=torch.float32) if wh > 1 else image.new_zeros(1)
            rr = gx[None].square() + gy[:, None].square()
            envelope = (-rr * 1.8).exp()
            ghosts = torch.zeros_like(seed)
            for scale, weight in ((.35, .4), (.62, .35), (1.05, .25)):
                grid = torch.stack((-gx[None].expand(wh, ww) / scale,
                                    -gy[:, None].expand(wh, ww) / scale), dim=-1)
                ghosts.add_(F.grid_sample(seed[None], grid[None], padding_mode="zeros",
                    align_corners=True)[0], alpha=weight)
            ghosts = _spline(_spline(ghosts, work_w * .003, -1), work_w * .003, -2)
            result.add_(ghosts * envelope[None] * ((tint + .4) / 1.4)[:, None, None], alpha=a * s["ghosts"] * .7)
        return result

    def _bokeh(self, image):
        s = self.settings
        if not s["bokeh"] or s["focus_width"] == 1:
            return image
        h, w = image.shape[1:]
        factor = max(1, math.ceil(max(h, w) / 1536))
        work = F.interpolate(image[None], size=(max(1, math.ceil(h / factor)), max(1, math.ceil(w / factor))), mode="area")[0]
        radius = work.shape[2] * .004 * s["bokeh"]
        work = _spline(_spline(work, max(1., radius), -1), max(1., radius * s["ovality"]), -2)
        blur = F.interpolate(work[None], size=(h, w), mode="bilinear", align_corners=True)[0]
        y = (self._geometry["y"] + 1) * .5
        distance = (y - s["focus_y"]).abs() - s["focus_width"] * .5
        mask = _smoothstep(distance / max(.025, (1 - s["focus_width"]) * .18))
        return torch.lerp(image, blur, (mask * s["bokeh"] * s["amount"] * .9)[None, :, None])

    def _finish(self, image, light, frame_index):
        s, a = self.settings, self.settings["amount"]
        h, w = image.shape[1:]
        # A single full-resolution light canvas is released before encoding.
        glow = F.interpolate(light[None], size=(h, w), mode="bilinear", align_corners=True)[0]
        image = image + (1 - image) * (-glow.clamp_min(0).neg().expm1())
        del glow
        x, y = self._geometry["x"], self._geometry["y"]
        if s["vignette"]:
            for start in range(0, h, _TILE_ROWS):
                radial = (x[None].square() + y[start:start + _TILE_ROWS, None].square()) * .5
                falloff = 1 - _smoothstep(radial).square() * a * s["vignette"] * .75
                image[:, start:start + _TILE_ROWS].mul_(falloff[None])
        encoded = _encode(image, self.transfer)
        del image
        if s["grain"]:
            # Local integer coordinate hash; never changes torch's global RNG.
            cell = max(1, round(w / 1920))
            nx = torch.arange(w, device=encoded.device, dtype=torch.int64).div_(cell, rounding_mode="floor")[None]
            seed = (0xA341316C + frame_index * 0x9E3779B9) & 0xFFFFFFFF
            for start in range(0, h, _TILE_ROWS):
                ny = torch.arange(start, min(h, start + _TILE_ROWS), device=encoded.device, dtype=torch.int64).div_(cell, rounding_mode="floor")[:, None]
                hashed = (nx * 374761393 + ny * 668265263 + seed).bitwise_and_(0xFFFFFFFF)
                hashed = ((hashed ^ (hashed >> 13)) * 1274126177).bitwise_and_(0xFFFFFFFF)
                hashed.bitwise_xor_(hashed >> 16)
                noise = hashed.bitwise_and_(65535).to(torch.float32).mul_(1 / 32768).sub_(1)
                tile = encoded[:, start:start + _TILE_ROWS]
                envelope = (tile * (1 - tile)).clamp_min(0).sqrt_()
                tile.add_(noise[None] * envelope, alpha=a * s["grain"] * .055).clamp_(0, 1)
        return encoded.contiguous()

    @torch.inference_mode()
    def render(self, frame: torch.Tensor, frame_index: int = 0, time_s: float = 0) -> torch.Tensor:
        """Process one encoded RGB frame; time_s is validated, grain uses index.

        The amount=0 bypass is value-exact for every supported transfer. Frame
        input is never modified. There is no temporal frame cache or optical flow.
        """
        _validate_frame(frame)
        if type(frame_index) is not int or frame_index < 0:
            raise ValueError("frame_index must be a nonnegative int")
        if isinstance(time_s, bool) or not isinstance(time_s, Real) or not math.isfinite(float(time_s)) or time_s < 0:
            raise ValueError("time_s must be a finite nonnegative number")
        if self.settings["amount"] == 0:
            return frame.clone(memory_format=torch.contiguous_format)
        self._prepare(frame)
        image = self._grade(self._warp(_decode(frame, self.transfer)))
        image = self._bokeh(image)
        if any(self.settings[key] for key in ("flare", "bloom", "halation", "ghosts")):
            light = self._light(image)
        else:
            light = image.new_zeros((3, 1, 1))
        return self._finish(image, light, frame_index)

    __call__ = render


__all__ = ["DEFAULTS", "PROFILES", "normalize_settings", "StudioRenderer", "to_display_rgb"]
