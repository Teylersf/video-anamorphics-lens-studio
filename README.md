# Video Anamorphics & Lens Studio

[![Studio checks](https://github.com/Teylersf/video-anamorphics-lens-studio/actions/workflows/studio.yml/badge.svg)](https://github.com/Teylersf/video-anamorphics-lens-studio/actions/workflows/studio.yml)

A local GPU video app for shaping anamorphic lens character, cinematic light
and film texture. Import footage, compare before/after, adjust the look and
export at the original resolution—tested through **7680×4320, 10-bit HDR**.

**Local processing. Original files preserved. Match-source export by default.**
No camera firmware, cloud account or Vercel deployment is required.

## What it does

- Four starting profiles: **Modern, Vintage, Clean and Bypass**.
- Nineteen controls for smooth horizontal flares, blue/amber coating color,
  lens ghosts, bloom, halation, chromatic separation, distortion, vignette,
  grain, focus falloff, ovality and color adjustments.
- GPU frame previews with a draggable before/after comparison and source
  timeline, plus a rendered three-second preview with playback.
- Save/load look settings, export progress, cancellation and verified downloads.
- Source timing including VFR, display rotation, bit depth/chroma and compatible
  original audio are retained in supported match-source exports.
- Explicit HEVC delivery, ProRes editing and lossless RGB16/FFV1 master options.

The lens renderer is an artistic optical approximation. Oval bokeh uses an
adjustable focus band, not estimated scene depth. It does not recover hidden
detail, add field of view or turn lower-resolution footage into captured 8K.

## Run locally

Tested on **Windows, Python 3.13, Node 22.19, RTX 3090 24 GB, CUDA-enabled
PyTorch 2.11.0+cu126, PyAV 17 and FFmpeg 7.0.2**. A working NVIDIA CUDA GPU is
required for effects; the UI and CPU contract tests do not require GPU rendering.

1. Install Git, Python, Node and FFmpeg/ffprobe on PATH.
2. Install CUDA-enabled PyTorch appropriate for your GPU using the
   [official PyTorch instructions](https://pytorch.org/get-started/locally/).
3. Clone and install the small application dependencies:

```powershell
git clone https://github.com/Teylersf/video-anamorphics-lens-studio.git
cd video-anamorphics-lens-studio
python -m venv --system-site-packages .venv-studio
.venv-studio\Scripts\python.exe -m pip install -r tools/anamorphic-studio/requirements.txt
npm --prefix app ci
python tools/anamorphic-studio/launch.py
```

The launcher builds the client and opens **http://127.0.0.1:8766**. On Windows
you can subsequently double-click **START ANAMORPHIC STUDIO.cmd**. Use
**STOP ANAMORPHIC STUDIO.cmd** to stop the local service and its renders.
Stop another Studio instance before launching a different checkout on port 8766.

The project venv can reuse an existing CUDA PyTorch installation. A fresh venv
without `--system-site-packages` needs its own compatible PyTorch installation.
Uploads and generated media stay in ignored `out/anamorphic-studio/`.

## Export quality

**Match source** is the default. It retains the supported source container,
codec, dimensions, timing, pixel format and compatible audio. H.264/HEVC use
verified lossless codec settings where supported; other codecs use lossless
encoding where available or their highest practical quality setting. Unsupported
exact matches fail explicitly instead of silently changing format.

Effects change the video pixels, so re-encoding is necessary. **Lossless codec
encoding of processed YUV is not mathematically lossless RGB:** returning to
the source chroma sampling/bit depth can discard information. Choose the
separate **Lossless RGB master** for verified processed RGB16 samples. Lossless
files can be substantially larger. ProRes is a high-quality editing codec,
not a mathematically lossless one.

See the [full setup, color/HDR notes, format limits and validation](tools/anamorphic-studio/README.md).
The tested envelope is 8192 pixels per axis / 35.65 MP, with a 64 GiB import
limit. HDR mastering/dynamic metadata is not rebuilt after effects and is
reported explicitly. Alpha-channel match export is currently unsupported.

## Development and checks

```powershell
npm --prefix app run dev
npm --prefix app run build
npm --prefix app run lint
.venv-studio\Scripts\python.exe tools/anamorphic-studio/test_server.py
.venv-studio\Scripts\python.exe tools/anamorphic-studio/test_media_pipeline.py
.venv-studio\Scripts\python.exe tools/anamorphic-studio/test_studio_engine.py
```

The Vite development-only API proxy targets the loopback GPU service. The
production build is served directly by that local service. CI builds/lints the
client and runs CPU media contracts; optional real CUDA/codec integration cases
are documented in the engine guide.

Validated flows include real T6 MOV footage, an 8K 10-bit PQ/BT.2020 test,
variable frame timestamps, copied multi-track audio, rotation, exact RGB16
recovery, cancellation, new-file protection and desktop/mobile browser flows.

## Hosting and license

Deployment is on hold. No Vercel services, optimizer, storage or media proxy are
configured. Any future hosted UI must send media directly to an explicitly
approved GPU backend, without relaying video through Vercel.

Application source is **GPL-2.0-only**; see [LICENSE](LICENSE) and
[provenance / dependency notices](NOTICE.md). Third-party dependencies retain
their respective licenses. Footage, exports, camera ROMs and local environments
are not part of this repository.
