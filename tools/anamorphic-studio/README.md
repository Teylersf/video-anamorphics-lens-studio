# Anamorphic Studio — local CUDA / Metal / CPU app

On Windows, double-click **START ANAMORPHIC STUDIO.cmd** in the repository
root. On Apple Silicon Mac, first follow [the Mac setup guide](../../docs/MACOS.md),
then open **START ANAMORPHIC STUDIO.command**. Use **http://127.0.0.1:8766**.
The corresponding **STOP ANAMORPHIC STUDIO.cmd / .command** stops this service and
cancels its renders. No deployment is performed. Uploaded media, previews and
exports stay in ignored `out/anamorphic-studio/` on this computer.

Import a video, choose Modern/Vintage/Clean or Bypass, move through the source
timeline, then adjust the lens controls. The frame preview uses the same lens
renderer as export at a smaller preview resolution. Compare before/after with
the draggable wipe, render the first three seconds for playback, or export the
whole source. Save/load a look as a small JSON file. Editing is locked during a
render so a completed preview cannot silently show older settings.

## Default export: Match source

The default preserves the source container, video codec, raster dimensions,
bit depth/chroma, display rotation, pixel aspect, frame presentation timing
(including VFR), and all compatible original audio streams. The adapter checks
the result before publishing a new filename. Audio packet payload hashes and
their timestamp extents are compared when audio is copied. Originals and
existing output paths are never overwritten.

H.264/HEVC use NVIDIA's verified lossless codec mode where supported on CUDA.
On Mac, Match source uses software libx264/libx265 lossless encoding rather
than VideoToolbox. Other
supported codecs use lossless encoding where available, or their highest
practical quality setting. An unsupported exact match produces an explicit
error instead of silently changing formats. **Lossless encoding of processed
YUV samples is different from mathematically lossless RGB:** converting the
effect's RGB result back into the original chroma sampling/bit depth can lose
information. Lossless files can be substantially larger than their sources.

Other explicit export choices:

- **HEVC / H.265:** efficient high-quality MP4. Incompatible source audio is
  converted to AAC with a reported warning. On Mac, supported VideoToolbox
  encoding is capability-probed; unavailable formats use a reported software
  fallback. VideoToolbox delivery is not mathematically lossless.
- **ProRes master:** high-quality editing MOV, not mathematically lossless.
- **Lossless RGB master:** FFV1/MKV with 16-bit RGB samples. Every decoded
  processed sample is hash-verified. This is a larger editing/archive file,
  not a restoration of detail lost in the source.

The output result panel reports the actual codec, pixel format, quality mode,
audio handling and any conversion warnings. Effects require re-encoding the
video; the modified movie cannot be byte-identical to its original.

## Lens engine

The CUDA, Metal/MPS and CPU renderer includes relative linear-light, soft-knee highlight
selection; smooth normalized multiscale horizontal flare kernels; blue/amber
coating color; chromatic dispersion; reflected lens ghosts; anisotropic bloom;
warm halation; strip-sampled radial distortion and edge color separation;
vignette; exposure/contrast/saturation; and deterministic animated grain.
Controls scale with image dimensions, and no full-clip frame buffer is kept.

The default `STUDIO_DEVICE=auto` selects an available CUDA device, then Metal
MPS, then CPU. Set `STUDIO_DEVICE=cpu`, `mps` or `cuda` before launching to choose
explicitly; an unavailable explicit accelerator fails rather than silently
changing that choice. Rendering failures propagate instead of automatically
moving a failed frame to another backend. On MPS, lens/color operations stay
on Metal; deterministic integer grain noise is computed in bounded CPU tiles
and transferred as floating noise. This is not a global unsupported-operation
fallback. Floating results can differ slightly between backends; byte identity
across CUDA, MPS and CPU is not promised.

Oval bokeh/focus falloff is optional and uses a protected horizontal focus band.
It is an artistic mask, **not estimated scene depth or a calibrated optical
lens model**. It cannot reconstruct hidden background detail or extra field of
view. There is no automatic global squeeze, desqueeze, letterbox or upscale.

Input/output support BT.709, sRGB, PQ and HLG transfer functions. HDR exports
retain the source signal's transfer/primaries/matrix/range/bit depth where the
codec/container supports them. Mastering-display, content-light and dynamic
HDR/Dolby Vision side data are **not rebuilt after effects**; the result reports
this limitation. HDR previews use an approximate SDR shoulder, with known
source primaries converted for display. SD/P3 preview primaries are supported.
Unspecified source matrix/transfer receives a reported processing assumption.
Camera Log/RAW development and calibrated HDR mastering are separate work.

## Resolution, formats and local storage

The CUDA-tested processing envelope is **up to 8192 pixels per axis and 35.65 MP**,
including UHD 8K 7680×4320 and its portrait orientation. Export preserves the
source resolution; processing a 1080p clip does not create captured 8K detail.
Import accepts common MOV/MP4, Matroska/WebM, AVI, MXF, MPEG/TS, ASF and Ogg
video that the installed decoders can read. Exact match still depends on the
available encoder/container combination. Alpha-channel match export is
explicitly rejected rather than silently discarded. Damaged/missing/nonmonotonic
timestamps and unsupported orientation are reported as errors.

The local import limit is 64 GiB per file. Keep enough free disk space for the
uploaded copy, proxy, temporary encode and final export. Generated media is
retained locally; it is not uploaded to GitHub or a cloud service. The queue
serializes GPU jobs. Cancelling terminates only the owned export-worker tree;
the two-hour default deadline also stops a stuck worker before publication.

## Setup / development

For Apple Silicon, use [the Mac guide](../../docs/MACOS.md) and run
`bash "SETUP MAC.command"` from the repository root. It verifies native ARM64
Python 3.13 and installs the compatibility-pinned PyTorch 2.11.0 environment.
Then run `bash "START ANAMORPHIC STUDIO.command"`. Setup does not start a
persistent service. Metal/VideoToolbox throughput requires actual Mac testing;
conditional tests and dependency checks alone do not establish it.

The tested machine uses Windows, Python 3.13, PyTorch 2.11.0+cu126, RTX 3090
24 GB, driver 621.01, PyAV 17.0.0, system FFmpeg 7.0.2 and Node 22.19.0.
PyAV's codec libraries and the system FFmpeg executable are separate; both are
validated by tests. Install a CUDA-enabled PyTorch appropriate for the GPU
using the [official selector](https://pytorch.org/get-started/locally/), plus
FFmpeg/ffprobe on PATH. A project venv can reuse an existing CUDA install:

```powershell
python -m venv --system-site-packages .venv-studio
.venv-studio\Scripts\python.exe -m pip install -r tools/anamorphic-studio/requirements.txt
npm --prefix app ci
python tools/anamorphic-studio/launch.py
```

The launcher automatically uses `.venv-studio` when present, builds the client
and starts a loopback-only service. The frontend is Vite/React; it is not a
Next.js server. For frontend development, run the service and
`npm --prefix app run dev`; the development-only proxy forwards `/api` to the
local service. The built frontend is served by that service on port 8766.

```powershell
npm --prefix app run build
npm --prefix app run lint
.venv-studio\Scripts\python.exe tools/anamorphic-studio/test_server.py
.venv-studio\Scripts\python.exe tools/anamorphic-studio/test_media_pipeline.py
.venv-studio\Scripts\python.exe tools/anamorphic-studio/test_studio_engine.py
# Optional real codec/CUDA integration cases:
$env:STUDIO_GPU_TESTS = '1'
.venv-studio\Scripts\python.exe tools/anamorphic-studio/test_media_pipeline.py
```

## Verified end to end

The physical rendering and large-format results in this section were obtained
on the Windows/CUDA configuration above. CPU engine tests also run; Metal and
VideoToolbox cases are conditional on actual backend availability and are not
claimed as hardware-verified by these Windows results.

- The owner's complete 349-frame T6 clip exported as 1920×1080 MOV/H.264 at
  exact 30000/1001 timing, original 8-bit 4:2:0/full range and verified PCM packet
  copying. The source SHA-256 remained unchanged. Result: 374,155,076 bytes.
- An actual 7680×4320, 10-bit PQ/BT.2020 synthetic source passed full GPU
  rendering with Vintage and oval blur, same-format lossless HEVC export,
  independent FFmpeg decoding and source/timestamp/color-tag checks. Two frames
  at 24000/1001; 25,401,853-byte output. This tests PC processing, not camera
  capture or sustained throughput.
- GPU effect tests cover smooth flare shape, tint, focus-mask behavior, grain,
  malformed inputs, transfer round trips and independently derived primary
  matrices. A full 8K renderer frame used about 2.7 GiB peak allocated memory;
  this excludes codec/driver/host allocations.
- API tests cover upload preservation, controlled asset paths, local origin and
  mutation headers, strict numeric controls, queue/cancel races and cover-art
  stream selection. Codec tests cover VFR, 10-bit HDR, rotation, multiple copied
  audio tracks, lossless RGB recovery, cancellation and new-file collisions.

## Hosting boundary

**Local only; deployment is on hold.** No Vercel configuration, optimizer,
Blob, Edge Config, analytics or media proxy is included. The small static UI
could later be hosted separately, but all uploads, previews and exports must
travel directly between the browser and an explicitly approved GPU origin.
Media must not be relayed through Vercel. Public hosting additionally needs
authentication and an approved compute/storage plan; neither is enabled here.
