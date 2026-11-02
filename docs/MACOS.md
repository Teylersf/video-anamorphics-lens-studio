# Anamorphic Studio on Mac

The primary target is an Apple Silicon Mac (M-series) running native ARM64 Python. Effects use PyTorch's Metal/MPS backend when available, with a slower CPU fallback. VideoToolbox provides supported HEVC delivery and H.264 preview encoding; the separate software lossless options preserve their stated codec/pixel contracts. Encoding support and speed depend on the particular Mac and source. No real-time, 8K-throughput or GPU-performance promise is made without testing that hardware.

On Mac, **H.264/HEVC Match source uses software libx264/libx265 lossless encoding**. VideoToolbox is reserved for supported high-quality HEVC delivery and H.264 previews, because its delivery mode does not promise lossless encoding. Hardware capability is probed and a software fallback is reported when needed. The separate Lossless RGB master uses software FFV1 with verified processed RGB16 samples. Lossless YUV codec encoding still differs from mathematically lossless RGB conversion.

## Requirements

- Recommended installation: macOS 15 or newer, which is the current supported baseline for Homebrew. The pinned PyTorch 2.11.0 MPS requirements and PyAV 17 ARM64 wheels permit macOS 14 or newer with separately maintained compatible dependencies.
- Native ARM64 Python 3.13, Node.js 22.12 or newer within the 22.x series, and `ffmpeg-full` with ffprobe plus the `zscale` and `tonemap` filters.
- Apple's Xcode command-line tools. Run `xcode-select --install` in Terminal if they are missing, then complete Apple's installer.
- Enough disk space for the imported source, temporary processing files and completed exports. Lossless masters can be much larger than the source.

[Apple's PyTorch instructions](https://developer.apple.com/metal/pytorch/) explicitly describe PyTorch 2.11.0, Apple Silicon and macOS 14+. This project pins that release for compatibility rather than following a changing latest release. The [official PyTorch 2.11.0 files](https://pypi.org/project/torch/2.11.0/#files) include native CPython 3.13 ARM64 wheels. [PyAV 17.0.0](https://pypi.org/project/av/17.0.0/#files) supplies an ARM64 macOS 14+ CPython 3.11 `abi3` wheel, also usable by Python 3.13.

## First setup

1. Keep the project in a writable local folder. Obtain Homebrew from its [official installation instructions](https://docs.brew.sh/Installation) if needed. The project does not install Homebrew or request administrator access for you.
2. In a native Terminal session, install the prerequisites explicitly:

   ```bash
   brew install python@3.13 node@22 ffmpeg-full
   ```

   These formulae are documented by Homebrew: [Python 3.13](https://formulae.brew.sh/formula/python@3.13), [Node 22](https://formulae.brew.sh/formula/node@22), [FFmpeg full](https://formulae.brew.sh/formula/ffmpeg-full). `ffmpeg-full` is keg-only. Each launcher puts `/opt/homebrew/opt/ffmpeg-full/bin` first in its own process PATH, ahead of the normal Homebrew binaries. There is no global linking, uninstall or change to shell startup files.

   The full build is needed for HDR viewing proxies/posters: the regular Homebrew FFmpeg 9 build can omit `zscale`. The [official full formula](https://github.com/Homebrew/homebrew-core/blob/HEAD/Formula/f/ffmpeg-full.rb) depends on `zimg` and enables `libzimg`. Setup checks the actual selected binary for both `zscale` and `tonemap`, and stops with an installation hint if either is missing. Tone mapping applies to the viewing proxy/poster; it never overwrites the uploaded source. An already installed regular FFmpeg can remain in place.
3. Open Terminal in the project folder and run:

   ```bash
   bash "SETUP MAC.command"
   ```

   Setup checks native architecture, versions and HDR preview filters, creates `.venv-studio`, installs `requirements-macos.txt`, checks imports and small Modern/Vintage effect renders on the selected backend, and builds the browser interface. It downloads dependencies but does not start Studio. Rerunning setup uses the existing compatible environment; it refuses to delete or replace an incompatible one.

If the project was obtained as a ZIP and the `.command` files are not executable, enable these three files once from the project folder:

```bash
chmod +x "SETUP MAC.command" "START ANAMORPHIC STUDIO.command" "STOP ANAMORPHIC STUDIO.command"
```

Alternatively, run any wrapper with `bash "filename.command"`. A separately installed native Python 3.13 can be selected explicitly for setup:

```bash
PYTHON3="/absolute/path/to/python3.13" bash "SETUP MAC.command"
```

Avoid Terminal's **Open using Rosetta** setting and Intel Python packages. The setup deliberately refuses Rosetta and Intel architectures. An existing Windows `.venv-studio` cannot be reused on Mac; move that folder aside manually, then rerun setup.

## Start and stop

Double-click **START ANAMORPHIC STUDIO.command**, or run it with Bash. The Python launcher builds the interface and opens `http://127.0.0.1:8766`. Footage uploads and downloads stay between your browser and the local service. The project does not deploy a site or use Vercel services.

Double-click **STOP ANAMORPHIC STUDIO.command** to request shutdown through that local service. This does not delete imports or exports. Setup and stop never start the server. Diagnostics are written under the project's ignored `out/anamorphic-studio` folder. See [the Studio guide](../tools/anamorphic-studio/README.md) for controls, export contracts and source-preservation details.

The top status indicator should identify the selected backend. Setup checks actual small effect renders on the selected backend; it does not prove every resolution, encoder or source combination works. Test a short clip on the actual Mac before committing to a large export. Full-resolution processing retains source dimensions and does not create additional camera detail.

The default `STUDIO_DEVICE=auto` chooses CUDA, then MPS, then CPU according to availability. To request a specific backend from Terminal, use `STUDIO_DEVICE=mps bash "START ANAMORPHIC STUDIO.command"` or `STUDIO_DEVICE=cpu bash "START ANAMORPHIC STUDIO.command"`. Stop an existing instance before changing its backend. An explicitly requested unavailable accelerator fails helpfully; a rendering error does not silently transfer that frame to CPU. MPS computes deterministic integer grain noise in bounded CPU tiles while the lens/color operations remain on Metal. Results need not be byte-identical across backends.

Intel Mac CPU operation may be possible with separately compatible PyTorch and Python versions, but it is outside this ARM64 setup. PyTorch 2.11.0 does not provide the same Intel Mac wheel path; these scripts do not silently downgrade packages or imply Intel GPU acceleration.

## Validation scope

The shell scripts can be syntax-checked with `bash -n`. Dependency metadata and setup guards can be verified without owning a Mac; actual MPS/VideoToolbox execution requires compatible Apple hardware. [GitHub's standard runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners) lists `macos-15` as an ARM64 M1 runner. Architecture alone is not evidence of GPU access: CI must query `torch.backends.mps.is_available()` and perform an actual operation before treating MPS as verified. CPU tests on a virtual runner are not a Metal performance benchmark.

## 8K validation on Apple M4

Measured on an Apple M4 Mac mini with a 10-core GPU and 16 GB unified memory on 2026-11-01. Native Metal/MPS rendering was selected with generic MPS CPU fallback disabled. Tests used generated footage, not camera captures or upscaled lower-resolution clips.

| Check | Verified result |
|---|---|
| Modern effects, one 7680×4320 frame | 1.61 s effects-only wall time; sampled Metal driver memory about 5.0 GiB |
| Default Match source, 24 native 7680×4320 frames at 24 fps, 10-bit HEVC | Software lossless libx265; 120.56 s including export verification, about 0.20 export fps; 148 MB output |
| HEVC delivery, two native 8K frames | Actual Apple VideoToolbox encoder; 5.08 s including startup and verification |

Independent FFmpeg decoding checked every saved frame. Source dimensions, bit depth, presentation timestamps and compatible original audio were preserved in the Match source test, and the source SHA-256 remained unchanged. The driver-memory observation is not total system RAM or an allocator peak API. The two-frame hardware result is not a sustained throughput measurement. Default lossless export is substantially slower than delivery encoding; these results do not establish real-time 8K processing.

The opt-in accelerator test exercises full-resolution Vintage effects with focus falloff and checks that input pixels stay unchanged:

```bash
.venv-studio/bin/python tools/anamorphic-studio/test_studio_engine.py MpsTests.test_full_8k_accelerator_memory --8k
```

Ordinary CI skips that large allocation. Use one 8K render at a time and retain the normal Metal allocator safeguards.
