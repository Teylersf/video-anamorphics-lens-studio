# Provenance and dependencies

This standalone application was extracted from
[Teylersf/canon-t6-cinema](https://github.com/Teylersf/canon-t6-cinema) at commit
`05fb50d231926d79927d2be8b8936bf3afc96cc6` on 2026-11-01. It contains the local
Studio client, original CUDA renderer, media adapter, API, launchers and tests.
It does not depend on Canon firmware, Magic Lantern, camera ROM dumps, Soup
stock recipes or the camera-side custom HEVC experiment.

The application code retains its GPL-2.0-only headers and the GPL v2 license
text in LICENSE. The engine's comments identify its numerical color-science
references; primary-coordinate and matrix calculations are independently
tested. The UI uses system fonts and an original SVG mark.

The README comparisons use Wilfredor's CC0 photograph, rendered by the actual
application. See the [image credits and reproduction notes](docs/images/README.md)
for the original source, license and processing details. These small public
documentation examples are separate from users' private footage and exports.

Dependencies are installed separately rather than vendored. Their own licenses
apply, including React/React DOM, Vite, TypeScript, Lucide, PyTorch, NumPy,
Pillow, PyAV, FastAPI, Uvicorn and their transitive dependencies. FFmpeg and its
enabled codecs have build-dependent licensing requirements. The NVIDIA driver
and CUDA components have their respective terms. This repository's GPL does
not relicense any of those components or the user's footage.
