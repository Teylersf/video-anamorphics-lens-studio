# README comparison images

The before/after figures use the real CUDA application renderer, with the
deliberately exaggerated settings in
[Extreme Blue](../../examples/looks/extreme-blue.json) and
[Extreme Amber](../../examples/looks/extreme-amber.json). They are documentation
assets, not the normal default look. The application renderer was not modified
to produce them.

## Source photograph and license

- Photograph: [*Hong Kong night street 2*](https://commons.wikimedia.org/wiki/File:Hong_Kong_night_street_2.jpg).
- Author: [Wilfredor](https://commons.wikimedia.org/wiki/User:Wilfredor).
- Source license: [CC0 1.0 Universal Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/),
  as stated by the copyright holder on the photograph's Commons page.
- Original image: 2848×2483 JPEG. Its SHA-256 is
  `d242cf64cf244bcd9b4bba96d7f8b8249d726260eee8e67997203a0a7babbe2e`.

The JPEG is decoded, oriented and scaled once to 1280×1116. That same sRGB
image feeds the before panel and the actual `StudioRenderer` after panel.
Both use frame index zero. The comparison layout resizes both panels equally
and adds labels/credits; nothing is painted into the processed photograph.
Camera footage and personal media are not used.

Blue has no focus blur. Amber enables the renderer's horizontal focus-band
falloff; this is not estimated depth or a reconstructed optical lens. Exposure
is zero in both looks. No cropping, desqueeze or aspect-ratio change is applied.

## Reproduce

From the repository root with the application's CUDA dependencies installed:

```powershell
.venv-studio\Scripts\python.exe tools/anamorphic-studio/render_readme_demo.py
```

The script downloads the fixed, hash-pinned CC0 source into ignored
`out/readme-extreme-demo/` if needed. It applies each checked JSON preset and
rebuilds the documentation JPEGs. The renderer's source tensor and original
photo are verified unchanged; output dimensions, range and finite samples
are checked. [demo-manifest.json](demo-manifest.json) records the source and
renderer hashes, settings, image hashes and RGB difference statistics. Those
statistics show that processing occurred; they do not measure optical realism.

Fonts and GPU/library rounding can change reproduced JPEG hashes on other
systems. The checked-in examples were made with the environment recorded in
the manifest. Each 1600×838 figure is under 500 KB, and these assets are outside
`app/public`, so they are not included in the local web app's build.
