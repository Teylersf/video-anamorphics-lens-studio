"""Reproduce the README comparisons with the real CUDA renderer. GPL-2.0-only.

Downloads only a fixed, hash-pinned CC0 photograph when no local source exists.
No images are painted into the processed frame; Pillow only adds layout/labels.
"""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch

from studio_engine import StudioRenderer, normalize_settings

ROOT = Path(__file__).resolve().parents[2]
URL = 'https://upload.wikimedia.org/wikipedia/commons/0/01/Hong_Kong_night_street_2.jpg'
SOURCE_PAGE = 'https://commons.wikimedia.org/wiki/File:Hong_Kong_night_street_2.jpg'
SOURCE_SHA = 'd242cf64cf244bcd9b4bba96d7f8b8249d726260eee8e67997203a0a7babbe2e'


def font(size, bold=False):
    names = (['C:/Windows/Fonts/segoeuib.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf']
             if bold else ['C:/Windows/Fonts/segoeui.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'])
    for name in names:
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default(size=size)


def board(before, after, label):
    panel_width, gap, margin, header, footer = 768, 16, 24, 104, 64
    panel_height = round(panel_width * before.height / before.width)
    width, height = 2 * panel_width + gap + 2 * margin, panel_height + header + footer
    canvas = Image.new('RGB', (width, height), '#101416')
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 13), 'ANAMORPHIC STUDIO  /  EXTREME DEMONSTRATION', font=font(16, True), fill='#d9a66a')
    draw.text((margin, 49), 'BEFORE', font=font(26, True), fill='#f3f3ed')
    draw.text((margin + panel_width + gap, 49), 'AFTER  /  ' + label, font=font(26, True), fill='#f3f3ed')
    draw.text((margin, 82), 'Same source photograph', font=font(13), fill='#a9b3b5')
    draw.text((margin + panel_width + gap, 82), 'Actual CUDA render - deliberately stronger than default', font=font(13), fill='#a9b3b5')
    size = (panel_width, panel_height)
    canvas.paste(before.resize(size, Image.Resampling.LANCZOS), (margin, header))
    canvas.paste(after.resize(size, Image.Resampling.LANCZOS), (margin + panel_width + gap, header))
    y = header + panel_height + 14
    draw.text((margin, y), '100% amount  |  200% flare  |  exposure unchanged', font=font(17, True), fill='#d9a66a')
    draw.text((margin, y + 27), 'Photo: Wilfredor / Wikimedia Commons / CC0 1.0.  Source and exact settings linked in README.', font=font(13), fill='#8d9b9e')
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'out/readme-extreme-demo/source.jpg')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/images')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise ValueError('CUDA-enabled PyTorch is required to reproduce the actual effects')
    if not args.source.exists():
        args.source.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(URL, headers={'User-Agent': 'VideoAnamorphicsLensStudio/0.2 (https://github.com/Teylersf/video-anamorphics-lens-studio)'})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(8 * 1024 * 1024 + 1)
        if len(data) > 8 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != SOURCE_SHA:
            raise ValueError('CC0 source does not match the pinned photo')
        with args.source.open('xb') as target:
            target.write(data)
    source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if source_hash != SOURCE_SHA:
        raise ValueError('Refusing a different source; this demo pins the credited CC0 photograph')
    with Image.open(args.source) as photo:
        source_size = photo.size
        before = ImageOps.exif_transpose(photo).convert('RGB')
        before.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
    args.output.mkdir(parents=True, exist_ok=True)
    values = np.array(before, dtype=np.uint8)
    tensor = torch.from_numpy(values).to('cuda', dtype=torch.float32).permute(2, 0, 1) / 255
    original = tensor.clone()
    receipt = {'source_page': SOURCE_PAGE, 'source_author': 'Wilfredor', 'source_license': 'CC0-1.0',
               'source_sha256': source_hash, 'original_dimensions': list(source_size),
               'render_dimensions': list(before.size), 'transfer': 'srgb', 'frame_index': 0,
               'renderer_sha256': hashlib.sha256((ROOT / 'tools/anamorphic-studio/studio_engine.py').read_bytes()).hexdigest(),
               'torch_version': torch.__version__, 'gpu': torch.cuda.get_device_name(0), 'renders': {}}
    with torch.inference_mode():
        for name, label in (('blue', 'EXTREME BLUE'), ('amber', 'EXTREME AMBER + FOCUS FALLOFF')):
            preset = ROOT / 'examples/looks' / f'extreme-{name}.json'
            settings = normalize_settings(json.loads(preset.read_text())['settings'])
            rendered = StudioRenderer(settings, transfer='srgb').render(tensor, frame_index=0)
            assert torch.equal(tensor, original), 'Renderer changed the source tensor'
            assert rendered.shape == tensor.shape and bool(torch.isfinite(rendered).all())
            assert 0 <= float(rendered.min()) <= float(rendered.max()) <= 1
            pixels = (rendered.permute(1, 2, 0).clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
            after = Image.fromarray(pixels)
            composite = board(before, after, label)
            output = args.output / f'extreme-{name}-before-after.jpg'
            composite.save(output, format='JPEG', quality=90, subsampling=0, optimize=True)
            difference = np.abs(pixels.astype(np.float32) - values.astype(np.float32)) / 255
            receipt['renders'][name] = {'preset': preset.relative_to(ROOT).as_posix(),
                'image': output.name, 'image_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                'image_bytes': output.stat().st_size, 'image_dimensions': list(composite.size),
                'processed_rgb_sha256': hashlib.sha256(pixels.tobytes()).hexdigest(),
                'mean_absolute_rgb_difference': round(float(difference.mean()), 6),
                'fraction_pixels_changed_over_3_percent': round(float((difference.max(axis=2) > .03).mean()), 6),
                'settings': settings}
    assert hashlib.sha256(args.source.read_bytes()).hexdigest() == source_hash
    receipt['source_file_and_tensor_unchanged'] = True
    (args.output / 'demo-manifest.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({name: {key: value for key, value in data.items() if key != 'settings'}
                      for name, data in receipt['renders'].items()}, indent=2))


if __name__ == '__main__':
    main()
