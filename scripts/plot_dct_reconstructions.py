"""Assemble the DCT reconstruction grid figure (Original / ET=16 / ET=64 per image).

Reads the per-image PNGs saved by dct_image_pipeline.py --save-images and the
PSNR/SSIM CSV, and lays them out as a rows=images x cols=ET grid with quality
captions. This is the figure shown in the thesis (dct_reconstructions.png) and on
the presentation slide.

Usage (from repo root):
    python scripts/plot_dct_reconstructions.py \
        --img-dir output/dct_images \
        --quality output/report/dct_image_quality.csv \
        --ets 16 64 --out output/figure/dct_reconstructions.png

Use --images circles  to emit a single-row strip (slide-friendly).
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

plt.rcParams.update({'font.family': 'serif', 'figure.dpi': 300,
                     'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05})


def _load(img_dir, name, et):
    fn = f'{name}_orig.png' if et == 'orig' else f'{name}_et{et}.png'
    path = os.path.join(img_dir, fn)
    if not os.path.exists(path):
        return None
    return np.asarray(Image.open(path).convert('L'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--img-dir', default='output/dct_images')
    p.add_argument('--quality', default='output/report/dct_image_quality.csv')
    p.add_argument('--ets', nargs='+', type=int, default=[16, 64])
    p.add_argument('--images', nargs='+', default=None,
                   help='subset/order of image names (default: all found)')
    p.add_argument('--out', default='output/figure/dct_reconstructions.png')
    args = p.parse_args()

    q = pd.read_csv(args.quality)
    q['et'] = q['et'].astype(str)

    def caption(name, et):
        row = q[(q['image'] == name) & (q['et'] == str(et))]
        if row.empty:
            return ''
        return f"{row.iloc[0]['psnr']:.1f} dB / SSIM {row.iloc[0]['ssim']:.2f}"

    names = args.images or sorted({fn.split('_orig.png')[0]
                                   for fn in os.listdir(args.img_dir)
                                   if fn.endswith('_orig.png')})
    cols = ['orig'] + list(args.ets)
    col_titles = ['Original'] + [f'ET={e}' for e in args.ets]

    nrows, ncols = len(names), len(cols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.3 * nrows),
                             squeeze=False)
    for r, name in enumerate(names):
        for c, et in enumerate(cols):
            ax = axes[r][c]
            img = _load(args.img_dir, name, et)
            if img is not None:
                ax.imshow(img, cmap='gray', vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=16)
            if c == 0:
                ax.set_ylabel(name, fontsize=15)
            if et != 'orig':
                ax.set_xlabel(caption(name, et), fontsize=12)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=300)
    print(f'Saved {args.out}  ({nrows}x{ncols} grid: {", ".join(names)})')


if __name__ == '__main__':
    main()
