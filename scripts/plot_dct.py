"""Plots for the approximate-DCT study (WP5 area, WP6 image quality, Pareto).

Usage (from repo root):
    python scripts/plot_dct.py --area output/report/dct_area.csv \
        --quality output/report/dct_image_quality.csv --out-dir output/figure
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 11,
    'axes.titlesize': 12, 'legend.fontsize': 9, 'figure.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})
COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def _save(fig, out_dir, name):
    for fmt in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'{name}.{fmt}'), dpi=300)
    plt.close(fig)
    print(f'  Saved: {name}.pdf')


def plot_area(area: pd.DataFrame, out_dir):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for (bm, grp), c in zip(area.groupby('benchmark'), COLORS):
        grp = grp.sort_values('et')
        label = bm.split('_')[0]  # dct4c0
        ax.plot(grp['et'], grp['area_reduction_pct'], 'o-', color=c, label=label)
    ax.set_xlabel('Error threshold ET'); ax.set_ylabel('Area reduction (%)')
    ax.set_title('Approximate DCT: area reduction vs error threshold')
    ax.grid(linestyle='--', alpha=0.4); ax.legend(title='coefficient')
    fig.tight_layout(); _save(fig, out_dir, 'dct_area_vs_et')


def plot_quality(q: pd.DataFrame, out_dir):
    qa = q[q['et'] != 'exact'].copy()
    qa['et'] = qa['et'].astype(int)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for (name, grp), c in zip(qa.groupby('image'), COLORS):
        grp = grp.sort_values('et')
        a1.plot(grp['et'], grp['psnr'], 'o-', color=c, label=name)
        a2.plot(grp['et'], grp['ssim'], 'o-', color=c, label=name)
    a1.set_xlabel('Error threshold ET'); a1.set_ylabel('PSNR (dB)')
    a1.set_title('Reconstruction PSNR vs ET')
    a2.set_xlabel('Error threshold ET'); a2.set_ylabel('SSIM')
    a2.set_title('Reconstruction SSIM vs ET')
    for a in (a1, a2):
        a.grid(linestyle='--', alpha=0.4); a.legend()
    fig.tight_layout(); _save(fig, out_dir, 'dct_quality_vs_et')


def plot_pareto(area: pd.DataFrame, q: pd.DataFrame, out_dir):
    # mean area reduction across coefficients per ET vs mean PSNR across images per ET
    a_by_et = area.groupby('et')['area_reduction_pct'].mean()
    qa = q[q['et'] != 'exact'].copy(); qa['et'] = qa['et'].astype(int)
    p_by_et = qa.groupby('et')['psnr'].mean()
    ets = sorted(set(a_by_et.index) & set(p_by_et.index))
    if not ets:
        print('  (no overlapping ET for Pareto)'); return
    xs = [a_by_et[e] for e in ets]
    ys = [p_by_et[e] for e in ets]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(xs, ys, 'o-', color='#4C72B0')
    for e, x, y in zip(ets, xs, ys):
        ax.annotate(f'ET={e}', (x, y), textcoords='offset points', xytext=(6, 4), fontsize=8)
    ax.set_xlabel('Mean area reduction (%)'); ax.set_ylabel('Mean PSNR (dB)')
    ax.set_title('Area--quality Pareto (approximate DCT)')
    ax.grid(linestyle='--', alpha=0.4)
    fig.tight_layout(); _save(fig, out_dir, 'dct_pareto')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--area', default='output/report/dct_area.csv')
    p.add_argument('--quality', default='output/report/dct_image_quality.csv')
    p.add_argument('--out-dir', default='output/figure')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    area = pd.read_csv(args.area)
    plot_area(area, args.out_dir)
    if os.path.exists(args.quality):
        q = pd.read_csv(args.quality)
        plot_quality(q, args.out_dir)
        plot_pareto(area, q, args.out_dir)
    print(f'\nFigures in {args.out_dir}')


if __name__ == '__main__':
    main()
