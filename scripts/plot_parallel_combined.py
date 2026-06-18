"""Combined parallel-labelling figure: one plot, all three execution modes,
with exact wall-clock values annotated.

Merges the previous two parallel plots (grouped times + orig-vs-improved) into a
single grouped-bar visualisation per benchmark family: sequential, original
parallel, and improved parallel wall-clock time, with the exact seconds printed
above each bar.

Usage (from repo root):
    python scripts/plot_parallel_combined.py \
        --csv output/report/labeling_benchmark_habrok.csv --out-dir output/figure
"""

import argparse
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10, 'axes.labelsize': 11,
    'axes.titlesize': 12, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'figure.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})
COLORS = {'sequential': '#4C72B0', 'original_parallel': '#DD8452', 'improved_parallel': '#55A868'}


def _family(n):
    m = re.match(r'^([a-z_]+?)_i', n)
    return m.group(1) if m else n


def _short(n):
    m = re.search(r'_i(\d+)', n)
    return f'i{m.group(1)}' if m else n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='output/report/labeling_benchmark_habrok.csv')
    p.add_argument('--out-dir', default='output/figure')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    g = (df.groupby('benchmark', sort=False)
         .agg(num_inputs=('num_inputs', 'first'),
              sequential_s=('sequential_s', 'mean'),
              original_parallel_s=('original_parallel_s', 'mean'),
              improved_parallel_s=('improved_parallel_s', 'mean'))
         .reset_index())
    g['family'] = g['benchmark'].apply(_family)
    families = {f: grp.sort_values('num_inputs') for f, grp in g.groupby('family')}
    n = len(families)

    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 4.6), sharey=True)
    if n == 1:
        axes = [axes]
    series = [('sequential_s', 'sequential', 'Sequential'),
              ('original_parallel_s', 'original_parallel', 'Original par.'),
              ('improved_parallel_s', 'improved_parallel', 'Improved par.')]

    for ax, (fam, fdf) in zip(axes, sorted(families.items())):
        x = np.arange(len(fdf)); w = 0.27
        for off, (col, key, lab) in zip((-w, 0, w), series):
            bars = ax.bar(x + off, fdf[col].values, width=w,
                          label=lab if ax is axes[0] else '_nolegend_',
                          color=COLORS[key], edgecolor='white', linewidth=0.4)
            for b, v in zip(bars, fdf[col].values):
                ax.text(b.get_x() + b.get_width() / 2, v * 1.05, f'{v:.0f}',
                        ha='center', va='bottom', fontsize=5.5, rotation=90,
                        color=COLORS[key])
        ax.set_xticks(x); ax.set_xticklabels([_short(b) for b in fdf.benchmark], fontsize=8)
        ax.set_title(fam, fontsize=10, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    axes[0].set_yscale('log'); axes[0].set_ylabel('Wall-clock time (s, log)')
    axes[0].legend(fontsize=8)
    fig.suptitle('Gate labelling time: sequential vs original vs improved parallel '
                 '(exact seconds annotated)', fontsize=12)
    fig.tight_layout()
    for fmt in ('pdf', 'png'):
        fig.savefig(os.path.join(args.out_dir, f'labeling_parallel_combined.{fmt}'), dpi=300)
    plt.close(fig)
    print(f'  Saved labeling_parallel_combined.pdf in {args.out_dir}')


if __name__ == '__main__':
    main()
