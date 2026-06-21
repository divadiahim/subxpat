"""Plots for the full-run benchmark (per-stage Z3 calls, mode comparison).

Consumes the merged full-run CSV (one row per benchmark x ET x labelling method,
with per-stage Z3-call counts/time, total time, and area). Produces:

  1. profiling: Z3 calls and time split by execution stage (labelling /
     rewriting / verification), for the baseline 'exact' method;
  2. mode comparison: total Z3 calls, total run time, and approximate area per
     labelling method -- each optimisation shown independently (1A/1B/1C).

Usage (from repo root):
    python scripts/plot_fullrun.py --csv output/report/fullrun_habrok.csv \
        --out-dir output/figure
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
    'axes.titlesize': 12, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'figure.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

STAGE_COLORS = {'labelling': '#4C72B0', 'rewriting': '#DD8452', 'verification': '#55A868'}
# wce (alternative objective) excluded from the paper figures
METHOD_ORDER = ['full', 'exact', 'prefilter', 'simulation', 'warmstart']
METHOD_COLORS = {'full': '#4C72B0', 'exact': '#55A868', 'prefilter': '#8172B3',
                 'simulation': '#C44E52', 'warmstart': '#CCB974', 'wce': '#937860'}


def _save(fig, out_dir, name):
    for fmt in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'{name}.{fmt}'), dpi=300)
    plt.close(fig)
    print(f'  Saved: {name}.pdf')


def _short(b):
    import re
    m = re.search(r'(_i\d+|c\d+)', b)
    return b.split('_')[0] + (m.group(0) if m else '')


def _agg(df):
    """Mean over ETs per (benchmark, method)."""
    g = df.groupby(['benchmark', 'method']).mean(numeric_only=True).reset_index()
    return g


def plot_profiling_stages(df, out_dir):
    """Stacked Z3 calls and time by stage, baseline 'exact' method."""
    d = _agg(df[df.method == 'exact']).sort_values('benchmark')
    if d.empty:
        print('  (no exact rows for profiling)'); return
    x = np.arange(len(d))
    labels = [_short(b) for b in d.benchmark]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    # calls
    bottom = np.zeros(len(d))
    for st in ('labelling', 'rewriting', 'verification'):
        vals = d[f'{st}_calls'].values
        a1.bar(x, vals, bottom=bottom, label=st, color=STAGE_COLORS[st],
               edgecolor='white', linewidth=0.4)
        bottom += vals
    a1.set_ylabel('Z3 solver calls'); a1.set_title('Z3 calls by execution stage')
    # time
    bottom = np.zeros(len(d))
    for st in ('labelling', 'rewriting', 'verification'):
        vals = d[f'{st}_s'].values
        a2.bar(x, vals, bottom=bottom, label=st, color=STAGE_COLORS[st],
               edgecolor='white', linewidth=0.4)
        bottom += vals
    a2.set_ylabel('Z3 solver time (s)'); a2.set_title('Z3 time by execution stage')
    for a in (a1, a2):
        a.set_xticks(x); a.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        a.grid(axis='y', linestyle='--', alpha=0.4); a.legend()
    fig.suptitle('Full-run profiling: Z3 calls and time per stage (baseline labelling)', fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, 'fullrun_profiling_stages')


def plot_mode_metric(df, out_dir, col, ylabel, title, name, logy=False):
    """Grouped bar of one metric across methods, per benchmark."""
    d = _agg(df)
    methods = [m for m in METHOD_ORDER if m in set(d.method)]
    benches = sorted(d.benchmark.unique())
    x = np.arange(len(benches))
    w = 0.8 / max(1, len(methods))

    fig, ax = plt.subplots(figsize=(max(7, 0.7 * len(benches)), 4.2))
    for i, m in enumerate(methods):
        sub = d[d.method == m].set_index('benchmark').reindex(benches)
        ax.bar(x + i * w - 0.4 + w / 2, sub[col].values, width=w, label=m,
               color=METHOD_COLORS[m], edgecolor='white', linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels([_short(b) for b in benches],
                                         rotation=45, ha='right', fontsize=7)
    ax.set_ylabel(ylabel); ax.set_title(title)
    if logy:
        ax.set_yscale('log')
    ax.grid(axis='y', linestyle='--', alpha=0.4); ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, name)


def _panel(ax, d, methods, benches, col, ylabel, title, logy=False):
    x = np.arange(len(benches)); w = 0.8 / max(1, len(methods))
    for i, m in enumerate(methods):
        sub = d[d.method == m].set_index('benchmark').reindex(benches)
        ax.bar(x + i * w - 0.4 + w / 2, sub[col].values, width=w, label=m,
               color=METHOD_COLORS[m], edgecolor='white', linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels([_short(b) for b in benches],
                                         rotation=45, ha='right', fontsize=6.5)
    ax.set_ylabel(ylabel); ax.set_title(title)
    if logy:
        ax.set_yscale('log')
    ax.grid(axis='y', linestyle='--', alpha=0.4); ax.legend(fontsize=8)


def plot_1a_prefilter(df, out_dir):
    """1A: full vs exact(=partial) vs prefilter -- labelling calls, total time, area."""
    d = _agg(df); benches = sorted(d.benchmark.unique())
    methods = ['full', 'exact', 'prefilter']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    _panel(axes[0], d, methods, benches, 'labelling_calls', 'Labelling Z3 calls',
           'Labelling calls', logy=True)
    _panel(axes[1], d, methods, benches, 'total_s', 'Total run time (s)', 'Total run time')
    _panel(axes[2], d, methods, benches, 'approx_area', 'Approximate area (cells)',
           'Approximate area (quality)')
    fig.suptitle('1A -- Structural pre-filter (zero labelling Z3 calls): full vs exact vs pre-filter',
                 fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, 'fullrun_1a_prefilter')


def plot_1b_simulation(df, out_dir):
    """1B: exact vs simulation -- labelling calls, total time, area."""
    d = _agg(df); benches = sorted(d.benchmark.unique())
    methods = ['exact', 'simulation']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    _panel(axes[0], d, methods, benches, 'labelling_calls', 'Labelling Z3 calls',
           'Labelling calls', logy=True)
    _panel(axes[1], d, methods, benches, 'total_s', 'Total run time (s)', 'Total run time')
    _panel(axes[2], d, methods, benches, 'approx_area', 'Approximate area (cells)',
           'Approximate area (quality)')
    fig.suptitle('1B -- Simulation upper bound: exact vs simulation',
                 fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, 'fullrun_1b_simulation')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv', default='output/report/fullrun_habrok.csv')
    p.add_argument('--out-dir', default='output/figure')
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    print(f'Loaded {len(df)} rows from {args.csv}')

    plot_profiling_stages(df, args.out_dir)
    plot_1a_prefilter(df, args.out_dir)
    plot_1b_simulation(df, args.out_dir)
    print(f'\nFigures in {args.out_dir}')


if __name__ == '__main__':
    main()
