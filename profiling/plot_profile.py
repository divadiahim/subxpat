"""Generate thesis-quality plots from the phase profiling CSV.

Reads the merged phase_breakdown CSV and writes figures to an output directory:

  1. stacked_bar.pdf       — stacked bar: time per phase, seq vs par, per benchmark
  2. exec_speedup.pdf      — bar chart: Z3 execution phase speedup per benchmark
  3. total_speedup.pdf     — bar chart: total labelling time speedup per benchmark
  4. phase_pie_<bench>.pdf — pie chart: phase breakdown for each benchmark (both modes)
  5. phase_dominance.pdf   — line chart: Z3 exec % of total vs circuit size

Usage (from repo root):
    python profiling/plot_profile.py
    python profiling/plot_profile.py --csv profiling/results/phase_breakdown_habrok.csv
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

PHASES = ['setup_s', 'gen_s', 'exec_s', 'import_s', 'overhead_s']
PHASE_LABELS = {
    'setup_s':    'Setup',
    'gen_s':      'Script gen',
    'exec_s':     'Z3 solving',
    'import_s':   'CSV import',
    'overhead_s': 'Overhead',
}
PHASE_COLORS = {
    'setup_s':    '#4C72B0',
    'gen_s':      '#DD8452',
    'exec_s':     '#C44E52',
    'import_s':   '#55A868',
    'overhead_s': '#BBBBBB',
}

PIE_MERGE_THRESHOLD = 0.03
FIG_DIR = os.path.join('profiling', 'results', 'figures')
DEFAULT_CSV = os.path.join('profiling', 'results', 'phase_breakdown_habrok.csv')


def _save(fig, out_dir, name):
    for fmt in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'{name}.{fmt}'), dpi=300)
    plt.close(fig)
    print(f'  Saved: {name}.pdf')


def load_csv(path: str):
    import csv
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            for col in PHASES + ['total_s', 'exec_pct', 'gen_pct']:
                if col in row:
                    row[col] = float(row[col])
            rows.append(row)
    return rows


def pivot(rows):
    """Return {benchmark: {mode: row}}."""
    d = {}
    for r in rows:
        d.setdefault(r['benchmark'], {})[r['mode']] = r
    return d


def _extract_inputs(name: str) -> int:
    import re
    m = re.search(r'_i(\d+)', name)
    return int(m.group(1)) if m else 0


def plot_stacked_bars(data: dict, out_dir: str):
    benchmarks = sorted(data.keys(), key=_extract_inputs)
    n = len(benchmarks)
    modes = ['sequential', 'parallel']

    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(7, n * 1.2), 5))

    for m_idx, mode in enumerate(modes):
        offset = (m_idx - 0.5) * width
        bottoms = np.zeros(n)
        for phase in PHASES:
            vals = np.array([data[bm].get(mode, {}).get(phase, 0.0) for bm in benchmarks])
            label = PHASE_LABELS[phase] if m_idx == 0 else '_nolegend_'
            ax.bar(x + offset, vals, width, bottom=bottoms,
                   color=PHASE_COLORS[phase], label=label,
                   edgecolor='white', linewidth=0.5)
            bottoms += vals

        for i, bm in enumerate(benchmarks):
            total = data[bm].get(mode, {}).get('total_s', 0.0)
            ax.text(x[i] + offset, bottoms[i] + 0.05, f'{total:.1f}s',
                    ha='center', va='bottom', fontsize=7,
                    color='#333333', fontweight='bold')

    tick_positions = []
    tick_labels_list = []
    for i in range(n):
        tick_positions.extend([x[i] - width/2, x[i] + width/2])
        tick_labels_list.extend(['seq', 'par'])

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels_list, fontsize=7)

    for i, bm in enumerate(benchmarks):
        ax.text(x[i], -0.13, bm, ha='center', va='top',
                fontsize=7, fontweight='bold',
                transform=ax.get_xaxis_transform())

    ax.set_ylabel('Time (s)')
    ax.set_title('Labelling pipeline phase breakdown (sequential vs parallel)')
    ax.legend(loc='upper left', fontsize=8, ncol=len(PHASES))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()
    _save(fig, out_dir, 'stacked_bar')


def plot_speedup(data: dict, out_dir: str, phase: str, title: str, filename: str):
    benchmarks = sorted(data.keys(), key=_extract_inputs)
    speedups = []
    for bm in benchmarks:
        seq = data[bm].get('sequential', {}).get(phase, 0.0)
        par = data[bm].get('parallel', {}).get(phase, 0.0)
        speedups.append(seq / par if par > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(max(6, len(benchmarks) * 1.0), 4))
    bars = ax.bar(benchmarks, speedups, color='#4C72B0', edgecolor='white', linewidth=0.5)

    for bar, val in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f'{val:.2f}×',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.axhline(1.0, color='grey', linestyle='--', linewidth=0.8, label='1× (no speedup)')
    ax.set_ylabel('Speedup (seq / par)')
    ax.set_title(title)
    ax.set_ylim(0, max(speedups) * 1.2 + 0.5)
    ax.tick_params(axis='x', rotation=30)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, filename)


def plot_pies(data: dict, out_dir: str):
    for bm, modes in data.items():
        n_modes = len(modes)
        fig, axes = plt.subplots(1, n_modes, figsize=(5 * n_modes, 4.5))
        if n_modes == 1:
            axes = [axes]

        for ax, (mode, row) in zip(axes, sorted(modes.items())):
            total = row.get('total_s', 1e-9)
            vals = [row.get(p, 0.0) for p in PHASES]
            colors = [PHASE_COLORS[p] for p in PHASES]
            labels = [PHASE_LABELS[p] for p in PHASES]

            big_v, big_c, big_l = [], [], []
            other_v = 0.0
            for v, c, lbl in zip(vals, colors, labels):
                if v / total >= PIE_MERGE_THRESHOLD:
                    big_v.append(v)
                    big_c.append(c)
                    big_l.append(lbl)
                else:
                    other_v += v

            if other_v > 0:
                big_v.append(other_v)
                big_c.append('#CCCCCC')
                big_l.append('Other')

            wedges, texts, autotexts = ax.pie(
                big_v, labels=big_l, colors=big_c,
                autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
                startangle=140, pctdistance=0.65,
                wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
            )
            for t in autotexts:
                t.set_fontsize(9)
            for t in texts:
                t.set_fontsize(9)

            ax.set_title(f'{bm}\n[{mode}]  total={total:.2f}s', fontsize=10)

        fig.suptitle('Phase breakdown', fontsize=12, y=1.02)
        fig.tight_layout()
        _save(fig, out_dir, f'pie_{bm}')


def plot_phase_dominance(data: dict, out_dir: str):
    """Line chart: Z3 exec % of total vs circuit input width for both modes."""
    benchmarks = sorted(data.keys(), key=_extract_inputs)
    inputs = [_extract_inputs(bm) for bm in benchmarks]

    fig, ax = plt.subplots(figsize=(7, 4))

    for mode, style, color in [('sequential', 's--', '#4C72B0'), ('parallel', 'o-', '#55A868')]:
        pcts = []
        valid_inputs = []
        for bm, inp in zip(benchmarks, inputs):
            row = data[bm].get(mode)
            if row and row.get('total_s', 0) > 0:
                pcts.append(100.0 * row.get('exec_s', 0) / row['total_s'])
                valid_inputs.append(inp)
        if pcts:
            ax.plot(valid_inputs, pcts, style, color=color, linewidth=1.5, markersize=5, label=mode)

    ax.set_xlabel('Circuit input width (bits)')
    ax.set_ylabel('Z3 solving (% of total time)')
    ax.set_title('Z3 solving phase dominance vs circuit size')
    ax.set_ylim(0, 105)
    ax.grid(linestyle='--', alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir, 'phase_dominance')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--csv', default=DEFAULT_CSV,
                        help=f'Input CSV (default: {DEFAULT_CSV})')
    parser.add_argument('--out-dir', default=FIG_DIR,
                        help=f'Output directory (default: {FIG_DIR})')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f'ERROR: CSV not found: {args.csv}')
        print('Run: python cluster/merge_results.py  (after cluster jobs finish)')
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    rows = load_csv(args.csv)
    data = pivot(rows)

    print(f'Generating plots for {len(data)} benchmarks')

    plot_stacked_bars(data, args.out_dir)
    plot_speedup(data, args.out_dir,
                 phase='exec_s',
                 title='Z3 solving phase speedup (parallel vs sequential)',
                 filename='exec_speedup')
    plot_speedup(data, args.out_dir,
                 phase='total_s',
                 title='Total labelling time speedup (parallel vs sequential)',
                 filename='total_speedup')
    plot_pies(data, args.out_dir)
    plot_phase_dominance(data, args.out_dir)

    print(f'\nAll figures saved to: {args.out_dir}')


if __name__ == '__main__':
    main()
