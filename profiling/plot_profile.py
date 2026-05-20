"""Generate plots from the phase profiling CSV.

Reads profiling/results/phase_breakdown.csv and writes four figures to
profiling/results/figures/:

  1. stacked_bar.png   — stacked bar: time per phase, seq vs par, per benchmark
  2. exec_speedup.png  — bar chart: exec-phase speedup (seq/par) per benchmark
  3. total_speedup.png — bar chart: total-time speedup per benchmark
  4. phase_pie_<bench>.png — pie chart: phase breakdown for each benchmark (both modes)

Usage (from repo root, with venv active):
    python profiling/plot_profile.py
    python profiling/plot_profile.py --csv profiling/results/phase_breakdown.csv
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PHASES      = ['setup_s', 'gen_s', 'exec_s', 'import_s', 'overhead_s']
PHASE_LABELS = {
    'setup_s':    'Setup',
    'gen_s':      'Script gen',
    'exec_s':     'Z3 subproc',
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

# Phases below this fraction of total are merged into "Other" in pie charts
PIE_MERGE_THRESHOLD = 0.03

FIG_DIR = os.path.join('profiling', 'results', 'figures')


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Plot 1: stacked bar (seq vs par) per benchmark
# ---------------------------------------------------------------------------

def plot_stacked_bars(data: dict, out_dir: str):
    benchmarks = list(data.keys())
    n = len(benchmarks)
    modes = ['sequential', 'parallel']

    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, n * 1.8), 5))

    for m_idx, mode in enumerate(modes):
        offset = (m_idx - 0.5) * width
        bottoms = np.zeros(n)
        for phase in PHASES:
            vals = np.array([data[bm].get(mode, {}).get(phase, 0.0) for bm in benchmarks])
            label = PHASE_LABELS[phase] if m_idx == 0 else '_nolegend_'
            bars = ax.bar(x + offset, vals, width, bottom=bottoms,
                          color=PHASE_COLORS[phase], label=label,
                          edgecolor='white', linewidth=0.5)
            bottoms += vals

        # Annotate total time on top of each bar
        for i, bm in enumerate(benchmarks):
            total = data[bm].get(mode, {}).get('total_s', 0.0)
            ax.text(x[i] + offset, bottoms[i] + 0.05, f'{total:.1f}s',
                    ha='center', va='bottom', fontsize=8,
                    color='#333333', fontweight='bold')

    # Mode labels below x-axis
    tick_labels = []
    for bm in benchmarks:
        tick_labels += [f'{bm}\n(seq)', f'{bm}\n(par)']

    ax.set_xticks(np.concatenate([x - width/2, x + width/2]))
    ax.set_xticklabels(
        [f'seq' for _ in benchmarks] + [f'par' for _ in benchmarks],
        fontsize=8,
    )

    # Group benchmark names above
    for i, bm in enumerate(benchmarks):
        ax.text(x[i], -0.12, bm, ha='center', va='top',
                fontsize=9, fontweight='bold',
                transform=ax.get_xaxis_transform())

    ax.set_ylabel('Time (s)')
    ax.set_title('Labelling pipeline — time per phase\n(sequential vs parallel)')
    ax.legend(loc='upper left', fontsize=8)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()

    path = os.path.join(out_dir, 'stacked_bar.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved: {path}')


# ---------------------------------------------------------------------------
# Plot 2: exec-phase speedup per benchmark
# ---------------------------------------------------------------------------

def plot_speedup(data: dict, out_dir: str, phase: str, title: str, filename: str):
    benchmarks = list(data.keys())
    speedups = []
    for bm in benchmarks:
        seq = data[bm].get('sequential', {}).get(phase, 0.0)
        par = data[bm].get('parallel', {}).get(phase, 0.0)
        speedups.append(seq / par if par > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(max(5, len(benchmarks) * 1.6), 4))
    bars = ax.bar(benchmarks, speedups, color='#4C72B0', edgecolor='white')

    # Annotate value inside each bar
    for bar, val in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f'{val:.2f}x',
                ha='center', va='center', fontsize=10,
                color='white', fontweight='bold')

    ax.axhline(1.0, color='grey', linestyle='--', linewidth=0.8, label='1× (no speedup)')
    ax.set_ylabel('Speedup (seq / par)')
    ax.set_title(title)
    ax.set_ylim(0, max(speedups) * 1.25 + 0.5)
    ax.tick_params(axis='x', rotation=15)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()

    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved: {path}')


# ---------------------------------------------------------------------------
# Plot 3: pie charts per benchmark + mode
# ---------------------------------------------------------------------------

def plot_pies(data: dict, out_dir: str):
    for bm, modes in data.items():
        n_modes = len(modes)
        fig, axes = plt.subplots(1, n_modes, figsize=(5 * n_modes, 5))
        if n_modes == 1:
            axes = [axes]

        for ax, (mode, row) in zip(axes, sorted(modes.items())):
            total = row.get('total_s', 1e-9)
            vals   = [row.get(p, 0.0) for p in PHASES]
            colors = [PHASE_COLORS[p] for p in PHASES]
            labels = [PHASE_LABELS[p] for p in PHASES]

            # Merge slices below threshold into "Other"
            big_v, big_c, big_l = [], [], []
            other_v = 0.0
            for v, c, l in zip(vals, colors, labels):
                if v / total >= PIE_MERGE_THRESHOLD:
                    big_v.append(v)
                    big_c.append(c)
                    big_l.append(l)
                else:
                    other_v += v

            if other_v > 0:
                big_v.append(other_v)
                big_c.append('#CCCCCC')
                big_l.append('Other (<3%)')

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

        fig.suptitle('Phase breakdown — time distribution', fontsize=12)
        fig.tight_layout()
        path = os.path.join(out_dir, f'pie_{bm}.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'  Saved: {path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_CSV = os.path.join('profiling', 'results', 'phase_breakdown.csv')


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
        print('Run `python profiling/profile_labeling.py` first.')
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    rows = load_csv(args.csv)
    data = pivot(rows)

    print(f'Generating plots for: {list(data.keys())}')

    plot_stacked_bars(data, args.out_dir)
    plot_speedup(data, args.out_dir,
                 phase='exec_s',
                 title='Z3 subprocess execution speedup\n(parallel vs sequential)',
                 filename='exec_speedup.png')
    plot_speedup(data, args.out_dir,
                 phase='total_s',
                 title='Total labelling time speedup\n(parallel vs sequential)',
                 filename='total_speedup.png')
    plot_pies(data, args.out_dir)

    print(f'\nAll figures saved to: {args.out_dir}')


if __name__ == '__main__':
    main()
