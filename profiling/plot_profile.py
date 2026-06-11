"""Generate thesis-quality plots from the phase profiling CSV.

Reads the merged phase_breakdown CSV and writes figures to an output directory.

Usage (from repo root):
    python profiling/plot_profile.py
    python profiling/plot_profile.py --csv profiling/results/phase_breakdown_habrok.csv
"""

import argparse
import os
import re
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
    d = {}
    for r in rows:
        d.setdefault(r['benchmark'], {})[r['mode']] = r
    return d


def _extract_inputs(name: str) -> int:
    m = re.search(r'_i(\d+)', name)
    return int(m.group(1)) if m else 0


def _extract_family(name: str) -> str:
    m = re.match(r'^([a-z_]+?)_i', name)
    return m.group(1) if m else name


def _short_label(name: str) -> str:
    m = re.search(r'_i(\d+)_o(\d+)', name)
    return f'i{m.group(1)}' if m else name


def _group_by_family(benchmarks):
    families = {}
    for bm in benchmarks:
        fam = _extract_family(bm)
        families.setdefault(fam, []).append(bm)
    for fam in families:
        families[fam].sort(key=_extract_inputs)
    return families


def plot_stacked_bars(data: dict, out_dir: str):
    """Stacked bar split into one subplot per benchmark family."""
    benchmarks = sorted(data.keys(), key=_extract_inputs)
    families = _group_by_family(benchmarks)
    n_families = len(families)
    modes = ['sequential', 'parallel']

    fig, axes = plt.subplots(n_families, 1, figsize=(7, 2.8 * n_families),
                             sharex=False)
    if n_families == 1:
        axes = [axes]

    for ax, (fam, bms) in zip(axes, sorted(families.items())):
        n = len(bms)
        x = np.arange(n)
        width = 0.35

        for m_idx, mode in enumerate(modes):
            offset = (m_idx - 0.5) * width
            bottoms = np.zeros(n)
            for phase in PHASES:
                vals = np.array([data[bm].get(mode, {}).get(phase, 0.0) for bm in bms])
                label = PHASE_LABELS[phase] if (m_idx == 0 and ax is axes[0]) else '_nolegend_'
                ax.bar(x + offset, vals, width, bottom=bottoms,
                       color=PHASE_COLORS[phase], label=label,
                       edgecolor='white', linewidth=0.5)
                bottoms += vals

            for i, bm in enumerate(bms):
                total = data[bm].get(mode, {}).get('total_s', 0.0)
                ax.text(x[i] + offset, bottoms[i] + 0.02 * bottoms.max(),
                        f'{total:.0f}s', ha='center', va='bottom', fontsize=7,
                        color='#333', fontweight='bold')

        tick_positions = []
        tick_labels_list = []
        for i in range(n):
            tick_positions.extend([x[i] - width/2, x[i] + width/2])
            tick_labels_list.extend(['seq', 'par'])
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels_list, fontsize=7)

        for i, bm in enumerate(bms):
            ax.text(x[i], -0.18, _short_label(bm), ha='center', va='top',
                    fontsize=8, fontweight='bold',
                    transform=ax.get_xaxis_transform())

        ax.set_ylabel('Time (s)')
        ax.set_title(fam, fontsize=11, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    axes[0].legend(loc='upper left', fontsize=8, ncol=len(PHASES))
    fig.suptitle('Labelling pipeline phase breakdown (sequential vs parallel)',
                 fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, out_dir, 'stacked_bar')


def plot_speedup(data: dict, out_dir: str, phase: str, title: str, filename: str):
    benchmarks = sorted(data.keys(), key=_extract_inputs)
    families = _group_by_family(benchmarks)
    n_families = len(families)

    fig, axes = plt.subplots(1, n_families, figsize=(2.5 * n_families, 4),
                             sharey=True)
    if n_families == 1:
        axes = [axes]

    all_speedups = []
    for ax, (fam, bms) in zip(axes, sorted(families.items())):
        speedups = []
        for bm in bms:
            seq = data[bm].get('sequential', {}).get(phase, 0.0)
            par = data[bm].get('parallel', {}).get(phase, 0.0)
            speedups.append(seq / par if par > 0 else 0.0)
        all_speedups.extend(speedups)

        labels = [_short_label(bm) for bm in bms]
        x = np.arange(len(bms))
        bars = ax.bar(x, speedups, color='#4C72B0', edgecolor='white', linewidth=0.5)

        for bar, val in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.1f}×', ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(fam, fontsize=10, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.axhline(1.0, color='grey', linestyle='--', linewidth=0.8)

    axes[0].set_ylabel('Speedup (seq / par)')
    for ax in axes:
        ax.set_ylim(0, max(all_speedups) * 1.2 + 0.5)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, filename)


def plot_phase_dominance(data: dict, out_dir: str):
    benchmarks = sorted(data.keys(), key=_extract_inputs)
    families = _group_by_family(benchmarks)

    fig, ax = plt.subplots(figsize=(7, 4))

    fam_colors = plt.cm.tab10(np.linspace(0, 1, len(families)))
    for (fam, bms), color in zip(sorted(families.items()), fam_colors):
        inputs_list = []
        pcts = []
        for bm in bms:
            row = data[bm].get('sequential')
            if row and row.get('total_s', 0) > 0:
                inputs_list.append(_extract_inputs(bm))
                pcts.append(100.0 * row.get('exec_s', 0) / row['total_s'])
        if pcts:
            ax.plot(inputs_list, pcts, 'o-', color=color, linewidth=1.5,
                    markersize=5, label=fam)

    ax.set_xlabel('Circuit input width (bits)')
    ax.set_ylabel('Z3 solving (% of total time)')
    ax.set_title('Z3 solving phase dominance vs circuit size')
    ax.set_ylim(60, 102)
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
    plot_phase_dominance(data, args.out_dir)

    print(f'\nAll figures saved to: {args.out_dir}')


if __name__ == '__main__':
    main()
