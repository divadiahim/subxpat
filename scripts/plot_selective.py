"""Generate thesis-quality plots from the selective-relabelling benchmark CSV.

Reads the merged CSV (modes: full, selective, combined = selective + pre-filter)
and generates publication-quality figures split by benchmark family, showing
actual SMT-call counts and wall-clock seconds (not ratios). Counts and times
are averaged over trials (random rewrite sites) and repeats.

Usage (from repo root):
    python scripts/plot_selective.py
    python scripts/plot_selective.py --csv output/report/selective_benchmark_habrok.csv
"""

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

INPUT_CSV = os.path.join("output", "report", "selective_benchmark_habrok.csv")
OUTPUT_DIR = os.path.join("output", "figure")

COLORS = {
    'full':      '#4C72B0',
    'selective': '#DD8452',
    'prefilter': '#8172B3',
    'combined':  '#55A868',
}

FRAC_ORDER = ["1/16", "1/8", "1/4", "1/2"]


def _save(fig, out_dir, name):
    for fmt in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'{name}.{fmt}'), dpi=300)
    plt.close(fig)
    print(f"  Saved: {name}.pdf")


def _family(name: str) -> str:
    m = re.match(r'^([a-z_]+?)_i', name)
    return m.group(1) if m else name


def _short(name: str) -> str:
    m = re.search(r'_i(\d+)', name)
    return f'i{m.group(1)}' if m else name


def _frac_slug(frac: str) -> str:
    return frac.replace('/', '-')


def _mean_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Average trials and repeats per (benchmark, et)."""
    out = (
        df.groupby(["benchmark", "et", "et_frac"], sort=False)
        .agg(
            num_inputs=("num_inputs", "first"),
            num_outputs=("num_outputs", "first"),
            num_cpu_cores=("num_cpu_cores", "first"),
            n_modified=("n_modified", "first"),
            full_s=("full_s", "mean"),
            full_calls=("full_calls", "mean"),
            selective_s=("selective_s", "mean"),
            selective_calls=("selective_calls", "mean"),
            prefilter_s=("prefilter_s", "mean"),
            prefilter_calls=("prefilter_calls", "mean"),
            combined_s=("combined_s", "mean"),
            combined_calls=("combined_calls", "mean"),
        )
        .reset_index()
    )
    out['family'] = out['benchmark'].apply(_family)
    return out.sort_values(["family", "num_inputs"])


def _split_families(df: pd.DataFrame):
    return {fam: grp.sort_values('num_inputs') for fam, grp in df.groupby('family')}


def _grouped_bar(df: pd.DataFrame, out_dir: str, frac: str, columns, ylabel,
                 title, name, fmt):
    """Shared per-family grouped-bar plot for three modes at one ET fraction."""
    fdf_all = df[df["et_frac"] == frac]
    if fdf_all.empty:
        print(f"  (no rows for et_frac={frac}, skipping {name})")
        return
    families = _split_families(fdf_all)
    n_fam = len(families)

    fig, axes = plt.subplots(1, n_fam, figsize=(2.5 * n_fam, 4.5), sharey=False)
    if n_fam == 1:
        axes = [axes]

    for ax, (fam, fdf) in zip(axes, sorted(families.items())):
        n = len(fdf)
        x = np.arange(n)
        k = len(columns)
        w = 0.8 / k
        offs = [(i - (k - 1) / 2) * w for i in range(k)]

        top = fdf[[c for c, _, _ in columns]].values.max()
        for off, (col, key, label) in zip(offs, columns):
            bars = ax.bar(x + off, fdf[col].values, width=w,
                          label=label if ax is axes[0] else "_nolegend_",
                          color=COLORS[key], edgecolor='white', linewidth=0.5)
            for bar, val in zip(bars, fdf[col]):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + top * 0.01,
                        fmt.format(val), ha="center", va="bottom", fontsize=5.5,
                        color=COLORS[key])

        labels = [_short(b) for b in fdf["benchmark"]]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(fam, fontsize=10, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.set_ylim(0, top * 1.15)

    axes[0].set_ylabel(ylabel)
    axes[0].legend(fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, name)


def plot_calls_bar(df: pd.DataFrame, out_dir: str, frac: str):
    _grouped_bar(
        df, out_dir, frac,
        columns=[
            ("full_calls", 'full', "Full relabel"),
            ("selective_calls", 'selective', "Selective"),
            ("prefilter_calls", 'prefilter', "Pre-filter"),
            ("combined_calls", 'combined', "Selective + pre-filter"),
        ],
        ylabel="SMT optimisation calls",
        title=f"SMT calls per simulated iteration (ET = {frac} of max output)",
        name=f"selective_calls_{_frac_slug(frac)}",
        fmt="{:.0f}",
    )


def plot_times_bar(df: pd.DataFrame, out_dir: str, frac: str):
    _grouped_bar(
        df, out_dir, frac,
        columns=[
            ("full_s", 'full', "Full relabel"),
            ("selective_s", 'selective', "Selective"),
            ("prefilter_s", 'prefilter', "Pre-filter"),
            ("combined_s", 'combined', "Selective + pre-filter"),
        ],
        ylabel="Wall-clock time (s)",
        title=f"Relabelling time per simulated iteration (ET = {frac} of max output)",
        name=f"selective_times_{_frac_slug(frac)}",
        fmt="{:.1f}",
    )


def plot_calls_vs_size(df: pd.DataFrame, out_dir: str, frac: str):
    """SMT calls vs circuit input width per family, three modes, at one ET fraction."""
    fdf_all = df[df["et_frac"] == frac]
    if fdf_all.empty:
        print(f"  (no rows for et_frac={frac}, skipping calls vs size)")
        return
    families = _split_families(fdf_all)
    n_fam = len(families)

    fig, axes = plt.subplots(1, n_fam, figsize=(2.7 * n_fam, 4), sharex=False)
    if n_fam == 1:
        axes = [axes]

    for ax, (fam, fdf) in zip(axes, sorted(families.items())):
        ax.plot(fdf["num_inputs"], fdf["full_calls"], 'o--',
                color=COLORS['full'], linewidth=1.3, markersize=4,
                label="Full relabel" if ax is axes[0] else "_nolegend_")
        ax.plot(fdf["num_inputs"], fdf["selective_calls"], 'o-',
                color=COLORS['selective'], linewidth=1.6, markersize=4,
                label="Selective" if ax is axes[0] else "_nolegend_")
        ax.plot(fdf["num_inputs"], fdf["prefilter_calls"], 'o-',
                color=COLORS['prefilter'], linewidth=1.6, markersize=4,
                label="Pre-filter" if ax is axes[0] else "_nolegend_")
        ax.plot(fdf["num_inputs"], fdf["combined_calls"], 'o-',
                color=COLORS['combined'], linewidth=1.6, markersize=4,
                label="Selective + pre-filter" if ax is axes[0] else "_nolegend_")

        ax.set_title(fam, fontsize=10, fontweight='bold')
        ax.set_xlabel("Input width (bits)")
        ax.grid(linestyle='--', alpha=0.4)

    axes[0].set_ylabel("SMT optimisation calls")
    axes[0].legend(fontsize=7.5)
    fig.suptitle(f"SMT calls vs circuit size (mean over rewrite sites, ET = {frac})",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, f"selective_calls_vs_size_{_frac_slug(frac)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=INPUT_CSV,
                        help=f"Path to benchmark CSV (default: {INPUT_CSV})")
    parser.add_argument("--out-dir", default=OUTPUT_DIR,
                        help=f"Output directory for figures (default: {OUTPUT_DIR})")
    parser.add_argument("--fracs", nargs="+", default=["1/16", "1/2"],
                        help="ET fractions for the per-mode charts")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}", file=sys.stderr)
        print("Run: python cluster/merge_results.py  (after cluster jobs finish)",
              file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    df = _mean_rows(pd.read_csv(args.csv))
    print(f"\nLoaded {len(df)} (benchmark, et) rows from {args.csv}\n")

    for frac in args.fracs:
        plot_calls_bar(df, args.out_dir, frac)
        plot_times_bar(df, args.out_dir, frac)
        plot_calls_vs_size(df, args.out_dir, frac)

    print(f"\nAll figures saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
