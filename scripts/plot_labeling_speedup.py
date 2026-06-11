"""Generate thesis-quality plots from the labelling benchmark CSV.

Reads the merged CSV (3-mode: sequential, original_parallel, improved_parallel)
and generates publication-quality figures split by benchmark family for readability.

Usage (from repo root):
    python scripts/plot_labeling_speedup.py
    python scripts/plot_labeling_speedup.py --csv output/report/labeling_benchmark_habrok.csv
"""

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

INPUT_CSV = os.path.join("output", "report", "labeling_benchmark_habrok.csv")
OUTPUT_DIR = os.path.join("output", "figure")

COLORS = {
    'sequential':        '#4C72B0',
    'original_parallel': '#DD8452',
    'improved_parallel': '#55A868',
}


def _save(fig, out_dir, name):
    for fmt in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'{name}.{fmt}'), dpi=300)
    plt.close(fig)
    print(f"  Saved: {name}.pdf")


def _mean_by_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("benchmark", sort=False)
        .agg(
            num_inputs=("num_inputs", "first"),
            num_outputs=("num_outputs", "first"),
            num_cpu_cores=("num_cpu_cores", "first"),
            sequential_s=("sequential_s", "mean"),
            original_parallel_s=("original_parallel_s", "mean"),
            improved_parallel_s=("improved_parallel_s", "mean"),
            speedup_original=("speedup_original", "mean"),
            speedup_improved=("speedup_improved", "mean"),
        )
        .reset_index()
        .sort_values("num_inputs")
    )


def _family(name: str) -> str:
    m = re.match(r'^([a-z_]+?)_i', name)
    return m.group(1) if m else name


def _short(name: str) -> str:
    m = re.search(r'_i(\d+)', name)
    return f'i{m.group(1)}' if m else name


def _split_families(df: pd.DataFrame):
    df = df.copy()
    df['family'] = df['benchmark'].apply(_family)
    return {fam: grp.sort_values('num_inputs') for fam, grp in df.groupby('family')}


def plot_times_grouped(df: pd.DataFrame, out_dir: str):
    """Grouped bar per family in subplots — seq vs orig_par vs impr_par."""
    families = _split_families(df)
    n_fam = len(families)

    fig, axes = plt.subplots(1, n_fam, figsize=(2.5 * n_fam, 4.5), sharey=True)
    if n_fam == 1:
        axes = [axes]

    for ax, (fam, fdf) in zip(axes, sorted(families.items())):
        n = len(fdf)
        x = np.arange(n)
        w = 0.25

        ax.bar(x - w, fdf["sequential_s"].values, width=w,
               label="Sequential" if ax is axes[0] else "_nolegend_",
               color=COLORS['sequential'], edgecolor='white', linewidth=0.5)
        ax.bar(x, fdf["original_parallel_s"].values, width=w,
               label="Original par." if ax is axes[0] else "_nolegend_",
               color=COLORS['original_parallel'], edgecolor='white', linewidth=0.5)
        ax.bar(x + w, fdf["improved_parallel_s"].values, width=w,
               label="Improved par." if ax is axes[0] else "_nolegend_",
               color=COLORS['improved_parallel'], edgecolor='white', linewidth=0.5)

        labels = [_short(b) for b in fdf["benchmark"]]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(fam, fontsize=10, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    axes[0].set_ylabel("Wall-clock time (s)")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    fig.suptitle("Gate labelling time: three execution modes", fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, "labeling_times_grouped")


def plot_times_bar(df: pd.DataFrame, out_dir: str):
    """Actual wall-clock seconds per family — original vs improved parallel (linear)."""
    families = _split_families(df)
    n_fam = len(families)

    fig, axes = plt.subplots(1, n_fam, figsize=(2.5 * n_fam, 4.5), sharey=False)
    if n_fam == 1:
        axes = [axes]

    for ax, (fam, fdf) in zip(axes, sorted(families.items())):
        n = len(fdf)
        x = np.arange(n)
        w = 0.32

        bars_orig = ax.bar(x - w/2, fdf["original_parallel_s"].values, width=w,
                           label="Original par." if ax is axes[0] else "_nolegend_",
                           color=COLORS['original_parallel'], edgecolor='white', linewidth=0.5)
        bars_impr = ax.bar(x + w/2, fdf["improved_parallel_s"].values, width=w,
                           label="Improved par." if ax is axes[0] else "_nolegend_",
                           color=COLORS['improved_parallel'], edgecolor='white', linewidth=0.5)

        top = fdf[["original_parallel_s", "improved_parallel_s"]].values.max()
        for bar, val in zip(bars_orig, fdf["original_parallel_s"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + top * 0.01,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=6,
                    color=COLORS['original_parallel'])
        for bar, val in zip(bars_impr, fdf["improved_parallel_s"]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + top * 0.01,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=6,
                    fontweight='bold', color=COLORS['improved_parallel'])

        labels = [_short(b) for b in fdf["benchmark"]]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(fam, fontsize=10, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.set_ylim(0, top * 1.15)

    axes[0].set_ylabel("Wall-clock time (s)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Parallel labelling time: original vs improved", fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, "labeling_times_bar")


def plot_times_families(df: pd.DataFrame, out_dir: str):
    """Wall-clock time vs circuit input width, per family: sequential vs improved (log)."""
    families = _split_families(df)

    fig, ax = plt.subplots(figsize=(7, 4.2))

    fam_colors = plt.cm.tab10(np.linspace(0, 1, len(families)))
    for (fam, fdf), color in zip(sorted(families.items()), fam_colors):
        ax.plot(fdf["num_inputs"].values, fdf["sequential_s"].values, 'o--',
                color=color, linewidth=1.2, markersize=4, alpha=0.55,
                label=f"{fam} (seq)")
        ax.plot(fdf["num_inputs"].values, fdf["improved_parallel_s"].values, 'o-',
                color=color, linewidth=1.8, markersize=5,
                label=f"{fam} (improved)")

    ax.set_xlabel("Circuit input width (bits)")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_yscale("log")
    ax.set_title("Labelling time vs circuit size (sequential vs improved parallel)")
    ax.grid(linestyle="--", alpha=0.4, which="both")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    _save(fig, out_dir, "labeling_times_families")


def plot_time_saved(df: pd.DataFrame, out_dir: str):
    """Absolute seconds saved by improved vs original parallel, split by family."""
    df = df.copy()
    df['saved_s'] = df["original_parallel_s"] - df["improved_parallel_s"]
    families = _split_families(df)
    n_fam = len(families)

    fig, axes = plt.subplots(1, n_fam, figsize=(2.5 * n_fam, 4), sharey=True)
    if n_fam == 1:
        axes = [axes]

    for ax, (fam, fdf) in zip(axes, sorted(families.items())):
        n = len(fdf)
        x = np.arange(n)
        saved = fdf['saved_s'].values
        colors = [COLORS['improved_parallel'] if v >= 0 else '#CC4444' for v in saved]
        bars = ax.bar(x, saved, color=colors, edgecolor='white', linewidth=0.5)

        for bar, val in zip(bars, saved):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (0.05 if val >= 0 else -0.05),
                    f"{val:+.1f}", ha="center",
                    va="bottom" if val >= 0 else "top", fontsize=6.5,
                    fontweight='bold')

        labels = [_short(b) for b in fdf["benchmark"]]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(fam, fontsize=10, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.axhline(0, color='grey', linewidth=0.8)

    axes[0].set_ylabel("Time saved (s)")
    fig.suptitle("Improved vs original parallel: wall-clock time saved", fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, "labeling_time_saved")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=INPUT_CSV,
                        help=f"Path to benchmark CSV (default: {INPUT_CSV})")
    parser.add_argument("--out-dir", default=OUTPUT_DIR,
                        help=f"Output directory for figures (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}", file=sys.stderr)
        print("Run: python cluster/merge_results.py  (after cluster jobs finish)", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    summary = _mean_by_benchmark(df)
    num_cores = int(df["num_cpu_cores"].iloc[0])

    print(f"\nLoaded {len(df)} rows, {len(summary)} benchmarks from {args.csv}")
    print(f"CPU cores: {num_cores}\n")

    plot_times_grouped(summary, args.out_dir)
    plot_times_bar(summary, args.out_dir)
    plot_times_families(summary, args.out_dir)
    plot_time_saved(summary, args.out_dir)

    print(f"\nAll figures saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
