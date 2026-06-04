"""Generate thesis-quality plots from the labelling benchmark CSV.

Reads the merged CSV (3-mode: sequential, original_parallel, improved_parallel)
and generates publication-quality figures comparing all three modes.

Figures produced:
  1. labeling_times_grouped.pdf — grouped bar: seq vs orig_par vs impr_par times
  2. labeling_speedup_bar.pdf  — grouped bar: orig vs improved speedup per benchmark
  3. labeling_speedup_line.pdf — line chart: speedup scaling vs circuit input width
  4. labeling_improvement.pdf  — bar chart: % improvement of improved over original parallel

Usage (from repo root):
    python scripts/plot_labeling_speedup.py
    python scripts/plot_labeling_speedup.py --csv output/report/labeling_benchmark_habrok.csv
"""

import argparse
import os
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


def _short_name(name: str) -> str:
    return name.replace("_", " ").replace(" i", "\ni")


def plot_times_grouped(df: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.8), 5))

    x = np.arange(len(df))
    w = 0.25

    ax.bar(x - w, df["sequential_s"], width=w,
           label="Sequential", color=COLORS['sequential'], edgecolor='white', linewidth=0.5)
    ax.bar(x, df["original_parallel_s"], width=w,
           label="Original parallel", color=COLORS['original_parallel'], edgecolor='white', linewidth=0.5)
    ax.bar(x + w, df["improved_parallel_s"], width=w,
           label="Improved parallel", color=COLORS['improved_parallel'], edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(df["benchmark"], rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Gate labelling time: three execution modes")
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(ticker.LogLocator(subs='auto'))
    ax.grid(axis="y", which="major", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()

    for fmt in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f"labeling_times_grouped.{fmt}"), dpi=300)
    plt.close(fig)
    print(f"  Saved: labeling_times_grouped.pdf")


def plot_speedup_bar(df: pd.DataFrame, out_dir: str, num_cores: int):
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.8), 4.5))

    x = np.arange(len(df))
    w = 0.32

    bars_orig = ax.bar(x - w/2, df["speedup_original"], width=w,
                        label="Original parallel", color=COLORS['original_parallel'],
                        edgecolor='white', linewidth=0.5)
    bars_impr = ax.bar(x + w/2, df["speedup_improved"], width=w,
                        label="Improved parallel", color=COLORS['improved_parallel'],
                        edgecolor='white', linewidth=0.5)

    ax.axhline(num_cores, color='#CC4444', linestyle='--', linewidth=1.0,
               label=f"Ideal ({num_cores} cores)")
    ax.axhline(1.0, color='grey', linestyle=':', linewidth=0.8)

    for bar, val in zip(bars_orig, df["speedup_original"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.1f}×", ha="center", va="bottom", fontsize=7, color=COLORS['original_parallel'])
    for bar, val in zip(bars_impr, df["speedup_improved"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.1f}×", ha="center", va="bottom", fontsize=7,
                fontweight='bold', color=COLORS['improved_parallel'])

    ax.set_xticks(x)
    ax.set_xticklabels(df["benchmark"], rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Speedup (sequential / parallel)")
    ax.set_title("Parallel labelling speedup: original vs improved")
    ax.set_ylim(0, max(df["speedup_improved"].max(), df["speedup_original"].max(), num_cores) * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()

    for fmt in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f"labeling_speedup_bar.{fmt}"), dpi=300)
    plt.close(fig)
    print(f"  Saved: labeling_speedup_bar.pdf")


def plot_speedup_line(df: pd.DataFrame, out_dir: str, num_cores: int):
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(df["num_inputs"], df["speedup_original"], 's--',
            color=COLORS['original_parallel'], linewidth=1.5, markersize=5,
            label="Original parallel")
    ax.plot(df["num_inputs"], df["speedup_improved"], 'o-',
            color=COLORS['improved_parallel'], linewidth=2, markersize=6,
            label="Improved parallel")
    ax.axhline(num_cores, color='#CC4444', linestyle='--', linewidth=1.0,
               label=f"Ideal ({num_cores} cores)")
    ax.axhline(1.0, color='grey', linestyle=':', linewidth=0.8)

    ax.set_xlabel("Circuit input width (bits)")
    ax.set_ylabel("Speedup (×)")
    ax.set_title("Labelling speedup vs circuit size")
    ax.set_ylim(0, max(df["speedup_improved"].max(), num_cores) * 1.15)
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()

    for fmt in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f"labeling_speedup_line.{fmt}"), dpi=300)
    plt.close(fig)
    print(f"  Saved: labeling_speedup_line.pdf")


def plot_improvement(df: pd.DataFrame, out_dir: str):
    pct_improvement = (
        (df["original_parallel_s"] - df["improved_parallel_s"])
        / df["original_parallel_s"] * 100
    )

    fig, ax = plt.subplots(figsize=(max(7, len(df) * 0.7), 4))

    x = np.arange(len(df))
    colors = [COLORS['improved_parallel'] if v > 0 else '#CC4444' for v in pct_improvement]
    bars = ax.bar(x, pct_improvement, color=colors, edgecolor='white', linewidth=0.5)

    for bar, val in zip(bars, pct_improvement):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (1 if val >= 0 else -3),
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight='bold')

    ax.axhline(0, color='grey', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["benchmark"], rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Time reduction (%)")
    ax.set_title("Improved parallel vs original parallel: time reduction")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    for fmt in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f"labeling_improvement.{fmt}"), dpi=300)
    plt.close(fig)
    print(f"  Saved: labeling_improvement.pdf")


def plot_speedup_by_family(df: pd.DataFrame, out_dir: str, num_cores: int):
    """One line plot per benchmark family showing speedup scaling."""
    df = df.copy()
    df["family"] = df["benchmark"].str.extract(r"^([a-z_]+?)_i")[0]

    families = df["family"].unique()
    if len(families) < 2:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    family_colors = plt.cm.tab10(np.linspace(0, 1, len(families)))
    for fam, color in zip(sorted(families), family_colors):
        sub = df[df["family"] == fam].sort_values("num_inputs")
        ax.plot(sub["num_inputs"], sub["speedup_improved"], 'o-',
                color=color, linewidth=1.5, markersize=5, label=fam)

    ax.axhline(num_cores, color='#CC4444', linestyle='--', linewidth=1.0,
               label=f"Ideal ({num_cores} cores)")
    ax.set_xlabel("Circuit input width (bits)")
    ax.set_ylabel("Improved parallel speedup (×)")
    ax.set_title("Speedup scaling by benchmark family")
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    for fmt in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f"labeling_speedup_families.{fmt}"), dpi=300)
    plt.close(fig)
    print(f"  Saved: labeling_speedup_families.pdf")


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
    print(summary[["benchmark", "sequential_s", "original_parallel_s",
                    "improved_parallel_s", "speedup_original", "speedup_improved"]]
          .to_string(index=False))
    print()

    plot_times_grouped(summary, args.out_dir)
    plot_speedup_bar(summary, args.out_dir, num_cores)
    plot_speedup_line(summary, args.out_dir, num_cores)
    plot_improvement(summary, args.out_dir)
    plot_speedup_by_family(summary, args.out_dir, num_cores)

    print(f"\nAll figures saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
