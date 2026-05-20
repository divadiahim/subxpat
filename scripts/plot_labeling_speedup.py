"""Generate speedup plots from the labelling benchmark CSV.

Reads output/report/labeling_benchmark.csv (produced by benchmark_labeling.py)
and writes two figures to output/figure/:
  - labeling_speedup_bar.png   — bar chart: speedup per benchmark
  - labeling_times_grouped.png — grouped bar chart: sequential vs parallel times

Usage (from repo root, with venv active):
    python scripts/plot_labeling_speedup.py
    python scripts/plot_labeling_speedup.py --csv my_results.csv --out-dir results/
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

INPUT_CSV = os.path.join("output", "report", "labeling_benchmark.csv")
OUTPUT_DIR = os.path.join("output", "figure")

def _mean_by_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """Average over repeats so we plot one bar per benchmark."""
    return (
        df.groupby("benchmark", sort=False)
        .agg(
            num_inputs=("num_inputs", "first"),
            sequential_s=("sequential_s", "mean"),
            parallel_s=("parallel_s", "mean"),
            speedup=("speedup", "mean"),
            num_cpu_cores=("num_cpu_cores", "first"),
        )
        .reset_index()
        .sort_values("num_inputs")
    )


def plot_speedup_bar(df: pd.DataFrame, out_dir: str, num_cores: int):
    """Bar chart of speedup (sequential / parallel) per benchmark."""
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(df))
    bars = ax.bar(x, df["speedup"], color="#4C72B0", width=0.6, zorder=2)

    # Ideal speedup reference line.
    ax.axhline(num_cores, color="crimson", linestyle="--", linewidth=1.3,
               label=f"Ideal ({num_cores} cores)")
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=1.0, label="No speedup (1×)")

    # Value labels on bars.
    for bar, val in zip(bars, df["speedup"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{val:.2f}×", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(df["benchmark"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Speedup (sequential / parallel)")
    ax.set_title("Gate Labelling Speedup: Parallel vs Sequential")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", which="major", linestyle="--", alpha=0.5, zorder=0)
    ax.legend(fontsize=9)
    fig.tight_layout()

    path = os.path.join(out_dir, "labeling_speedup_bar.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_times_grouped(df: pd.DataFrame, out_dir: str):
    """Grouped bar chart showing absolute wall-clock times per benchmark."""
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(df))
    w = 0.35

    ax.bar(x - w / 2, df["sequential_s"], width=w, label="Sequential", color="#4C72B0", zorder=2)
    ax.bar(x + w / 2, df["parallel_s"],   width=w, label="Parallel",   color="#DD8452", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(df["benchmark"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Gate Labelling Time: Sequential vs Parallel")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", which="major", linestyle="--", alpha=0.5, zorder=0)
    ax.legend(fontsize=9)
    fig.tight_layout()

    path = os.path.join(out_dir, "labeling_times_grouped.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_speedup_line(df: pd.DataFrame, out_dir: str, num_cores: int):
    """Line chart of speedup vs circuit input width — shows scaling trend."""
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(df["num_inputs"], df["speedup"], marker="o", linewidth=2,
            color="#4C72B0", label="Measured speedup")
    ax.axhline(num_cores, color="crimson", linestyle="--", linewidth=1.3,
               label=f"Ideal ({num_cores} cores)")
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=1.0, label="No speedup")

    ax.set_xlabel("Circuit input width (bits)")
    ax.set_ylabel("Speedup (×)")
    ax.set_title("Labelling Speedup vs Circuit Size")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
    ax.grid(linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    fig.tight_layout()

    path = os.path.join(out_dir, "labeling_speedup_line.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=INPUT_CSV,
                        help=f"Path to benchmark CSV (default: {INPUT_CSV})")
    parser.add_argument("--out-dir", default=OUTPUT_DIR,
                        help=f"Output directory for figures (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}", file=sys.stderr)
        print("Run scripts/benchmark_labeling.py first.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    summary = _mean_by_benchmark(df)
    num_cores = int(df["num_cpu_cores"].iloc[0])

    print(f"\nLoaded {len(df)} rows from {args.csv}")
    print(summary[["benchmark", "sequential_s", "parallel_s", "speedup"]].to_string(index=False))

    plot_speedup_bar(summary, args.out_dir, num_cores)
    plot_times_grouped(summary, args.out_dir)
    plot_speedup_line(summary, args.out_dir, num_cores)

    print("\nAll figures saved.")


if __name__ == "__main__":
    main()
