"""Generate thesis-quality plots from the pre-filter benchmark CSV.

Reads the merged CSV (modes: full, partial, prefilter) and generates
publication-quality figures split by benchmark family, showing actual
SMT-call counts and wall-clock seconds (not ratios).

Usage (from repo root):
    python scripts/plot_prefilter.py
    python scripts/plot_prefilter.py --csv output/report/prefilter_benchmark_habrok.csv
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

INPUT_CSV = os.path.join("output", "report", "prefilter_benchmark_habrok.csv")
OUTPUT_DIR = os.path.join("output", "figure")

COLORS = {
    'full':      '#4C72B0',
    'partial':   '#DD8452',
    'prefilter': '#55A868',
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
    """Average repeats per (benchmark, et)."""
    out = (
        df.groupby(["benchmark", "et", "et_frac"], sort=False)
        .agg(
            num_inputs=("num_inputs", "first"),
            num_outputs=("num_outputs", "first"),
            num_cpu_cores=("num_cpu_cores", "first"),
            full_s=("full_s", "mean"),
            full_calls=("full_calls", "mean"),
            partial_s=("partial_s", "mean"),
            partial_calls=("partial_calls", "mean"),
            prefilter_s=("prefilter_s", "mean"),
            prefilter_calls=("prefilter_calls", "mean"),
        )
        .reset_index()
    )
    out['family'] = out['benchmark'].apply(_family)
    return out.sort_values(["family", "num_inputs"])


def _split_families(df: pd.DataFrame):
    return {fam: grp.sort_values('num_inputs') for fam, grp in df.groupby('family')}


def plot_calls_bar(df: pd.DataFrame, out_dir: str, frac: str):
    """Actual SMT-call counts per mode, per family, at one ET fraction."""
    fdf_all = df[df["et_frac"] == frac]
    if fdf_all.empty:
        print(f"  (no rows for et_frac={frac}, skipping calls bar)")
        return
    families = _split_families(fdf_all)
    n_fam = len(families)

    fig, axes = plt.subplots(1, n_fam, figsize=(2.5 * n_fam, 4.5), sharey=False)
    if n_fam == 1:
        axes = [axes]

    for ax, (fam, fdf) in zip(axes, sorted(families.items())):
        n = len(fdf)
        x = np.arange(n)
        w = 0.25

        series = [
            ("full_calls", 'full', "Full"),
            ("partial_calls", 'partial', "Partial"),
            ("prefilter_calls", 'prefilter', "Pre-filter"),
        ]
        top = fdf[[c for c, _, _ in series]].values.max()
        for off, (col, key, label) in zip((-w, 0, w), series):
            bars = ax.bar(x + off, fdf[col].values, width=w,
                          label=label if ax is axes[0] else "_nolegend_",
                          color=COLORS[key], edgecolor='white', linewidth=0.5)
            for bar, val in zip(bars, fdf[col]):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + top * 0.01,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=5.5,
                        color=COLORS[key])

        labels = [_short(b) for b in fdf["benchmark"]]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(fam, fontsize=10, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.set_ylim(0, top * 1.15)

    axes[0].set_ylabel("SMT optimisation calls")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"SMT calls per labelling mode (ET = {frac} of max output)", fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, f"prefilter_calls_{_frac_slug(frac)}")


def plot_times_bar(df: pd.DataFrame, out_dir: str, frac: str):
    """Actual wall-clock seconds per mode, per family, at one ET fraction."""
    fdf_all = df[df["et_frac"] == frac]
    if fdf_all.empty:
        print(f"  (no rows for et_frac={frac}, skipping times bar)")
        return
    families = _split_families(fdf_all)
    n_fam = len(families)

    fig, axes = plt.subplots(1, n_fam, figsize=(2.5 * n_fam, 4.5), sharey=False)
    if n_fam == 1:
        axes = [axes]

    for ax, (fam, fdf) in zip(axes, sorted(families.items())):
        n = len(fdf)
        x = np.arange(n)
        w = 0.25

        series = [
            ("full_s", 'full', "Full"),
            ("partial_s", 'partial', "Partial"),
            ("prefilter_s", 'prefilter', "Pre-filter"),
        ]
        top = fdf[[c for c, _, _ in series]].values.max()
        for off, (col, key, label) in zip((-w, 0, w), series):
            bars = ax.bar(x + off, fdf[col].values, width=w,
                          label=label if ax is axes[0] else "_nolegend_",
                          color=COLORS[key], edgecolor='white', linewidth=0.5)
            for bar, val in zip(bars, fdf[col]):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + top * 0.01,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=5.5,
                        color=COLORS[key])

        labels = [_short(b) for b in fdf["benchmark"]]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(fam, fontsize=10, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.set_ylim(0, top * 1.15)

    axes[0].set_ylabel("Wall-clock time (s)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Labelling time per mode (ET = {frac} of max output)", fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, f"prefilter_times_{_frac_slug(frac)}")


def plot_calls_vs_et(df: pd.DataFrame, out_dir: str):
    """SMT calls vs ET fraction, per family subplot. Pre-filter calls as solid
    lines, full labelling as dashed horizontal reference per benchmark."""
    families = _split_families(df)
    n_fam = len(families)

    fig, axes = plt.subplots(1, n_fam, figsize=(2.7 * n_fam, 4), sharex=True)
    if n_fam == 1:
        axes = [axes]

    fracs = [f for f in FRAC_ORDER if f in set(df["et_frac"])]
    xs = np.arange(len(fracs))

    for ax, (fam, fdf) in zip(axes, sorted(families.items())):
        benches = sorted(fdf["benchmark"].unique(),
                         key=lambda b: fdf[fdf.benchmark == b]["num_inputs"].iloc[0])
        colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(benches)))
        for bench, color in zip(benches, colors):
            bdf = fdf[fdf.benchmark == bench].set_index("et_frac")
            ys = [bdf.loc[f, "prefilter_calls"] if f in bdf.index else np.nan
                  for f in fracs]
            ax.plot(xs, ys, 'o-', color=color, linewidth=1.5, markersize=4,
                    label=_short(bench))
            full = bdf["full_calls"].iloc[0]
            ax.axhline(full, color=color, linestyle='--', linewidth=0.8, alpha=0.5)

        ax.set_xticks(xs)
        ax.set_xticklabels(fracs)
        ax.set_title(fam, fontsize=10, fontweight='bold')
        ax.grid(linestyle='--', alpha=0.4)
        ax.legend(fontsize=6.5)

    axes[0].set_ylabel("SMT optimisation calls")
    fig.supxlabel("ET (fraction of maximum output value)", fontsize=11)
    fig.suptitle("Pre-filter SMT calls vs ET (dashed = full labelling)", fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, "prefilter_calls_vs_et")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=INPUT_CSV,
                        help=f"Path to benchmark CSV (default: {INPUT_CSV})")
    parser.add_argument("--out-dir", default=OUTPUT_DIR,
                        help=f"Output directory for figures (default: {OUTPUT_DIR})")
    parser.add_argument("--fracs", nargs="+", default=["1/16", "1/2"],
                        help="ET fractions for the per-mode bar charts")
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
    plot_calls_vs_et(df, args.out_dir)

    print(f"\nAll figures saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
