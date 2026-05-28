"""Merge per-benchmark CSVs from array jobs into single result files.

Usage (from repo root):
    python cluster/merge_results.py
    python cluster/merge_results.py --bench-dir output/report/habrok --prof-dir profiling/results/habrok
"""

import argparse
import csv
import os
import sys

BENCH_DIR = os.path.join("output", "report", "habrok")
PROF_DIR = os.path.join("profiling", "results", "habrok")


def merge_benchmark_csvs(src_dir: str, out_path: str):
    all_rows = []
    header = None
    for fname in sorted(os.listdir(src_dir)):
        if not fname.endswith(".csv"):
            continue
        path = os.path.join(src_dir, fname)
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if header is None:
                header = reader.fieldnames
            for row in reader:
                all_rows.append(row)

    if not all_rows:
        print(f"  No CSV files found in {src_dir}")
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  Merged {len(all_rows)} rows -> {out_path}")


def merge_profile_csvs(src_dir: str, out_path: str):
    all_rows = []
    header = None
    for bm_dir in sorted(os.listdir(src_dir)):
        csv_path = os.path.join(src_dir, bm_dir, "phase_breakdown.csv")
        if not os.path.isfile(csv_path):
            continue
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            if header is None:
                header = reader.fieldnames
            for row in reader:
                all_rows.append(row)

    if not all_rows:
        print(f"  No phase_breakdown.csv files found in {src_dir}/*/")
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  Merged {len(all_rows)} rows -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-dir", default=BENCH_DIR)
    parser.add_argument("--prof-dir", default=PROF_DIR)
    args = parser.parse_args()

    print("Merging benchmark results...")
    if os.path.isdir(args.bench_dir):
        merge_benchmark_csvs(
            args.bench_dir,
            os.path.join("output", "report", "labeling_benchmark_habrok.csv"),
        )
    else:
        print(f"  Skipping (not found): {args.bench_dir}")

    print("Merging profiling results...")
    if os.path.isdir(args.prof_dir):
        merge_profile_csvs(
            args.prof_dir,
            os.path.join("profiling", "results", "phase_breakdown_habrok.csv"),
        )
    else:
        print(f"  Skipping (not found): {args.prof_dir}")


if __name__ == "__main__":
    main()
