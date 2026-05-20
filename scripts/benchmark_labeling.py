"""Benchmark sequential vs parallel gate labelling across adder circuits.

Runs time_labeling() in both modes on a configurable set of benchmarks,
records wall-clock time and speedup, and saves results to:
    output/report/labeling_benchmark.csv

Usage (from repo root, with venv active):
    python scripts/benchmark_labeling.py
    python scripts/benchmark_labeling.py --benchmarks adder_i4_o3 adder_i8_o5
    python scripts/benchmark_labeling.py --partial-cutoff 8 --repeats 3
"""

import argparse
import csv
import multiprocessing
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sxpat.labeling import time_labeling

DEFAULT_BENCHMARKS = [
    "adder_i4_o3",
    "adder_i6_o4",
    "adder_i8_o5",
    "adder_i10_o6",
    "adder_i12_o7",
    "adder_i16_o9",
]
DEFAULT_PARTIAL_CUTOFF = 4
DEFAULT_REPEATS = 1
OUTPUT_CSV = os.path.join("output", "report", "labeling_benchmark.csv")
CSV_HEADER = [
    "benchmark",
    "num_inputs",
    "num_outputs",
    "num_cpu_cores",
    "partial_cutoff",
    "repeat",
    "sequential_s",
    "parallel_s",
    "speedup",
]


def _parse_circuit_dims(name: str):
    import re
    m = re.search(r"_i(\d+)_o(\d+)", name)
    return (int(m.group(1)), int(m.group(2))) if m else (-1, -1)


def run_benchmark(benchmark: str, partial_cutoff: int, repeat: int) -> dict:
    num_inputs, num_outputs = _parse_circuit_dims(benchmark)
    num_cores = multiprocessing.cpu_count()

    print(f"  [{repeat}] sequential ... ", end="", flush=True)
    _, seq_time = time_labeling(
        benchmark,
        min_labeling=False,
        partial_labeling=False,
        partial_cutoff=partial_cutoff,
        parallel=False,
    )
    print(f"{seq_time:.2f}s")

    print(f"  [{repeat}] parallel   ... ", end="", flush=True)
    _, par_time = time_labeling(
        benchmark,
        min_labeling=False,
        partial_labeling=False,
        partial_cutoff=partial_cutoff,
        parallel=True,
    )
    print(f"{par_time:.2f}s")

    speedup = seq_time / par_time if par_time > 0 else float("inf")
    print(f"  [{repeat}] speedup    = {speedup:.2f}x")

    return {
        "benchmark": benchmark,
        "num_inputs": num_inputs,
        "num_outputs": num_outputs,
        "num_cpu_cores": num_cores,
        "partial_cutoff": partial_cutoff,
        "repeat": repeat,
        "sequential_s": round(seq_time, 4),
        "parallel_s": round(par_time, 4),
        "speedup": round(speedup, 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS,
        metavar="BENCHMARK",
    )
    parser.add_argument("--partial-cutoff", type=int, default=DEFAULT_PARTIAL_CUTOFF)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    all_rows = []
    for benchmark in args.benchmarks:
        print(f"\n=== {benchmark} ===")
        for r in range(1, args.repeats + 1):
            row = run_benchmark(benchmark, args.partial_cutoff, r)
            all_rows.append(row)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nResults saved to {OUTPUT_CSV}")

    print(f"\n{'Benchmark':<20} {'Seq (s)':>10} {'Par (s)':>10} {'Speedup':>10}")
    print("-" * 55)
    for row in all_rows:
        if row["repeat"] == 1:
            print(
                f"{row['benchmark']:<20} "
                f"{row['sequential_s']:>10.2f} "
                f"{row['parallel_s']:>10.2f} "
                f"{row['speedup']:>10.2f}x"
            )


if __name__ == "__main__":
    main()
