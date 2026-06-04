"""Benchmark sequential vs parallel gate labelling across circuits.

Runs time_labeling() in three modes on a configurable set of benchmarks:
  1. Sequential (one subprocess at a time)
  2. Original parallel (base Z3Log FIFO pool — the pre-improvement baseline)
  3. Improved parallel (ThreadPoolExecutor + as_completed)

Records wall-clock time and speedup, and saves results to a CSV.

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
    "original_parallel_s",
    "improved_parallel_s",
    "speedup_original",
    "speedup_improved",
]


def _parse_circuit_dims(name: str):
    import re
    m = re.search(r"_i(\d+)_o(\d+)", name)
    return (int(m.group(1)), int(m.group(2))) if m else (-1, -1)


def _patch_original_parallel():
    """Temporarily replace run_z3pyscript_labeling with the original Z3Log
    FIFO-pool implementation, so we can benchmark the pre-improvement baseline.
    Returns a restore function."""
    from Z3Log_patched.z3solver import Z3solver
    saved = Z3solver.run_z3pyscript_labeling

    from subprocess import PIPE, Popen
    _python = sys.executable

    def _original_run_z3pyscript_labeling(self):
        active_procs = []
        num_workers = multiprocessing.cpu_count()
        for pyscript in self.pyscript_files_for_labeling:
            proc = Popen([_python, pyscript], stderr=PIPE, stdout=PIPE)
            active_procs.append(proc)
            if len(active_procs) >= num_workers:
                finished_proc = active_procs.pop(0)
                finished_proc.communicate()
        for proc in active_procs:
            proc.communicate()

    Z3solver.run_z3pyscript_labeling = _original_run_z3pyscript_labeling

    def restore():
        Z3solver.run_z3pyscript_labeling = saved

    return restore


def run_benchmark(benchmark: str, partial_cutoff: int, repeat: int) -> dict:
    num_inputs, num_outputs = _parse_circuit_dims(benchmark)
    num_cores = multiprocessing.cpu_count()

    # 1) Sequential
    print(f"  [{repeat}] sequential        ... ", end="", flush=True)
    _, seq_time = time_labeling(
        benchmark,
        min_labeling=False,
        partial_labeling=False,
        partial_cutoff=partial_cutoff,
        parallel=False,
    )
    print(f"{seq_time:.2f}s")

    # 2) Original parallel (FIFO pool baseline)
    print(f"  [{repeat}] original parallel ... ", end="", flush=True)
    restore = _patch_original_parallel()
    try:
        _, orig_par_time = time_labeling(
            benchmark,
            min_labeling=False,
            partial_labeling=False,
            partial_cutoff=partial_cutoff,
            parallel=True,
        )
    finally:
        restore()
    print(f"{orig_par_time:.2f}s")

    # 3) Improved parallel (ThreadPoolExecutor)
    print(f"  [{repeat}] improved parallel ... ", end="", flush=True)
    _, impr_par_time = time_labeling(
        benchmark,
        min_labeling=False,
        partial_labeling=False,
        partial_cutoff=partial_cutoff,
        parallel=True,
    )
    print(f"{impr_par_time:.2f}s")

    speedup_orig = seq_time / orig_par_time if orig_par_time > 0 else float("inf")
    speedup_impr = seq_time / impr_par_time if impr_par_time > 0 else float("inf")
    print(f"  [{repeat}] speedup original  = {speedup_orig:.2f}x")
    print(f"  [{repeat}] speedup improved  = {speedup_impr:.2f}x")

    return {
        "benchmark": benchmark,
        "num_inputs": num_inputs,
        "num_outputs": num_outputs,
        "num_cpu_cores": num_cores,
        "partial_cutoff": partial_cutoff,
        "repeat": repeat,
        "sequential_s": round(seq_time, 4),
        "original_parallel_s": round(orig_par_time, 4),
        "improved_parallel_s": round(impr_par_time, 4),
        "speedup_original": round(speedup_orig, 4),
        "speedup_improved": round(speedup_impr, 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS,
        metavar="BENCHMARK",
    )
    parser.add_argument("--partial-cutoff", type=int, default=DEFAULT_PARTIAL_CUTOFF)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--output-csv", default=OUTPUT_CSV,
                        help=f"Output CSV path (default: {OUTPUT_CSV})")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    all_rows = []
    for benchmark in args.benchmarks:
        print(f"\n=== {benchmark} ===")
        for r in range(1, args.repeats + 1):
            row = run_benchmark(benchmark, args.partial_cutoff, r)
            all_rows.append(row)

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nResults saved to {args.output_csv}")

    print(f"\n{'Benchmark':<20} {'Seq (s)':>10} {'Orig Par':>10} {'Impr Par':>10} "
          f"{'Spdup Orig':>11} {'Spdup Impr':>11}")
    print("-" * 78)
    for row in all_rows:
        if row["repeat"] == 1:
            print(
                f"{row['benchmark']:<20} "
                f"{row['sequential_s']:>10.2f} "
                f"{row['original_parallel_s']:>10.2f} "
                f"{row['improved_parallel_s']:>10.2f} "
                f"{row['speedup_original']:>10.2f}x "
                f"{row['speedup_improved']:>10.2f}x"
            )


if __name__ == "__main__":
    main()
