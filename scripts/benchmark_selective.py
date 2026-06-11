"""Benchmark fanout-cone selective relabelling against full relabelling.

Simulates SubXPAT iterations: a random connected region of --n-modified gates
plays the role of a subcircuit rewrite site, and only gates whose miter could
have changed (the fanin cone of the region's transitive fanout) are relabelled.
Three modes are measured per (benchmark, trial):

  1. full      — relabel every gate (what SubXPAT currently does each iteration)
  2. selective — relabel only the affected gates (Strategy 2)
  3. combined  — selective + output-significance pre-filter at each ET (Strategy 2 + 1)

Each trial uses a different random rewrite site (seeded, reproducible).
Records wall-clock time and the number of SMT optimisation calls per mode.

By default ETs for the combined mode are fractions of the maximum output value
(matching the SubXPAT paper): ET = round(frac * (2^m - 1)) for frac in
1/16, 1/8, 1/4, 1/2.

Usage (from repo root, with venv active):
    python scripts/benchmark_selective.py --benchmarks adder_i8_o5
    python scripts/benchmark_selective.py --benchmarks adder_i8_o5 --n-modified 8 --trials 5
"""

import argparse
import csv
import multiprocessing
import os
import re
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sxpat.labeling import time_labeling, time_labeling_selective

DEFAULT_BENCHMARKS = [
    "adder_i4_o3",
    "adder_i6_o4",
    "adder_i8_o5",
    "adder_i10_o6",
    "adder_i12_o7",
    "adder_i16_o9",
]
DEFAULT_ET_FRACTIONS = [(1, 16), (1, 8), (1, 4), (1, 2)]
DEFAULT_N_MODIFIED = 5
DEFAULT_TRIALS = 3
DEFAULT_REPEATS = 1
OUTPUT_CSV = os.path.join("output", "report", "selective_benchmark.csv")
CSV_HEADER = [
    "benchmark",
    "num_inputs",
    "num_outputs",
    "num_cpu_cores",
    "n_modified",
    "trial",
    "et",
    "et_frac",
    "repeat",
    "full_s",
    "full_calls",
    "selective_s",
    "selective_calls",
    "combined_s",
    "combined_calls",
]


def _parse_circuit_dims(name: str):
    m = re.search(r"_i(\d+)_o(\d+)", name)
    return (int(m.group(1)), int(m.group(2))) if m else (-1, -1)


def _default_ets(num_outputs: int):
    """ET values as fractions of the maximum output value 2^m - 1."""
    max_err = 2 ** num_outputs - 1
    ets = []
    seen = set()
    for num, den in DEFAULT_ET_FRACTIONS:
        et = max(1, round(max_err * num / den))
        if et not in seen:
            seen.add(et)
            ets.append((et, f"{num}/{den}"))
    return ets


def _tag():
    return f"{uuid.uuid4().hex[:8]}_"


def run_benchmark(benchmark: str, ets, n_modified: int, trials: int,
                  repeat: int, parallel: bool):
    num_inputs, num_outputs = _parse_circuit_dims(benchmark)
    num_cores = multiprocessing.cpu_count()

    # Full relabelling is the per-iteration baseline: measure once per repeat.
    print(f"  [{repeat}] full                     ... ", end="", flush=True)
    full_labels, full_time = time_labeling(
        benchmark, min_labeling=False, partial_labeling=False,
        partial_cutoff=-1, parallel=parallel, run_tag=_tag(), cleanup=True)
    full_calls = len(full_labels)
    print(f"{full_time:.2f}s  calls={full_calls}")

    rows = []
    for trial in range(1, trials + 1):
        seed = 1000 * repeat + trial  # reproducible rewrite sites

        print(f"  [{repeat}] t{trial} selective           ... ", end="", flush=True)
        _, sel_time, sel_stats = time_labeling_selective(
            benchmark, n_modified, seed, parallel=parallel,
            run_tag=_tag(), cleanup=True)
        sel_calls = sel_stats["relabeled"]
        print(f"{sel_time:.2f}s  calls={sel_calls}")

        for et, frac in ets:
            print(f"  [{repeat}] t{trial} combined  (et={et}) ... ", end="", flush=True)
            _, comb_time, comb_stats = time_labeling_selective(
                benchmark, n_modified, seed, et=et, prefilter=True,
                parallel=parallel, run_tag=_tag(), cleanup=True)
            comb_calls = comb_stats["relabeled"]
            print(f"{comb_time:.2f}s  calls={comb_calls}")

            rows.append({
                "benchmark": benchmark,
                "num_inputs": num_inputs,
                "num_outputs": num_outputs,
                "num_cpu_cores": num_cores,
                "n_modified": n_modified,
                "trial": trial,
                "et": et,
                "et_frac": frac,
                "repeat": repeat,
                "full_s": round(full_time, 4),
                "full_calls": full_calls,
                "selective_s": round(sel_time, 4),
                "selective_calls": sel_calls,
                "combined_s": round(comb_time, 4),
                "combined_calls": comb_calls,
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS,
                        metavar="BENCHMARK")
    parser.add_argument("--ets", nargs="+", type=int, default=None,
                        help="Absolute ET values for the combined mode "
                             "(default: 1/16, 1/8, 1/4, 1/2 of max output per benchmark)")
    parser.add_argument("--n-modified", type=int, default=DEFAULT_N_MODIFIED,
                        help=f"Size of the simulated rewrite region (default: {DEFAULT_N_MODIFIED})")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                        help=f"Random rewrite sites per repeat (default: {DEFAULT_TRIALS})")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--sequential", action="store_true",
                        help="Run Z3 scripts sequentially instead of in parallel")
    parser.add_argument("--output-csv", default=OUTPUT_CSV,
                        help=f"Output CSV path (default: {OUTPUT_CSV})")
    args = parser.parse_args()

    parallel = not args.sequential
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    all_rows = []
    for benchmark in args.benchmarks:
        _, num_outputs = _parse_circuit_dims(benchmark)
        if args.ets is not None:
            ets = [(et, str(et)) for et in args.ets]
        else:
            ets = _default_ets(num_outputs)
        print(f"\n=== {benchmark} (n_modified={args.n_modified}, "
              f"ETs: {[e for e, _ in ets]}) ===")
        for r in range(1, args.repeats + 1):
            all_rows.extend(run_benchmark(
                benchmark, ets, args.n_modified, args.trials, r, parallel))

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nResults saved to {args.output_csv}")

    print(f"\n{'Benchmark':<18} {'trial':>5} {'ET':>8} {'frac':>5} "
          f"{'Full(s)':>9} {'Sel(s)':>9} {'Comb(s)':>9} "
          f"{'FullC':>6} {'SelC':>6} {'CombC':>6}")
    print("-" * 96)
    for row in all_rows:
        if row["repeat"] == 1:
            print(
                f"{row['benchmark']:<18} {row['trial']:>5} {row['et']:>8} {row['et_frac']:>5} "
                f"{row['full_s']:>9.2f} {row['selective_s']:>9.2f} {row['combined_s']:>9.2f} "
                f"{row['full_calls']:>6} {row['selective_calls']:>6} {row['combined_calls']:>6}"
            )


if __name__ == "__main__":
    main()
