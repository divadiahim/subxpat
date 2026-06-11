"""Benchmark the output-significance pre-filter against full and partial labelling.

For each benchmark and each error threshold (ET), runs gate labelling in three modes:
  1. full      — label every gate (no ET-based skipping)
  2. partial   — the original SubXPAT partial-labelling cone traversal
  3. prefilter — the output-significance pre-filter (skip gates whose
                 least-significant reachable output has weight > ET)

Records wall-clock time and the number of SMT optimisation calls per mode.
Each measurement runs in a fresh temporary output directory so that label
CSVs from one mode can never contaminate the counts of another.

By default ETs are derived from the circuit's output count m as fractions of
the maximum representable output value (matching the ET ranges used in the
SubXPAT paper): ET = round(frac * (2^m - 1)) for frac in 1/16, 1/8, 1/4, 1/2.

Usage (from repo root, with venv active):
    python scripts/benchmark_prefilter.py --benchmarks adder_i8_o5
    python scripts/benchmark_prefilter.py --benchmarks adder_i8_o5 --ets 1 4 16
    python scripts/benchmark_prefilter.py --repeats 3 --output-csv out.csv
"""

import argparse
import csv
import multiprocessing
import os
import re
import sys
import uuid

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
DEFAULT_ET_FRACTIONS = [(1, 16), (1, 8), (1, 4), (1, 2)]
DEFAULT_REPEATS = 1
OUTPUT_CSV = os.path.join("output", "report", "prefilter_benchmark.csv")
CSV_HEADER = [
    "benchmark",
    "num_inputs",
    "num_outputs",
    "num_cpu_cores",
    "et",
    "et_frac",
    "repeat",
    "full_s",
    "full_calls",
    "partial_s",
    "partial_calls",
    "prefilter_s",
    "prefilter_calls",
    "skipped_vs_full",
]


def _parse_circuit_dims(name: str):
    m = re.search(r"_i(\d+)_o(\d+)", name)
    return (int(m.group(1)), int(m.group(2))) if m else (-1, -1)


def _default_ets(num_outputs: int):
    """ET values as fractions of the maximum output value 2^m - 1."""
    max_err = 2 ** num_outputs - 1
    ets = []
    for num, den in DEFAULT_ET_FRACTIONS:
        et = max(1, round(max_err * num / den))
        ets.append((et, f"{num}/{den}"))
    # dedupe while keeping the first fraction label for each value
    seen = set()
    out = []
    for et, frac in ets:
        if et not in seen:
            seen.add(et)
            out.append((et, frac))
    return out


def _timed_labeling(benchmark: str, *, partial: bool, prefilter: bool,
                    et: int, parallel: bool):
    """Run one labelling invocation in an isolated run dir; return (calls, seconds).

    A unique run_tag forces RunFiles into a fresh temporary folder, so label
    CSVs from previous measurements cannot inflate the call count; cleanup=True
    removes the folder afterwards.
    """
    labels, elapsed = time_labeling(
        benchmark,
        min_labeling=False,
        partial_labeling=partial,
        partial_cutoff=et,
        parallel=parallel,
        prefilter=prefilter,
        run_tag=f"{uuid.uuid4().hex[:8]}_",
        cleanup=True,
    )
    return len(labels), elapsed


def run_benchmark(benchmark: str, ets, repeat: int, parallel: bool):
    num_inputs, num_outputs = _parse_circuit_dims(benchmark)
    num_cores = multiprocessing.cpu_count()

    # Full labelling is ET-independent: measure it once per repeat.
    print(f"  [{repeat}] full      (et=-)    ... ", end="", flush=True)
    full_calls, full_time = _timed_labeling(
        benchmark, partial=False, prefilter=False, et=-1, parallel=parallel)
    print(f"{full_time:.2f}s  calls={full_calls}")

    rows = []
    for et, frac in ets:
        print(f"  [{repeat}] partial   (et={et}) ... ", end="", flush=True)
        partial_calls, partial_time = _timed_labeling(
            benchmark, partial=True, prefilter=False, et=et, parallel=parallel)
        print(f"{partial_time:.2f}s  calls={partial_calls}")

        print(f"  [{repeat}] prefilter (et={et}) ... ", end="", flush=True)
        prefilter_calls, prefilter_time = _timed_labeling(
            benchmark, partial=False, prefilter=True, et=et, parallel=parallel)
        print(f"{prefilter_time:.2f}s  calls={prefilter_calls}")

        rows.append({
            "benchmark": benchmark,
            "num_inputs": num_inputs,
            "num_outputs": num_outputs,
            "num_cpu_cores": num_cores,
            "et": et,
            "et_frac": frac,
            "repeat": repeat,
            "full_s": round(full_time, 4),
            "full_calls": full_calls,
            "partial_s": round(partial_time, 4),
            "partial_calls": partial_calls,
            "prefilter_s": round(prefilter_time, 4),
            "prefilter_calls": prefilter_calls,
            "skipped_vs_full": full_calls - prefilter_calls,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS,
                        metavar="BENCHMARK")
    parser.add_argument("--ets", nargs="+", type=int, default=None,
                        help="Absolute ET values (default: 1/16, 1/8, 1/4, 1/2 "
                             "of the maximum output value per benchmark)")
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
        print(f"\n=== {benchmark} (ETs: {[e for e, _ in ets]}) ===")
        for r in range(1, args.repeats + 1):
            all_rows.extend(run_benchmark(benchmark, ets, r, parallel))

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nResults saved to {args.output_csv}")

    print(f"\n{'Benchmark':<18} {'ET':>8} {'frac':>5} "
          f"{'Full(s)':>9} {'Part(s)':>9} {'Pref(s)':>9} "
          f"{'FullC':>6} {'PartC':>6} {'PrefC':>6}")
    print("-" * 84)
    for row in all_rows:
        if row["repeat"] == 1:
            print(
                f"{row['benchmark']:<18} {row['et']:>8} {row['et_frac']:>5} "
                f"{row['full_s']:>9.2f} {row['partial_s']:>9.2f} {row['prefilter_s']:>9.2f} "
                f"{row['full_calls']:>6} {row['partial_calls']:>6} {row['prefilter_calls']:>6}"
            )


if __name__ == "__main__":
    main()
