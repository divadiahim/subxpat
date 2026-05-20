"""Profile a SubXPAT exploration to identify runtime hotspots.

Wraps explore_grid() in cProfile and prints the top N functions sorted by
cumulative time.  Optionally saves a .prof file for further inspection with
snakeviz or pstats.

Usage (from repo root, with venv active):
    python profiling/profile_subxpat.py adder_i8_o5 --max-error 4
    python profiling/profile_subxpat.py adder_i16_o9 --max-error 4 --parallel
    python profiling/profile_subxpat.py adder_i8_o5 --max-error 4 \\
        --output profiling/results/adder8.prof
"""

import argparse
import cProfile
import io
import os
import pstats
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Z3Log.utils import setup_folder_structure
from sxpat.specifications import Specifications
from sxpat.xplore import explore_grid


def _build_specs(benchmark: str, max_error: int, parallel: bool) -> Specifications:
    """Construct a Specifications object for a minimal SubXPAT run."""
    backup = sys.argv
    sys.argv = [
        "main.py",
        benchmark,                         # positional: exact-benchmark (must come first)
        "--current-benchmark", benchmark,
        "--max-error", str(max_error),
        "--subxpat",
        "--encoding", "z3int",
        "--extraction-mode", "2",
        "--input-max", "4",
        "--output-max", "1",
        "--min-subgraph-size", "1",
        "--max-lpp", "2",
        "--max-ppo", "2",
        "--error-partitioning", "asc",
    ]
    if parallel:
        sys.argv.append("--parallel")
    specs = Specifications.parse_args()
    sys.argv = backup
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a SubXPAT run.")
    parser.add_argument("benchmark", help="Benchmark name, e.g. adder_i8_o5")
    parser.add_argument("--max-error", type=int, default=4)
    parser.add_argument("--parallel", action="store_true",
                        help="Enable parallel labelling")
    parser.add_argument("--output", default=None,
                        help="Path to save .prof file (optional)")
    parser.add_argument("--top", type=int, default=25,
                        help="Number of top functions to print (default 25)")
    args = parser.parse_args()

    setup_folder_structure()
    specs = _build_specs(args.benchmark, args.max_error, args.parallel)

    profiler = cProfile.Profile()
    profiler.enable()
    explore_grid(specs)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(args.top)
    report = stream.getvalue()
    print(report)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        profiler.dump_stats(args.output)
        print(f"Profile data saved to: {args.output}")

        txt_path = os.path.splitext(args.output)[0] + ".txt"
        with open(txt_path, "w") as fh:
            fh.write(report)
        print(f"Profile report saved to: {txt_path}")


if __name__ == "__main__":
    main()
