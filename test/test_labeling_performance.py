"""Performance test: parallel labelling must be faster than sequential.

Complements test_labeling_correctness.py with timing assertions.
Also re-checks correctness on the performance benchmark.

Usage (from repo root, with venv active):
    pytest test/test_labeling_performance.py -v
    pytest test/test_labeling_performance.py -v --min-speedup 1.5
"""

import multiprocessing
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Z3Log.utils import setup_folder_structure
from sxpat.labeling import time_labeling


# A slightly larger benchmark than the correctness tests so the speedup is real.
BENCHMARK = "adder_i8_o5"
PARTIAL_CUTOFF = 4

# Minimum acceptable speedup. On a single-core machine this test is skipped.
# Passed via --min-speedup CLI option or via the module-level default.
DEFAULT_MIN_SPEEDUP = 1.2


def pytest_addoption(parser):
    parser.addoption(
        "--min-speedup", type=float, default=DEFAULT_MIN_SPEEDUP,
        help=f"Minimum parallel speedup to assert (default: {DEFAULT_MIN_SPEEDUP})",
    )


@pytest.fixture(scope="module", autouse=True)
def setup_benchmark():
    setup_folder_structure()


@pytest.fixture(scope="module")
def timing_results():
    """Run both modes once and cache the timing results for all tests."""
    _, seq_time = time_labeling(
        BENCHMARK, BENCHMARK,
        min_labeling=False,
        partial_labeling=False,
        partial_cutoff=PARTIAL_CUTOFF,
        parallel=False,
    )
    labels_par, par_time = time_labeling(
        BENCHMARK, BENCHMARK,
        min_labeling=False,
        partial_labeling=False,
        partial_cutoff=PARTIAL_CUTOFF,
        parallel=True,
    )
    return {"seq_time": seq_time, "par_time": par_time, "labels_par": labels_par}


@pytest.fixture(scope="module")
def labels_seq():
    labels, _ = time_labeling(
        BENCHMARK, BENCHMARK,
        min_labeling=False,
        partial_labeling=False,
        partial_cutoff=PARTIAL_CUTOFF,
        parallel=False,
    )
    return labels


class TestLabellingPerformance:
    """Timing assertions for the parallel labelling path."""

    @pytest.mark.skipif(
        multiprocessing.cpu_count() < 2,
        reason="Parallel speedup test requires at least 2 CPU cores",
    )
    def test_parallel_is_faster(self, timing_results, request):
        min_speedup = request.config.getoption("--min-speedup", default=DEFAULT_MIN_SPEEDUP)
        seq = timing_results["seq_time"]
        par = timing_results["par_time"]
        speedup = seq / par if par > 0 else float("inf")
        assert speedup >= min_speedup, (
            f"Expected parallel speedup >= {min_speedup}× on {BENCHMARK}, "
            f"got {speedup:.2f}× (seq={seq:.2f}s, par={par:.2f}s)"
        )

    def test_parallel_time_is_positive(self, timing_results):
        assert timing_results["par_time"] > 0, "Parallel labelling took zero seconds"

    def test_sequential_time_is_positive(self, timing_results):
        assert timing_results["seq_time"] > 0, "Sequential labelling took zero seconds"

    def test_parallel_correctness_under_timing(self, timing_results, labels_seq):
        """Correctness check: parallel labels must match sequential ones."""
        labels_par = timing_results["labels_par"]
        assert set(labels_par.keys()) == set(labels_seq.keys()), (
            "Key mismatch between sequential and parallel labels"
        )
        mismatches = {
            k: (labels_seq[k], labels_par[k])
            for k in labels_seq
            if labels_seq.get(k) != labels_par.get(k)
        }
        assert not mismatches, (
            "Label value mismatches (gate: (seq, par)):\n"
            + "\n".join(f"  {k}: {v}" for k, v in sorted(mismatches.items()))
        )
