"""Correctness test: parallel labelling must produce identical results to sequential.

Runs labeling_explicit() on a small benchmark (adder_i8_o5) with parallel=False
and parallel=True and asserts that the two label dictionaries are equal.

Usage (from repo root, with venv active):
    pytest test/test_labeling_correctness.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sxpat.labeling import time_labeling

BENCHMARK = "adder_i8_o5"
PARTIAL_CUTOFF = 4


def _label(parallel: bool) -> dict:
    labels, _ = time_labeling(
        BENCHMARK,
        min_labeling=False,
        partial_labeling=False,
        partial_cutoff=PARTIAL_CUTOFF,
        parallel=parallel,
    )
    return labels


class TestLabellingCorrectness:
    """Verify that the parallel path produces the same labels as the sequential path."""

    def test_parallel_keys_match_sequential(self):
        seq = _label(parallel=False)
        par = _label(parallel=True)
        assert set(par.keys()) == set(seq.keys()), (
            f"Key mismatch — sequential has {sorted(seq.keys())}, "
            f"parallel has {sorted(par.keys())}"
        )

    def test_parallel_values_match_sequential(self):
        seq = _label(parallel=False)
        par = _label(parallel=True)
        mismatches = {k: (seq[k], par[k]) for k in seq if seq.get(k) != par.get(k)}
        assert not mismatches, (
            "Label value mismatches (gate: (sequential, parallel)):\n"
            + "\n".join(f"  {k}: {v}" for k, v in sorted(mismatches.items()))
        )

    def test_parallel_returns_non_empty_dict(self):
        par = _label(parallel=True)
        assert len(par) > 0, "Parallel labelling returned an empty dictionary"
