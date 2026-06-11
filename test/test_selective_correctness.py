"""Correctness test: selective relabelling must agree with full labelling.

On an unchanged circuit, every label computed by label_circuit_selective()
(for any simulated rewrite region) must equal the label computed by a full
labelling run, and gates outside the affected region must inherit the
previous-iteration labels passed in.

Usage (from repo root, with venv active):
    pytest test/test_selective_correctness.py -v
"""

import os
import sys
import uuid

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sxpat.labeling import time_labeling, time_labeling_selective

BENCHMARK = "adder_i8_o5"
N_MODIFIED = 5


def _tag():
    return f"{uuid.uuid4().hex[:8]}_"


@pytest.fixture(scope="module")
def full_labels():
    labels, _ = time_labeling(
        BENCHMARK, min_labeling=False, partial_labeling=False,
        partial_cutoff=-1, parallel=True, run_tag=_tag(), cleanup=True)
    return labels


class TestSelectiveCorrectness:

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_selective_labels_match_full(self, full_labels, seed):
        sel, _, stats = time_labeling_selective(
            BENCHMARK, N_MODIFIED, seed, parallel=True,
            run_tag=_tag(), cleanup=True)
        assert stats["relabeled"] == len(sel)
        mismatches = {k: (full_labels.get(k), v)
                      for k, v in sel.items() if full_labels.get(k) != v}
        assert not mismatches, f"Label mismatches (gate: (full, selective)): {mismatches}"

    def test_previous_labels_are_merged(self, full_labels):
        sel, _, stats = time_labeling_selective(
            BENCHMARK, N_MODIFIED, 1, parallel=True,
            run_tag=_tag(), cleanup=True, previous_labels=full_labels)
        # with previous labels supplied, the result must cover every gate
        assert set(sel.keys()) == set(full_labels.keys())
        # and fresh labels must not be overwritten by the merge
        mismatches = {k: (full_labels[k], sel[k])
                      for k in full_labels if sel[k] != full_labels[k]}
        assert not mismatches

    def test_combined_with_prefilter_subset_of_selective(self, full_labels):
        sel, _, sel_stats = time_labeling_selective(
            BENCHMARK, N_MODIFIED, 2, parallel=True,
            run_tag=_tag(), cleanup=True)
        comb, _, comb_stats = time_labeling_selective(
            BENCHMARK, N_MODIFIED, 2, et=2, prefilter=True, parallel=True,
            run_tag=_tag(), cleanup=True)
        assert comb_stats["relabeled"] <= sel_stats["relabeled"]
        assert set(comb.keys()) <= set(sel.keys())
        mismatches = {k: (sel[k], v) for k, v in comb.items() if sel[k] != v}
        assert not mismatches
