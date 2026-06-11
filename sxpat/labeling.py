from typing import Dict, List, Tuple

import os
import random
import time
import tempfile
from os.path import join as path_join
from contextlib import redirect_stdout

from Z3Log_patched.verilog import Verilog
from Z3Log_patched.graph import Graph
from Z3Log_patched.z3solver import Z3solver

from Z3Log_patched.utils import convert_verilog_to_gv
from Z3Log_patched.config.config import SINGLE, MAXIMIZE
from sxpat.specifications import Paths


def labeling_explicit(exact_in_verilog_path: str, current_in_verilog_path: str,
                      run_paths: Paths.RunFiles,
                      min_labeling: bool,
                      partial_labeling: bool, partial_cutoff: int,
                      constant_value: bool = False,
                      parallel: bool = False,
                      prefilter: bool = False,
                      ) -> Tuple[Dict[str, int], Dict[str, int]]:

    # 1) create a clean verilog out of exact and approximate circuits
    verilog_obj_exact = Verilog(exact_in_verilog_path, tmp_exact_v := path_join(run_paths.temporary, f'lbl_exact.v'), run_paths.temporary)
    verilog_obj_exact.export_circuit()
    verilog_obj_approx = Verilog(current_in_verilog_path, tmp_current_v := path_join(run_paths.temporary, f'lbl_current.v'), run_paths.temporary)
    verilog_obj_approx.export_circuit()

    convert_verilog_to_gv(tmp_exact_v, tmp_exact_gv := path_join(run_paths.temporary, 'lbl_exact.gv'), run_paths.temporary)
    convert_verilog_to_gv(tmp_current_v, tmp_current_gv := path_join(run_paths.temporary, 'lbl_current.gv'), run_paths.temporary)

    # 2) convert clean exact and approximate circuits into their clean gv formats
    graph_obj_exact = Graph(tmp_exact_gv)
    graph_obj_current = Graph(tmp_current_gv)
    graph_obj_exact.export_graph()
    graph_obj_current.export_graph()

    # convert gv to z3 expression
    style = 'min' if min_labeling else 'max'
    z3py_obj = Z3solver(
        tmp_exact_gv, tmp_current_gv, run_paths.temporary,
        experiment=SINGLE, optimization=MAXIMIZE, style=style,
        partial=partial_labeling, parallel=parallel,
        prefilter=prefilter,
    )

    with open(os.devnull, 'w') as f, redirect_stdout(f): # suppress prints
        if constant_value is False:
            labels_pair = (
                z3py_obj.label_circuit(False, partial=partial_labeling, et=partial_cutoff),
            ) * 2
        elif constant_value is True:
            labels_pair = (
                z3py_obj.label_circuit(True, partial=partial_labeling, et=partial_cutoff),
            ) * 2
        else:
            labels_pair = (
                z3py_obj.label_circuit(False, partial=partial_labeling, et=partial_cutoff),
                z3py_obj.label_circuit(True, partial=partial_labeling, et=partial_cutoff),
            )

    return labels_pair


def time_labeling(
    benchmark_name: str,
    min_labeling: bool,
    partial_labeling: bool,
    partial_cutoff: int,
    parallel: bool,
    constant_value: bool = False,
    output_base: str = 'output',
    prefilter: bool = False,
    run_tag: str = '',
    cleanup: bool = False,
) -> Tuple[Dict[str, int], float]:
    """Run ``labeling_explicit`` for a named benchmark and return elapsed time.

    Convenience wrapper for benchmarking scripts.  Creates a temporary
    ``Paths.RunFiles`` so callers don't need to build one manually.

    Parameters
    ----------
    benchmark_name:
        Name of the benchmark, e.g. ``"adder_i8_o5"``.  The Verilog file is
        expected at ``input/ver/<benchmark_name>.v``.
    output_base:
        Base directory for output files (default: ``'output'``).
    run_tag:
        Optional unique tag mixed into the run id.  ``RunFiles`` places its
        temporary folder under the system temp dir keyed by run id, so two
        runs with the same benchmark share (and contaminate) that folder
        unless their tags differ.
    cleanup:
        If True, delete the run's temporary folder after the labels have been
        imported (recommended together with a unique ``run_tag``).

    Returns
    -------
    tuple[dict, float]
        ``(labels, elapsed_seconds)`` where *labels* maps gate names to WCE
        values and *elapsed_seconds* is the wall-clock duration of the call.
    """
    from sxpat.specifications import Paths

    verilog_path = os.path.join('input', 'ver', f'{benchmark_name}.v')
    run_id = f'labeling_{run_tag}{benchmark_name}' if run_tag else f'labeling_{benchmark_name}'
    run_paths = Paths.RunFiles(run_id, output_base)
    os.makedirs(run_paths.temporary, exist_ok=True)

    try:
        t0 = time.perf_counter()
        labels, _ = labeling_explicit(
            verilog_path, verilog_path, run_paths,
            min_labeling=min_labeling,
            partial_labeling=partial_labeling,
            partial_cutoff=partial_cutoff,
            constant_value=constant_value,
            parallel=parallel,
            prefilter=prefilter,
        )
        elapsed = time.perf_counter() - t0
    finally:
        if cleanup:
            import shutil
            shutil.rmtree(run_paths.temporary, ignore_errors=True)
    return labels, elapsed


def pick_rewrite_region(graph, size: int, rng: random.Random) -> List[str]:
    """Pick a random connected region of ``size`` gates to simulate a rewrite site.

    SubXPAT rewrites convex, connected subcircuits; this mimics one by a BFS
    over the undirected gate adjacency starting from a random gate.
    """
    gates = sorted(graph.gate_dict.values()) + sorted(graph.constant_dict.values())
    if not gates:
        return []
    gate_set = set(gates)
    start = rng.choice(gates)

    region: List[str] = []
    seen = {start}
    queue = [start]
    while queue and len(region) < size:
        v = queue.pop(0)
        region.append(v)
        neighbours = list(graph.graph.successors(v)) + list(graph.graph.predecessors(v))
        rng.shuffle(neighbours)
        for n in neighbours:
            if n in gate_set and n not in seen:
                seen.add(n)
                queue.append(n)
    return region


def time_labeling_selective(
    benchmark_name: str,
    n_modified: int,
    seed: int,
    *,
    et: int = -1,
    prefilter: bool = False,
    parallel: bool = True,
    min_labeling: bool = False,
    constant_value: bool = False,
    output_base: str = 'output',
    run_tag: str = '',
    cleanup: bool = False,
    previous_labels: Dict = None,
) -> Tuple[Dict[str, int], float, Dict]:
    """Time one selective-relabelling invocation for a simulated rewrite.

    A random connected region of ``n_modified`` gates (seeded by ``seed``) plays
    the role of the gates modified by a subcircuit rewrite; only gates whose
    miter could have changed are relabelled. With ``prefilter=True`` and a valid
    ``et``, the output-significance pre-filter is applied on top.

    Returns ``(labels, elapsed_seconds, stats)`` where *stats* is the solver's
    ``selective_stats`` dict (relabeled / skipped_cone / skipped_prefilter counts).
    """
    from sxpat.specifications import Paths
    from Z3Log_patched.verilog import Verilog
    from Z3Log_patched.graph import Graph
    from Z3Log_patched.z3solver import Z3solver
    from Z3Log_patched.utils import convert_verilog_to_gv
    from Z3Log_patched.config.config import SINGLE, MAXIMIZE

    verilog_path = os.path.join('input', 'ver', f'{benchmark_name}.v')
    run_id = f'labeling_{run_tag}{benchmark_name}' if run_tag else f'labeling_{benchmark_name}'
    run_paths = Paths.RunFiles(run_id, output_base)
    os.makedirs(run_paths.temporary, exist_ok=True)

    try:
        t0 = time.perf_counter()

        verilog_obj = Verilog(verilog_path, tmp_v := path_join(run_paths.temporary, 'lbl_sel.v'), run_paths.temporary)
        verilog_obj.export_circuit()
        convert_verilog_to_gv(tmp_v, tmp_gv := path_join(run_paths.temporary, 'lbl_sel.gv'), run_paths.temporary)
        graph_obj = Graph(tmp_gv)
        graph_obj.export_graph()

        style = 'min' if min_labeling else 'max'
        z3py_obj = Z3solver(
            tmp_gv, tmp_gv, run_paths.temporary,
            experiment=SINGLE, optimization=MAXIMIZE, style=style,
            partial=False, parallel=parallel,
            prefilter=prefilter,
        )

        rng = random.Random(seed)
        modified_gates = pick_rewrite_region(z3py_obj.labeling_graph, n_modified, rng)

        with open(os.devnull, 'w') as f, redirect_stdout(f):  # suppress prints
            labels = z3py_obj.label_circuit_selective(
                modified_gates, constant_value=constant_value, et=et,
                previous_labels=previous_labels,
            )
        elapsed = time.perf_counter() - t0
        stats = dict(z3py_obj.selective_stats)
        stats['modified_gates'] = modified_gates
    finally:
        if cleanup:
            import shutil
            shutil.rmtree(run_paths.temporary, ignore_errors=True)
    return labels, elapsed, stats
