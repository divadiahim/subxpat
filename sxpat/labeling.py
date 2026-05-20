from typing import Dict, Tuple

import os
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
        partial=partial_labeling, parallel=parallel
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

    Returns
    -------
    tuple[dict, float]
        ``(labels, elapsed_seconds)`` where *labels* maps gate names to WCE
        values and *elapsed_seconds* is the wall-clock duration of the call.
    """
    from sxpat.specifications import Paths

    verilog_path = os.path.join('input', 'ver', f'{benchmark_name}.v')
    run_paths = Paths.RunFiles(f'labeling_{benchmark_name}', output_base)
    os.makedirs(run_paths.temporary, exist_ok=True)

    t0 = time.perf_counter()
    labels, _ = labeling_explicit(
        verilog_path, verilog_path, run_paths,
        min_labeling=min_labeling,
        partial_labeling=partial_labeling,
        partial_cutoff=partial_cutoff,
        constant_value=constant_value,
        parallel=parallel,
    )
    elapsed = time.perf_counter() - t0
    return labels, elapsed
