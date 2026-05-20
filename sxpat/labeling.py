from typing import Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
import glob
import os
from sxpat.utils.filesystem import FS

from Z3Log.verilog import Verilog
from Z3Log.graph import Graph
from Z3Log.utils import convert_verilog_to_gv
from Z3Log.config.config import SINGLE, MAXIMIZE, MONOTONIC
from Z3Log.utils import setup_folder_structure
import Z3Log.config.path as paths
from Z3Log.z3solver import Z3solver


def _label_one_gate(args: tuple) -> Dict[str, int]:
    """Label a single gate and return its WCE as ``{gate_name: wce}``.

    Parameters:
    
    exact_name : str
        Name of the exact (reference) benchmark, e.g. ``"adder_i8_o5"``.
    approx_name : str
        Name of the approximate benchmark (same as exact during labelling).
    gate_key : int
        The integer key used to look up the gate node in the graph dict.
    gate_source : ``"gate"`` | ``"constant"``
        Which dict on the labeling graph to look up — normal logic gates live
        in ``gate_dict``; constant-0/1 nodes live in ``constant_dict``.
        Both are labelled by ``Z3solver.label_circuit()`` so we must handle both.
    style : ``"min"`` | ``"max"``
        Optimisation direction; ``"max"`` computes WCE (worst case error),
        ``"min"`` computes best case error.
    partial_labeling : bool
        Whether to restrict labelling to gates reachable within the error
        threshold (partial labelling mode).

    Returns:
    
    dict
        ``{gate_name: wce}``
    """
    exact_name, approx_name, gate_key, gate_source, style, partial_labeling = args

    # Workers run in fresh subprocesses
    setup_folder_structure()

    z3 = Z3solver(
        exact_name, approx_name,
        experiment=SINGLE, optimization=MAXIMIZE, style=style,
        partial=partial_labeling, parallel=False,
    )
 
    z3.set_strategy(MONOTONIC)

    src_dict = (
        z3.labeling_graph.gate_dict
        if gate_source == "gate"
        else z3.labeling_graph.constant_dict
    )
    removed_gate = [src_dict[gate_key]]

    # Step 1: deep-copy graph, mark this gate as PRUNED, write a .py Z3 script.
    z3.create_pruned_z3pyscript_approximate(removed_gate, constant_value=False)
    # Step 2: run that script as a subprocess; it writes its answer to a CSV.
    z3.run_z3pyscript_labeling()
    # Step 3: parse the CSV and return {gate_name: wce}.
    return z3.import_labels()


def labeling_explicit(exact_benchmark_name: str, approximate_benchmark: str,
                      min_labeling: bool,
                      partial_labeling: bool, partial_cutoff: int,
                      constant_value: bool = False,
                      parallel: bool = False,
                      ) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Label every gate in *approximate_benchmark* with its WCE (worst case error).

    Returns
    -------
    tuple[dict, dict]
        A pair ``(labels, labels)`` where *labels* is a dict mapping gate names to WCE values, e
    """
    # --- Prepare circuits ---------------------------------------------------
    # Convert Verilog to a cleaned GV file (runs Yosys internally).
    verilog_obj_exact = Verilog(exact_benchmark_name)
    verilog_obj_exact.export_circuit()

    verilog_obj_approx = Verilog(approximate_benchmark)
    verilog_obj_approx.export_circuit()

    convert_verilog_to_gv(exact_benchmark_name)
    convert_verilog_to_gv(approximate_benchmark)

    # Load NetworkX graphs from the GV files.
    graph_obj_exact = Graph(exact_benchmark_name)
    graph_obj_approx = Graph(approximate_benchmark)

    graph_obj_exact.export_graph()
    graph_obj_approx.export_graph()

    style = 'min' if min_labeling else 'max'

    if parallel:
        # New path: one subprocess per gate, all run concurrently.
        labels = _labeling_parallel(
            exact_benchmark_name, approximate_benchmark,
            style, partial_labeling,
        )
    else:
        # Original path: single Z3solver generates all scripts then runs them.
        z3py_obj = Z3solver(
            exact_benchmark_name, approximate_benchmark,
            experiment=SINGLE, optimization=MAXIMIZE, style=style,
            partial=partial_labeling, parallel=False,
        )

        if constant_value is False:
            labels = z3py_obj.label_circuit(False, partial=partial_labeling, et=partial_cutoff)
        elif constant_value is True:
            labels = z3py_obj.label_circuit(True, partial=partial_labeling, et=partial_cutoff)
        else:
            labels = z3py_obj.label_circuit(False, partial=partial_labeling, et=partial_cutoff)

    labels_pair = (labels,) * 2

    # Remove per-run Z3 scripts and CSV results; they are not needed after
    # labelling and would accumulate across iterations.
    for folder in [paths.OUTPUT_PATH['report'][0], paths.OUTPUT_PATH['z3'][0]]:
        for dir in glob.glob(f'{folder}/*labeling*'):
            if os.path.isdir(dir):
                FS.rmdir(dir, True)

    return labels_pair


def _labeling_parallel(
    exact_benchmark_name: str,
    approximate_benchmark: str,
    style: str,
    partial_labeling: bool,
) -> Dict[str, int]:
    """Label all gates concurrently, one subprocess per gate.

    Parameters
    ----------
    exact_benchmark_name, approximate_benchmark, style, partial_labeling
        Same semantics as in ``labeling_explicit``.

    Returns
    -------
    dict
        ``{gate_name: wce}`` for every gate and constant node in the circuit.
    """
    # Probe solver: used only to read gate/constant key lists.
    probe = Z3solver(
        exact_benchmark_name, approximate_benchmark,
        experiment=SINGLE, optimization=MAXIMIZE, style=style,
        partial=partial_labeling, parallel=False,
    )
    gate_keys = list(probe.labeling_graph.gate_dict.keys())
    constant_keys = list(probe.labeling_graph.constant_dict.keys())

    # One args tuple per gate, then one per constant node.
    worker_args = (
        [
            (exact_benchmark_name, approximate_benchmark,
             key, "gate", style, partial_labeling)
            for key in gate_keys
        ] + [
            (exact_benchmark_name, approximate_benchmark,
             key, "constant", style, partial_labeling)
            for key in constant_keys
        ]
    )

    merged: Dict[str, int] = {}
    n_workers = min(multiprocessing.cpu_count(), len(worker_args))

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all gates at once; executor manages the process pool.
        futures = {executor.submit(_label_one_gate, args): args[2] for args in worker_args}
        # as_completed yields each future the moment its process finishes.
        for future in as_completed(futures):
            result = future.result()   # re-raises any exception from the worker
            merged.update(result)

    return merged


def time_labeling(
    exact_benchmark_name: str,
    approximate_benchmark: str,
    min_labeling: bool,
    partial_labeling: bool,
    partial_cutoff: int,
    parallel: bool,
    constant_value: bool = False,
) -> Tuple[Dict[str, int], float]:
    """Run ``labeling_explicit`` and also return elapsed wall-clock time.

    A thin convenience wrapper used by the benchmarking script
    (``scripts/benchmark_labeling.py``) and the performance test suite
    (``test/test_labeling_performance.py``).

    Parameters
    ----------
    All parameters are forwarded unchanged to ``labeling_explicit``.

    Returns
    -------
    tuple[dict, float]
        ``(labels, elapsed_seconds)`` where *labels* maps gate names to WCE
        values and *elapsed_seconds* is the wall-clock duration of the full
        labelling call measured with ``time.perf_counter``.
    """
    t0 = time.perf_counter()
    labels, _ = labeling_explicit(
        exact_benchmark_name, approximate_benchmark,
        min_labeling=min_labeling,
        partial_labeling=partial_labeling,
        partial_cutoff=partial_cutoff,
        constant_value=constant_value,
        parallel=parallel,
    )
    elapsed = time.perf_counter() - t0
    return labels, elapsed
