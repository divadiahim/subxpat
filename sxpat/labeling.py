from typing import Dict, List, Optional, Tuple

import os
import random
import time
import tempfile
from os.path import join as path_join
from contextlib import redirect_stdout

import networkx as nx
import numpy as np

from Z3Log_patched.verilog import Verilog
from Z3Log_patched.graph import Graph
from Z3Log_patched.z3solver import Z3solver

from Z3Log_patched.utils import convert_verilog_to_gv
from Z3Log_patched.config.config import SINGLE, MAXIMIZE, MONOTONIC
from sxpat.specifications import Paths


# ---------------------------------------------------------------------------
# Simulation-based labelling (approximate mnze upper bounds)
# ---------------------------------------------------------------------------

import re as _re

# operation keyword as it appears at the start of a node's graphviz label,
# which may carry the gate name after a newline (e.g. "not\ng20")
_OP_PATTERN = _re.compile(r'(nand|nor|xnor|xor|and|or|not|buff?|TRUE|FALSE)', _re.IGNORECASE)


def _node_op(graph, node) -> Optional[str]:
    label = graph.graph.nodes[node].get('label', '') or ''
    m = _OP_PATTERN.search(label)
    return m.group(1).lower() if m else None


def _eval_node(graph, node, val: Dict[str, np.ndarray]) -> np.ndarray:
    """Evaluate one node's K-sample boolean vector from its predecessors.

    The synthesised circuits are AIG-style: gates are 2-input ``and`` or
    1-input ``not``; constants are ``TRUE``/``FALSE``; outputs are buffers of
    their single predecessor. Labels may carry the gate name after a newline
    (e.g. ``"not\\ng20"``), so the operation is matched, not compared.
    """
    op = _node_op(graph, node)
    preds = list(graph.graph.predecessors(node))
    if op == 'not':
        return ~val[preds[0]]
    if op == 'and':
        r = val[preds[0]]
        for p in preds[1:]:
            r = r & val[p]
        return r
    if op == 'or':
        r = val[preds[0]]
        for p in preds[1:]:
            r = r | val[p]
        return r
    if op == 'nand':
        r = val[preds[0]]
        for p in preds[1:]:
            r = r & val[p]
        return ~r
    if op == 'nor':
        r = val[preds[0]]
        for p in preds[1:]:
            r = r | val[p]
        return ~r
    if op == 'xor':
        r = val[preds[0]]
        for p in preds[1:]:
            r = r ^ val[p]
        return r
    if op == 'xnor':
        r = val[preds[0]]
        for p in preds[1:]:
            r = r ^ val[p]
        return ~r
    if op == 'true':
        return val['__ones__']
    if op == 'false':
        return val['__zeros__']
    # output buffer (or any single-input pass-through)
    return val[preds[0]]


def _output_integers(graph, val: Dict[str, np.ndarray]) -> np.ndarray:
    """Combine the output bits into one integer per sample: f = sum out_i * 2^i."""
    K = val['__ones__'].shape[0]
    res = np.zeros(K, dtype=np.int64)
    for idx, oname in graph.output_dict.items():
        res = res + (val[oname].astype(np.int64) << int(idx))
    return res


def simulate_mnze_upper_bounds(graph, samples: int, seed: int,
                               constant_value: bool = False) -> Dict[str, Optional[int]]:
    """Return an upper bound on ``mnze_g`` for every gate/constant, by simulation.

    This mirrors how the SMT labelling defines ``mnze_g``: the approximate
    circuit replaces gate ``g`` with the fixed constant ``constant_value`` (the
    same pruning the solver applies). For ``samples`` random input vectors we
    evaluate the exact circuit, then for each gate force it to ``constant_value``,
    re-evaluate its transitive fanout, and record the minimum *nonzero* output
    error observed.

    That observed minimum is a sound upper bound on the true ``mnze_g``: every
    value comes from a real input, so if it is ``<= ET`` the gate is provably
    feasible. A gate whose pruning never changed the output in any sample maps to
    ``None`` (no usable bound from simulation).
    """
    G = graph.graph
    topo = list(nx.topological_sort(G))
    rng = np.random.default_rng(seed)

    val: Dict[str, np.ndarray] = {}
    val['__ones__'] = np.ones(samples, dtype=bool)
    val['__zeros__'] = np.zeros(samples, dtype=bool)
    for idx, name in graph.input_dict.items():
        val[name] = rng.integers(0, 2, size=samples, dtype=bool)

    for node in topo:
        if node in val:
            continue
        val[node] = _eval_node(graph, node, val)

    exact = _output_integers(graph, val)
    const_vec = val['__ones__'] if constant_value else val['__zeros__']

    # transitive fanout (topologically ordered) per perturbable node
    perturbable = list(graph.gate_dict.values()) + list(graph.constant_dict.values())
    topo_index = {n: i for i, n in enumerate(topo)}

    bounds: Dict[str, Optional[int]] = {}
    for g in perturbable:
        cone = nx.descendants(G, g)
        cone_nodes = sorted(cone, key=lambda n: topo_index[n])
        val2 = dict(val)
        val2[g] = const_vec
        for node in cone_nodes:
            val2[node] = _eval_node(graph, node, val2)
        approx = _output_integers(graph, val2)
        err = np.abs(exact - approx)
        nz = err[err > 0]
        bounds[g] = int(nz.min()) if nz.size else None
    return bounds


def _min_output_index(graph) -> Dict[str, int]:
    """Least-significant reachable primary-output index per node (lower-bound side)."""
    rev_output = {node: idx for idx, node in graph.output_dict.items()}
    min_idx: Dict[str, int] = {}
    for node in reversed(list(nx.topological_sort(graph.graph))):
        if node in rev_output:
            min_idx[node] = rev_output[node]
            continue
        best = None
        for succ in graph.graph.successors(node):
            s = min_idx.get(succ)
            if s is not None and (best is None or s < best):
                best = s
        if best is not None:
            min_idx[node] = best
    return min_idx


def simulation_labeling(z3py_obj, et: int, samples: int, seed: int,
                        constant_value: bool = False,
                        smt_fallback: bool = True) -> Tuple[Dict[str, int], Dict]:
    """Hybrid simulation labelling (proposed Strategy 1).

    For each gate, derive a structural lower bound ``L_g = 2^(least-significant
    reachable output)`` and a simulation upper bound ``U_g`` on ``mnze_g``:

    - ``L_g > ET``                -> infeasible, excluded (weight -1, omitted).
    - ``U_g <= ET``               -> feasible, weight = ``U_g`` (no SMT call).
    - otherwise (uncertain)       -> resolved by an exact SMT optimisation call
                                     (when ``smt_fallback``), else excluded.

    Feasibility decisions are sound: ``U_g`` is a real achievable error, so
    ``U_g <= ET`` proves ``mnze_g <= ET``; ``L_g`` is a true lower bound, so
    ``L_g > ET`` proves ``mnze_g > ET``. Weights of simulation-resolved gates
    are upper bounds on the exact ``mnze`` (near-exact in practice), which only
    affects selection heuristics, never the ET-correctness of the final circuit.

    Returns ``(weights, stats)`` where ``weights`` is keyed like the exact
    labels (``g20``) and ``stats`` reports how each gate was resolved.
    """
    # match the setup label_circuit performs before generating SMT scripts
    z3py_obj.experiment = SINGLE
    z3py_obj.set_strategy(MONOTONIC)

    lg = z3py_obj.labeling_graph
    U = simulate_mnze_upper_bounds(lg, samples, seed, constant_value)
    min_out = _min_output_index(lg)
    strip = lambda k: _re.sub(r'^app_', '', k)

    weights: Dict[str, int] = {}
    uncertain: List[str] = []
    n_feasible_sim = n_infeasible = 0

    for node in list(lg.gate_dict.values()) + list(lg.constant_dict.values()):
        L = min_out.get(node)
        if L is not None and 2 ** L > et:
            n_infeasible += 1
            continue
        ub = U.get(node)
        if ub is not None and ub <= et:
            weights[strip(node)] = ub
            n_feasible_sim += 1
        else:
            uncertain.append(node)

    n_smt = 0
    if uncertain and smt_fallback:
        for node in uncertain:
            z3py_obj.create_pruned_z3pyscript_approximate([node], constant_value)
        z3py_obj.run_z3pyscript_labeling()
        z3py_obj.import_labels(constant_value)
        for k, v in z3py_obj.labels.items():
            weights[k] = v
        n_smt = len(uncertain)

    stats = {
        'et': et,
        'samples': samples,
        'feasible_by_simulation': n_feasible_sim,
        'infeasible_structural': n_infeasible,
        'smt_fallback_calls': n_smt,
        'total_gates': n_feasible_sim + n_infeasible + len(uncertain),
    }
    print(f'[simulation] et={et} K={samples} sim_feasible={n_feasible_sim} '
          f'infeasible={n_infeasible} smt_fallback={n_smt}')
    return weights, stats


def warmstart_labeling(z3py_obj, et: int, samples: int, seed: int,
                       constant_value: bool = False) -> Tuple[Dict[str, int], Dict]:
    """Simulation warm-started exact labelling (proposed Option 2).

    Computes the same gate set as partial labelling (gates within the ET cone),
    but seeds each gate's SMT minimisation with the simulation upper bound
    ``U_g`` as a tight initial ceiling (``f_error <= U_g``). The optimum is
    unchanged -- ``mnze_g <= U_g`` always -- so the labels are **identical** to
    exact labelling; only the per-gate solve is cheaper. Gates with structural
    lower bound ``L_g > ET`` are excluded (as in partial labelling).
    """
    z3py_obj.experiment = SINGLE
    z3py_obj.set_strategy(MONOTONIC)

    lg = z3py_obj.labeling_graph
    U = simulate_mnze_upper_bounds(lg, samples, seed, constant_value)
    min_out = _min_output_index(lg)

    # max possible error, used when simulation gives no bound
    max_err = 2 ** lg.num_outputs - 1
    n_labeled = n_infeasible = 0
    for node in list(lg.gate_dict.values()) + list(lg.constant_dict.values()):
        L = min_out.get(node)
        if L is not None and 2 ** L > et:
            n_infeasible += 1
            continue
        ub = U.get(node)
        z3py_obj.set_warm_bound(ub if ub is not None else max_err)
        z3py_obj.create_pruned_z3pyscript_approximate([node], constant_value)
        n_labeled += 1
    z3py_obj.set_warm_bound(None)

    if n_labeled:
        z3py_obj.run_z3pyscript_labeling()
        z3py_obj.import_labels(constant_value)

    stats = {'et': et, 'samples': samples, 'labeled': n_labeled,
             'infeasible_structural': n_infeasible}
    print(f'[warmstart] et={et} K={samples} labeled={n_labeled} infeasible={n_infeasible}')
    return dict(z3py_obj.labels), stats


def labeling_explicit(exact_in_verilog_path: str, current_in_verilog_path: str,
                      run_paths: Paths.RunFiles,
                      min_labeling: bool,
                      partial_labeling: bool, partial_cutoff: int,
                      constant_value: bool = False,
                      parallel: bool = False,
                      prefilter: bool = False,
                      labeling_method: str = 'exact',
                      sim_samples: int = 1024,
                      sim_seed: int = 0,
                      sim_smt_fallback: bool = True,
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

    if labeling_method == 'simulation' and partial_cutoff != -1:
        const = constant_value if isinstance(constant_value, bool) else False
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            weights, _ = simulation_labeling(
                z3py_obj, et=partial_cutoff, samples=sim_samples, seed=sim_seed,
                constant_value=const, smt_fallback=sim_smt_fallback,
            )
        return (weights,) * 2

    if labeling_method == 'warmstart' and partial_cutoff != -1:
        const = constant_value if isinstance(constant_value, bool) else False
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            weights, _ = warmstart_labeling(
                z3py_obj, et=partial_cutoff, samples=sim_samples, seed=sim_seed,
                constant_value=const,
            )
        return (weights,) * 2

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
