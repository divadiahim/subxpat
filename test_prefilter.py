"""Standalone test for the structural pre-filter logic in z3zolvernew.

Builds a real circuit graph (via the working Z3Log_patched utilities), then:
  1. runs the pre-filter's min-output-index computation,
  2. derives the set of gates the pre-filter would LABEL (2^LSB <= et),
  3. derives the set the existing partial-labelling cone traversal would label,
  4. asserts the two sets match (correctness of the structural pre-filter),
  5. reports how many SMT calls the pre-filter eliminates vs labelling all gates.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

import networkx as nx

from Z3Log_patched.verilog import Verilog
from Z3Log_patched.graph import Graph
from Z3Log_patched.utils import convert_verilog_to_gv


def build_graph(benchmark: str, tmp: str) -> Graph:
    vpath = os.path.join("input", "ver", f"{benchmark}.v")
    v = Verilog(vpath, tv := os.path.join(tmp, "c.v"), tmp)
    v.export_circuit()
    convert_verilog_to_gv(tv, gv := os.path.join(tmp, "c.gv"), tmp)
    g = Graph(gv)
    g.export_graph()
    return g


def min_output_index(graph):
    """Replicate Z3solver._compute_min_output_index without instantiating it."""
    rev = {node: idx for idx, node in graph.output_dict.items()}
    min_idx = {}
    for node in reversed(list(nx.topological_sort(graph.graph))):
        if node in rev:
            min_idx[node] = rev[node]
            continue
        best = None
        for succ in graph.graph.successors(node):
            s = min_idx.get(succ)
            if s is not None and (best is None or s < best):
                best = s
        if best is not None:
            min_idx[node] = best
    return min_idx


def prefilter_label_set(graph, et):
    """Gates the pre-filter would label (not skip)."""
    min_out = min_output_index(graph)
    labeled = set()
    for key in graph.gate_dict:
        gate = graph.gate_dict[key]
        lsb = min_out.get(gate)
        if lsb is not None and 2 ** lsb > et:
            continue
        labeled.add(gate)
    return labeled, min_out


def partial_label_set(graph, et):
    """Replicate the existing partial-labelling cone traversal."""
    def is_input(node):
        return node in graph.input_dict.values()

    already = set()
    for output_idx in sorted(graph.output_dict):
        if 2 ** output_idx > et:
            break
        stack = list(graph.graph.predecessors(graph.output_dict[output_idx]))
        while stack:
            gate = stack.pop()
            if not is_input(gate) and gate not in already:
                if gate in graph.gate_dict.values() or gate in graph.constant_dict.values():
                    already.add(gate)
                stack.extend(list(graph.graph.predecessors(gate)))
    return already


def main():
    benchmark = sys.argv[1] if len(sys.argv) > 1 else "adder_i8_o5"
    ets = [1, 2, 4, 8, 16]

    with tempfile.TemporaryDirectory() as tmp:
        g = build_graph(benchmark, tmp)
        total_gates = len(g.gate_dict) + len(g.constant_dict)
        print(f"Benchmark: {benchmark}")
        print(f"  inputs={g.num_inputs} outputs={g.num_outputs} "
              f"gates={g.num_gates} constants={g.num_constants}")
        print(f"  total labelable gates = {total_gates}\n")

        all_ok = True
        print(f"  {'et':>4} {'prefilter':>10} {'partial':>9} {'skipped':>8} "
              f"{'match':>6}")
        print("  " + "-" * 44)
        for et in ets:
            pf, _ = prefilter_label_set(g, et)
            pl = partial_label_set(g, et)
            skipped = total_gates - len(pf)
            match = (pf == pl)
            all_ok = all_ok and match
            print(f"  {et:>4} {len(pf):>10} {len(pl):>9} {skipped:>8} "
                  f"{'OK' if match else 'FAIL':>6}")
            if not match:
                print(f"    pf-only: {sorted(pf - pl)}")
                print(f"    pl-only: {sorted(pl - pf)}")

        print()
        print("RESULT:", "ALL MATCH — prefilter == partial labelling set"
              if all_ok else "MISMATCH FOUND")
        return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
