"""Bit-exact correctness check for a generated DCT circuit.

Synthesises the Verilog the same way SubXPAT does (Yosys -> graphviz), simulates
the resulting gate-level circuit on every (or many) input vectors, and compares
the packed output against the integer golden model ``M @ x + bias`` from the
sidecar JSON. Verifies exactly the circuit SubXPAT ingests.

Usage (from repo root):
    python scripts/dct_verify.py input/ver/dct_p4_b4.v
    python scripts/dct_verify.py input/ver/dct_p8_b8.v --samples 200000
"""

import argparse
import itertools
import json
import os
import sys
import tempfile
from os.path import join as path_join

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import networkx as nx
from Z3Log_patched.verilog import Verilog
from Z3Log_patched.graph import Graph
from Z3Log_patched.utils import convert_verilog_to_gv
from sxpat.labeling import _eval_node, _output_integers


def golden_packed(M, bias, widths, in_bits, X, coeff=None):
    """Packed integer golden model for input matrix X (S x N).

    Full transform: flattened bus-major layout out0=y0[0], ... .
    Single coefficient: the lone output bus value.
    """
    N = M.shape[0]
    if coeff is not None:
        return (X @ M[coeff] + bias[coeff]).astype(object)
    offsets = np.cumsum([0] + list(widths))[:-1]
    packed = np.zeros(X.shape[0], dtype=object)  # python ints (may exceed 64 bit)
    for k in range(N):
        yk = X @ M[k] + bias[k]
        packed = packed + (yk.astype(object) << int(offsets[k]))
    return packed


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('verilog')
    p.add_argument('--samples', type=int, default=None,
                   help='random samples if input space too large for exhaustive')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    side_path = os.path.splitext(args.verilog)[0] + '.json'
    with open(side_path) as f:
        side = json.load(f)
    M = np.array(side['matrix'], dtype=np.int64)
    bias = side['bias']
    widths = side['out_widths']
    W = side['in_bits']
    N = side['points']
    in_max = side['in_max']
    total_in = N * W

    tmp = tempfile.mkdtemp()
    v = Verilog(args.verilog, tv := path_join(tmp, 'c.v'), tmp); v.export_circuit()
    convert_verilog_to_gv(tv, gv := path_join(tmp, 'c.gv'), tmp)
    g = Graph(gv); g.export_graph()

    # build input value matrix X (S x N) and the per-input-bit boolean vectors
    exhaustive = (args.samples is None and total_in <= 20)
    if exhaustive:
        S = 1 << total_in
        codes = np.arange(S, dtype=np.int64)
        bits = {i: ((codes >> i) & 1).astype(bool) for i in range(total_in)}
        print(f'exhaustive: {S} input vectors')
    else:
        S = args.samples or 100000
        rng = np.random.default_rng(args.seed)
        # sample each input bit independently to avoid 64-bit code overflow
        bits = {i: rng.integers(0, 2, size=S, dtype=bool) for i in range(total_in)}
        print(f'random: {S} input vectors')

    # input bit i; x_k = bits [k*W .. k*W+W-1]
    val = {'__ones__': np.ones(S, dtype=bool), '__zeros__': np.zeros(S, dtype=bool)}
    for i in range(total_in):
        val[f'in{i}'] = bits[i]
    X = np.zeros((S, N), dtype=np.int64)
    for k in range(N):
        for b in range(W):
            X[:, k] += bits[k * W + b].astype(np.int64) << b

    for node in nx.topological_sort(g.graph):
        if node not in val:
            val[node] = _eval_node(g, node, val)
    sim_packed = _output_integers(g, val)  # int64 packed (Σ out_i 2^i)

    gold = golden_packed(M, bias, widths, W, X, coeff=side.get('coeff'))
    # compare as python ints (output may exceed 63 bits for big configs)
    sim_obj = np.array([int(x) for x in sim_packed], dtype=object)
    mism = np.where(sim_obj != gold)[0]

    if len(mism) == 0:
        print(f'PASS: all {S} vectors match the golden model (bit-exact)')
        return 0
    print(f'FAIL: {len(mism)} / {S} mismatches')
    for idx in mism[:5]:
        print(f'  x={X[idx].tolist()} sim={int(sim_obj[idx])} gold={int(gold[idx])}')
    return 1


if __name__ == '__main__':
    sys.exit(main())
