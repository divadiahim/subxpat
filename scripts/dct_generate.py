"""Generate a combinational fixed-point integer DCT circuit in Verilog.

The transform is an integer-scaled DCT-II: each output is a dot product of the
input vector with a row of an integer coefficient matrix

    M[k,n] = round(SCALE * a_k * cos(pi*(2n+1)*k / (2N))),   a_0 = sqrt(1/N), a_k = sqrt(2/N)

which is exact integer arithmetic, so the circuit has a bit-exact golden model
(numpy ``M @ x``). To keep the circuit unsigned (matching the SubXPAT benchmark
convention and its weighted-sum error model), each output is biased by a
constant ``B[k] = -min_x(M[k]·x)`` so that the stored value ``M[k]·x + B[k]`` is
always >= 0. The bias is a known constant, removed again in the image pipeline.

Outputs a Verilog module and a JSON sidecar (matrix, bias, scale, widths) used
by the verification and image-quality scripts.

Usage (from repo root):
    python scripts/dct_generate.py --points 4 --in-bits 4 --out input/ver/dct_p4_b4.v
    python scripts/dct_generate.py --points 8 --in-bits 8 --out input/ver/dct_p8_b8.v
"""

import argparse
import json
import math
import os

import numpy as np


def dct_matrix(N: int, scale: int) -> np.ndarray:
    M = np.zeros((N, N), dtype=np.int64)
    for k in range(N):
        a = math.sqrt(1.0 / N) if k == 0 else math.sqrt(2.0 / N)
        for n in range(N):
            M[k, n] = round(scale * a * math.cos(math.pi * (2 * n + 1) * k / (2 * N)))
    return M


def output_bias_and_width(M: np.ndarray, in_max: int):
    """Per-output bias making the stored value non-negative, and the bit width."""
    N = M.shape[0]
    biases = []
    widths = []
    for k in range(N):
        row = M[k]
        min_val = int(sum(c * (in_max if c < 0 else 0) for c in row))
        max_val = int(sum(c * (in_max if c > 0 else 0) for c in row))
        bias = -min_val
        top = max_val + bias  # max stored value
        width = max(1, top.bit_length())
        biases.append(bias)
        widths.append(width)
    return biases, widths


def _coeff_expr(M, biases, k, N):
    terms = []
    for n in range(N):
        c = int(M[k, n])
        if c == 0:
            continue
        terms.append(f'{c}*x{n}')
    expr = ' + '.join(terms) if terms else '0'
    if biases[k]:
        expr = f'{expr} + {biases[k]}'
    return expr


def emit_verilog(M: np.ndarray, biases, widths, in_bits: int, module: str,
                 coeff: int = None) -> str:
    """Emit the full N-output transform, or a single coefficient (``coeff``).

    A single-coefficient circuit has one integer output, so SubXPAT's
    weighted-sum error metric measures exactly that coefficient's error -- the
    correct granularity for approximating a multi-output transform.
    """
    N = M.shape[0]
    in_ports = [f'x{n}' for n in range(N)]
    lines = []
    if coeff is None:
        out_ports = [f'y{k}' for k in range(N)]
        lines.append(f'module {module}({", ".join(in_ports + out_ports)});')
        for n in range(N):
            lines.append(f'input [{in_bits-1}:0] x{n};')
        for k in range(N):
            lines.append(f'output [{widths[k]-1}:0] y{k};')
        lines.append('')
        for k in range(N):
            lines.append(f'assign y{k} = {_coeff_expr(M, biases, k, N)};')
    else:
        lines.append(f'module {module}({", ".join(in_ports + ["y"])});')
        for n in range(N):
            lines.append(f'input [{in_bits-1}:0] x{n};')
        lines.append(f'output [{widths[coeff]-1}:0] y;')
        lines.append('')
        lines.append(f'assign y = {_coeff_expr(M, biases, coeff, N)};')
    lines.append('endmodule')
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--points', type=int, default=4, help='DCT size N (default 4)')
    p.add_argument('--in-bits', type=int, default=4, help='input bit width W (default 4)')
    p.add_argument('--scale', type=int, default=None,
                   help='integer scale factor (default: 2^(in-bits) so coeffs are well resolved)')
    p.add_argument('--module', default=None, help='Verilog module name (default: top)')
    p.add_argument('--coeff', type=int, default=None,
                   help='emit only this output coefficient (single-output circuit)')
    p.add_argument('--out', required=True, help='output Verilog path')
    args = p.parse_args()

    N = args.points
    W = args.in_bits
    scale = args.scale if args.scale is not None else (1 << W)
    module = args.module or 'top'
    in_max = (1 << W) - 1

    M = dct_matrix(N, scale)
    biases, widths = output_bias_and_width(M, in_max)
    verilog = emit_verilog(M, biases, widths, W, module, coeff=args.coeff)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(verilog)

    sidecar = {
        'points': N,
        'in_bits': W,
        'scale': scale,
        'module': module,
        'matrix': M.tolist(),
        'bias': biases,
        'out_widths': widths,
        'in_max': in_max,
        'coeff': args.coeff,
    }
    side_path = os.path.splitext(args.out)[0] + '.json'
    with open(side_path, 'w') as f:
        json.dump(sidecar, f, indent=2)

    total_in = N * W
    print(f'Wrote {args.out}  (N={N}, in_bits={W}, scale={scale})')
    print(f'  total input bits: {total_in}')
    print(f'  output widths: {widths}')
    print(f'  sidecar: {side_path}')


if __name__ == '__main__':
    main()
