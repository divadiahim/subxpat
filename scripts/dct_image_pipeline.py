"""WP6: image-quality pipeline for approximate DCT circuits.

Demonstrates the end-to-end effect of SubXPAT-approximated DCT coefficients on
image quality. Because the SubXPAT-tractable circuit takes 4 four-bit inputs,
the pipeline uses a 1-D 4-point transform-coding scheme:

  * the image is quantised to 4 bits/pixel,
  * each horizontal run of 4 pixels is forward-transformed (exact or approximate
    DCT coefficients), then inverse-transformed (exact, float) and reconstructed,
  * PSNR and SSIM are measured against the 4-bit original.

The exact forward+inverse is a perfect round trip, so all reconstruction error
comes from the approximate forward transform. Sweeping the error threshold (ET)
of the approximate circuits gives the image-quality-vs-ET trade-off, paired with
the area-vs-ET data from WP5 to form the area--quality Pareto.

Usage (from repo root):
    python scripts/dct_image_pipeline.py --approx-dir output/dct_approx \
        --ets 16 64 --out output/report/dct_image_quality.csv
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
import tempfile
from os.path import join as path_join

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import networkx as nx
from Z3Log_patched.verilog import Verilog
from Z3Log_patched.graph import Graph
from Z3Log_patched.utils import convert_verilog_to_gv
from sxpat.labeling import _eval_node, _output_integers

SIDE = 'input/ver/dct_p4_b4.json'
IMG_SIZE = 64
BITS = 4
LEVELS = (1 << BITS) - 1


# ---------------------------------------------------------------- circuit LUT
def build_coeff_lut(verilog_path: str) -> np.ndarray:
    """Return array L[code] = circuit output for every 16-bit input code."""
    tmp = tempfile.mkdtemp()
    v = Verilog(verilog_path, tv := path_join(tmp, 'c.v'), tmp); v.export_circuit()
    convert_verilog_to_gv(tv, gv := path_join(tmp, 'c.gv'), tmp)
    g = Graph(gv); g.export_graph()
    total_in = g.num_inputs
    S = 1 << total_in
    codes = np.arange(S, dtype=np.int64)
    val = {'__ones__': np.ones(S, dtype=bool), '__zeros__': np.zeros(S, dtype=bool)}
    for i in range(total_in):
        val[f'in{i}'] = ((codes >> i) & 1).astype(bool)
    for node in nx.topological_sort(g.graph):
        if node not in val:
            val[node] = _eval_node(g, node, val)
    return _output_integers(g, val)  # int per code


def codes_from_blocks(blocks: np.ndarray) -> np.ndarray:
    """blocks: (B,4) four-bit pixels -> 16-bit codes (bus-major, LSB-first)."""
    code = np.zeros(blocks.shape[0], dtype=np.int64)
    for k in range(4):
        code |= (blocks[:, k].astype(np.int64) & 0xF) << (4 * k)
    return code


# ---------------------------------------------------------------- metrics
def psnr(a: np.ndarray, b: np.ndarray, data_range: float) -> float:
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(data_range ** 2 / mse)


def ssim(a: np.ndarray, b: np.ndarray, data_range: float, win: int = 7) -> float:
    a = a.astype(float); b = b.astype(float)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    from numpy.lib.stride_tricks import sliding_window_view
    aw = sliding_window_view(a, (win, win))
    bw = sliding_window_view(b, (win, win))
    mu_a = aw.mean((-1, -2)); mu_b = bw.mean((-1, -2))
    va = aw.var((-1, -2)); vb = bw.var((-1, -2))
    cov = (aw * bw).mean((-1, -2)) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * cov + C2)) / \
        ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2))
    return float(s.mean())


# ---------------------------------------------------------------- pipeline
def load_images():
    """High-contrast structured test patterns (full 0--255 range) that exercise
    the transform and make the approximation artefacts clearly visible."""
    imgs = {}
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    cx = cy = IMG_SIZE / 2

    # concentric rings: smooth radial structure, full contrast
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    imgs['circles'] = (127.5 * (1 + np.cos(r / 3.0))).astype(np.uint8)

    # checkerboard: sharp edges (strong high-frequency content)
    imgs['checker'] = (((xx // 8 + yy // 8) % 2) * 255).astype(np.uint8)

    # gaussian blobs: localised smooth features, full contrast
    blobs = np.zeros((IMG_SIZE, IMG_SIZE))
    for bx, by, s in [(16, 16, 8), (46, 22, 11), (30, 48, 10)]:
        blobs += np.exp(-(((xx - bx) ** 2 + (yy - by) ** 2) / (2 * s ** 2)))
    blobs = blobs / blobs.max() * 255
    imgs['blobs'] = blobs.astype(np.uint8)
    return imgs


def quantize(img: np.ndarray) -> np.ndarray:
    return np.round(img.astype(float) / 255 * LEVELS).astype(np.int64)


def reconstruct(img4: np.ndarray, M, Minv, bias, luts):
    """1-D 4-point transform coding over horizontal 4-pixel runs.

    luts=None -> exact transform (coeff = M@x); else approximate via circuit LUTs
    (coeff = lut[code] - bias)."""
    H, W = img4.shape
    Wc = (W // 4) * 4
    out = img4.copy().astype(float)
    blocks = img4[:, :Wc].reshape(H, Wc // 4, 4).reshape(-1, 4)  # (B,4)
    if luts is None:
        coeffs = blocks @ M.T  # exact M@x, (B,4)
    else:
        codes = codes_from_blocks(blocks)
        coeffs = np.stack([luts[k][codes] - bias[k] for k in range(4)], axis=1)
    rec = coeffs @ Minv.T  # inverse: x = Minv @ coeff
    rec = np.clip(np.round(rec), 0, LEVELS)
    out[:, :Wc] = rec.reshape(H, Wc // 4, 4).reshape(H, Wc)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--approx-dir', default='output/dct_approx')
    p.add_argument('--ets', nargs='+', type=int, required=True)
    p.add_argument('--out', default='output/report/dct_image_quality.csv')
    p.add_argument('--save-images', action='store_true')
    args = p.parse_args()

    with open(SIDE) as f:
        side = json.load(f)
    M = np.array(side['matrix'], dtype=np.int64)
    bias = side['bias']
    Minv = np.linalg.inv(M.astype(float))

    images = {name: quantize(img) for name, img in load_images().items()}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []

    # exact baseline (perfect round trip -> sanity check)
    for name, img4 in images.items():
        rec = reconstruct(img4, M, Minv, bias, luts=None)
        rows.append({'image': name, 'et': 'exact',
                     'psnr': round(psnr(img4, rec, LEVELS), 3),
                     'ssim': round(ssim(img4, rec, LEVELS), 4)})
        if args.save_images:
            d = 'output/dct_images'; os.makedirs(d, exist_ok=True)
            Image.fromarray((img4 / LEVELS * 255).astype(np.uint8)).save(
                os.path.join(d, f'{name}_orig.png'))

    # approximate circuits per ET
    for et in args.ets:
        luts = {}
        ok = True
        for k in range(4):
            vpath = os.path.join(args.approx_dir, f'dct4c{k}_i16_o9_et{et}.v')
            if not os.path.exists(vpath):
                print(f'  missing {vpath}, skipping et={et}')
                ok = False
                break
            luts[k] = build_coeff_lut(vpath)
        if not ok:
            continue
        for name, img4 in images.items():
            rec = reconstruct(img4, M, Minv, bias, luts=luts)
            rows.append({'image': name, 'et': et,
                         'psnr': round(psnr(img4, rec, LEVELS), 3),
                         'ssim': round(ssim(img4, rec, LEVELS), 4)})
            if args.save_images:
                d = 'output/dct_images'; os.makedirs(d, exist_ok=True)
                Image.fromarray((rec / LEVELS * 255).astype(np.uint8)).save(
                    os.path.join(d, f'{name}_et{et}.png'))

    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['image', 'et', 'psnr', 'ssim'])
        w.writeheader(); w.writerows(rows)

    print(f"{'image':10}{'et':>8}{'PSNR(dB)':>10}{'SSIM':>8}")
    print('-' * 36)
    for r in rows:
        print(f"{r['image']:10}{str(r['et']):>8}{r['psnr']:>10}{r['ssim']:>8}")
    print(f'\nSaved {args.out}')


if __name__ == '__main__':
    main()
