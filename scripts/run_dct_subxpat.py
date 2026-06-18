"""WP5: run (improved) SubXPAT on DCT coefficient circuits across error thresholds.

For each benchmark and ET, runs the full SubXPAT explore loop and records the
exact circuit area and the best approximate-circuit area, giving the area-vs-ET
trade-off used in the evaluation. Uses the improved parallel labelling.

Requires yosys on PATH and a `sta` stub (area comes from yosys; power/delay are
not needed here). Run from repo root, e.g.:

    PATH="/tmp/stabin:$PWD/.venv/bin:$PATH" .venv/bin/python \
        scripts/run_dct_subxpat.py --benchmarks dct4c0_i16_o9 dct4c1_i16_o9 \
        --ets 4 16 64 --out output/report/dct_area.csv
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time

AREA_ROW = re.compile(r'gen_\S+\s+([0-9.]+)')
EXACT_AREA = re.compile(r'Exact\s+([0-9.]+)')
RUNDIR = re.compile(r'directory:\s+(\S+)')
# pareto table rows: "... output/<id>/verilog/gen_xxx.v │ <area> │ ..."
PARETO_ROW = re.compile(r'(\S+/verilog/gen_\S+\.v)\s*[│|]\s*([0-9.]+)')


def run_one(benchmark, et, args, save_dir):
    cmd = [
        sys.executable, 'main.py', f'input/ver/{benchmark}.v',
        '--subxpat', '--max-error', str(et),
        '--mode', '3', '--min-subgraph-size', str(args.min_subgraph_size),
        '--max-lpp', str(args.max_lpp), '--max-ppo', str(args.max_ppo),
        '--parallel', '--timeout', str(args.cell_timeout),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    wall = time.perf_counter() - t0
    out = proc.stdout
    if proc.returncode != 0:
        print(out[-1500:])
        raise RuntimeError(f'{benchmark} et={et} FAILED')
    areas = [float(m) for m in AREA_ROW.findall(out)]
    best = min(areas) if areas else float('nan')
    exact = float(EXACT_AREA.search(out).group(1)) if EXACT_AREA.search(out) else float('nan')

    # save the best (min-area) approximate circuit for the image pipeline (WP6)
    saved = ''
    pareto = [(path, float(a)) for path, a in PARETO_ROW.findall(out)]
    if pareto:
        best_path = min(pareto, key=lambda t: t[1])[0]
        if os.path.isfile(best_path):
            os.makedirs(save_dir, exist_ok=True)
            saved = os.path.join(save_dir, f'{benchmark}_et{et}.v')
            shutil.copyfile(best_path, saved)
    return exact, best, wall, saved


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--benchmarks', nargs='+', required=True)
    p.add_argument('--ets', nargs='+', type=int, required=True)
    p.add_argument('--min-subgraph-size', type=int, default=5)
    p.add_argument('--max-lpp', type=int, default=3)
    p.add_argument('--max-ppo', type=int, default=3)
    p.add_argument('--out', default='output/report/dct_area.csv')
    p.add_argument('--approx-dir', default='output/dct_approx',
                   help='where to save the best approximate circuit per run')
    p.add_argument('--cell-timeout', type=float, default=120,
                   help='per-cell SMT timeout in seconds (default 120)')
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fields = ['benchmark', 'et', 'exact_area', 'approx_area',
              'area_reduction_pct', 'wall_s', 'approx_verilog']
    f = open(args.out, 'w', newline='')
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); f.flush()

    print(f"{'benchmark':16}{'et':>6}{'exact':>9}{'approx':>9}{'reduc%':>8}{'wall_s':>8}")
    print('-' * 56)
    for bm in args.benchmarks:
        for et in args.ets:
            exact, best, wall, saved = run_one(bm, et, args, args.approx_dir)
            reduc = (exact - best) / exact * 100 if exact else float('nan')
            row = {'benchmark': bm, 'et': et, 'exact_area': round(exact, 4),
                   'approx_area': round(best, 4), 'area_reduction_pct': round(reduc, 2),
                   'wall_s': round(wall, 2), 'approx_verilog': saved}
            w.writerow(row); f.flush()
            print(f"{bm:16}{et:>6}{exact:>9.2f}{best:>9.2f}{reduc:>+8.1f}{wall:>8.1f}")
    f.close()
    print(f'\nSaved {args.out}')


if __name__ == '__main__':
    main()
