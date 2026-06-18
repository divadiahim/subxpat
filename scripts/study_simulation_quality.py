"""Compare exact vs simulation labelling end-to-end: area quality and runtime.

Runs the full SubXPAT explore loop (main.py) twice per benchmark -- once with
exact SMT labelling, once with the hybrid simulation labelling -- and reports
the best approximate-circuit area (quality) and total runtime (speed), plus the
labelling time and how many SMT calls the simulation path avoided.

Usage (from repo root, venv active; needs yosys on PATH and a `sta` stub):
    PATH="/tmp/stabin:$PWD/.venv/bin:$PATH" \
        .venv/bin/python scripts/study_simulation_quality.py \
        --benchmarks adder_i8_o5 mul_i6_o6 --max-error 8
"""

import argparse
import os
import re
import subprocess
import sys
import time

AREA_ROW = re.compile(r'gen_\S+\s+([0-9.]+)')
LABTIME = re.compile(r'labelling_time = ([0-9.]+)')
SIMLINE = re.compile(r'\[simulation\].*smt_fallback=(\d+)')
EXACT_AREA = re.compile(r'Exact\s+([0-9.]+)')


def run_one(benchmark, max_error, method, extra_env, args):
    env = dict(os.environ)
    env['SXPAT_LABELING_METHOD'] = method
    env.update(extra_env)
    cmd = [
        sys.executable, 'main.py', f'input/ver/{benchmark}.v',
        '--subxpat', '--max-error', str(max_error),
        '--mode', '3', '--min-subgraph-size', str(args.min_subgraph_size),
        '--max-lpp', str(args.max_lpp), '--max-ppo', str(args.max_ppo),
        '--parallel',
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    wall = time.perf_counter() - t0
    out = proc.stdout
    if proc.returncode != 0:
        print(out[-2000:])
        raise RuntimeError(f'{benchmark} [{method}] FAILED')

    areas = [float(m) for m in AREA_ROW.findall(out)]
    best_area = min(areas) if areas else float('nan')
    exact_area = float(EXACT_AREA.search(out).group(1)) if EXACT_AREA.search(out) else float('nan')
    lab_time = sum(float(x) for x in LABTIME.findall(out))
    smt_fallback = sum(int(x) for x in SIMLINE.findall(out))
    return {
        'best_area': best_area,
        'exact_area': exact_area,
        'labelling_time': lab_time,
        'wall': wall,
        'smt_fallback': smt_fallback,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmarks', nargs='+', required=True)
    p.add_argument('--max-error', type=int, required=True)
    p.add_argument('--sim-samples', type=int, default=1024)
    p.add_argument('--min-subgraph-size', type=int, default=5)
    p.add_argument('--max-lpp', type=int, default=2)
    p.add_argument('--max-ppo', type=int, default=2)
    args = p.parse_args()

    print(f"{'benchmark':16}{'orig':>8}{'exact_ar':>9}{'sim_ar':>8}"
          f"{'dArea%':>8}{'exact_s':>9}{'sim_s':>8}{'speedup':>8}{'simSMT':>7}")
    print('-' * 88)
    for bm in args.benchmarks:
        ex = run_one(bm, args.max_error, 'exact', {}, args)
        sm = run_one(bm, args.max_error, 'simulation',
                     {'SXPAT_SIM_SAMPLES': str(args.sim_samples)}, args)
        darea = (sm['best_area'] - ex['best_area']) / ex['best_area'] * 100 if ex['best_area'] else float('nan')
        speedup = ex['wall'] / sm['wall'] if sm['wall'] else float('nan')
        print(f"{bm:16}{ex['exact_area']:>8.1f}{ex['best_area']:>9.2f}{sm['best_area']:>8.2f}"
              f"{darea:>+8.1f}{ex['wall']:>9.1f}{sm['wall']:>8.1f}{speedup:>7.2f}x{sm['smt_fallback']:>7}")


if __name__ == '__main__':
    main()
