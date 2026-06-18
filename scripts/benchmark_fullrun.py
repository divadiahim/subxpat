"""Full-run SubXPAT benchmark with per-stage Z3-call instrumentation.

Runs the complete SubXPAT explore loop for one benchmark and error threshold
under a chosen labelling mode, and records how many Z3 solver calls each
execution stage makes and how long each takes:

  * labelling    - mnze (or alternative) calls, one per labelled gate
  * rewriting    - the per-cell subcircuit-approximation solves
  * verification - the ErrorEval worst-case-distance solves

It also records the total run time and the best approximate-circuit area, so the
effect of each labelling optimisation on the *complete* run (calls, time, and
area/correctness) can be compared. One labelling mode per invocation, so the
optimisations are measured independently.

Labelling modes (``--method`` / SXPAT_LABELING_METHOD):
  exact | prefilter | simulation | warmstart | wce
(see sxpat/xplore.py:label_graph for the meaning of each.)

Requires yosys on PATH and a `sta` stub (area comes from yosys). Run from repo
root, e.g.:

    PATH="/tmp/stabin:$PWD/.venv/bin:$PATH" .venv/bin/python \
        scripts/benchmark_fullrun.py --benchmarks adder_i8_o5 --ets 8 \
        --method prefilter --out output/report/fullrun.csv
"""

import argparse
import contextlib
import csv
import io
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

AREA_ROW = re.compile(r'gen_\S+\s+([0-9.]+)')
EXACT_AREA = re.compile(r'Exact\s+([0-9.]+)')

# ---------------------------------------------------------------------------
# Per-stage Z3 instrumentation
# ---------------------------------------------------------------------------
_STAGE = 'rewriting'                       # default stage for _run_script calls
_COUNTERS = {}                             # stage -> [calls, time_s]


def _reset_counters():
    global _COUNTERS, _STAGE
    _COUNTERS = {s: [0, 0.0] for s in ('labelling', 'rewriting', 'verification')}
    _STAGE = 'rewriting'


def _install_instrumentation():
    import Z3Log_patched.z3solver as zmod
    import sxpat.solvers.Z3Solver as smod
    import sxpat.xplore as xmod

    # labelling: run_z3pyscript_labeling runs one subprocess per gate script
    orig_lbl = zmod.Z3solver.run_z3pyscript_labeling

    def timed_labelling(self):
        n = len(self.pyscript_files_for_labeling)
        t0 = time.perf_counter()
        try:
            return orig_lbl(self)
        finally:
            _COUNTERS['labelling'][0] += n
            _COUNTERS['labelling'][1] += time.perf_counter() - t0
    zmod.Z3solver.run_z3pyscript_labeling = timed_labelling

    # rewriting + verification: every cell/eval solve goes through _run_script
    orig_run = smod.Z3Solver._run_script.__func__

    def timed_run(cls, script_path):
        t0 = time.perf_counter()
        try:
            return orig_run(cls, script_path)
        finally:
            _COUNTERS[_STAGE][0] += 1
            _COUNTERS[_STAGE][1] += time.perf_counter() - t0
    smod.Z3Solver._run_script = classmethod(timed_run)

    # verification: wrap error_evaluation so its _run_script calls are tagged
    orig_eval = xmod.error_evaluation

    def staged_eval(*a, **k):
        global _STAGE
        prev = _STAGE
        _STAGE = 'verification'
        try:
            return orig_eval(*a, **k)
        finally:
            _STAGE = prev
    xmod.error_evaluation = staged_eval


# ---------------------------------------------------------------------------
# Run one full SubXPAT exploration in-process
# ---------------------------------------------------------------------------
def run_one(benchmark, et, args):
    from sxpat.specifications import Specifications
    from sxpat.utils.filesystem import FS
    from sxpat.utils.storage import LiveStorage, AppendStorage
    from sxpat.xplore import explore_grid

    sys.argv = [
        'main.py', f'input/ver/{benchmark}.v',
        '--subxpat', '--max-error', str(et),
        '--mode', '3', '--min-subgraph-size', str(args.min_subgraph_size),
        '--max-lpp', str(args.max_lpp), '--max-ppo', str(args.max_ppo),
        '--parallel', '--timeout', str(args.cell_timeout),
    ]
    specs = Specifications.parse_args()
    for d in specs.path.run.folders:
        FS.mkdir(d)
    specs.stats_storage = LiveStorage(specs.path.run.run_stats)
    specs.details_storage = AppendStorage(specs.path.run.run_details)

    _reset_counters()
    buf = io.StringIO()
    t0 = time.perf_counter()
    with specs.stats_storage, specs.details_storage:
        specs.details_storage.add(specs.constant_fields)
        with contextlib.redirect_stdout(buf):
            explore_grid(specs)
    total = time.perf_counter() - t0

    if not specs.debug:
        FS.rmdir(specs.path.run.temporary, True)

    out = buf.getvalue()
    areas = [float(m) for m in AREA_ROW.findall(out)]
    approx = min(areas) if areas else float('nan')
    exact = float(EXACT_AREA.search(out).group(1)) if EXACT_AREA.search(out) else float('nan')
    return total, exact, approx


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--benchmarks', nargs='+', required=True)
    p.add_argument('--ets', nargs='+', type=int, required=True)
    p.add_argument('--method', default='exact',
                   choices=['full', 'exact', 'prefilter', 'simulation', 'warmstart', 'wce'])
    p.add_argument('--sim-samples', type=int, default=1024)
    p.add_argument('--min-subgraph-size', type=int, default=5)
    p.add_argument('--max-lpp', type=int, default=2)
    p.add_argument('--max-ppo', type=int, default=2)
    p.add_argument('--cell-timeout', type=float, default=120)
    p.add_argument('--out', default='output/report/fullrun.csv')
    args = p.parse_args()

    os.environ['SXPAT_LABELING_METHOD'] = args.method
    os.environ['SXPAT_SIM_SAMPLES'] = str(args.sim_samples)
    _install_instrumentation()

    fields = ['benchmark', 'et', 'method',
              'labelling_calls', 'labelling_s',
              'rewriting_calls', 'rewriting_s',
              'verification_calls', 'verification_s',
              'total_z3_calls', 'total_s', 'exact_area', 'approx_area']
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    f = open(args.out, 'w', newline='')
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); f.flush()

    print(f"{'benchmark':16}{'et':>5}{'method':>11}{'lblC':>7}{'rwC':>6}{'vrfC':>6}"
          f"{'totC':>7}{'time':>8}{'area':>9}")
    print('-' * 81)
    for bm in args.benchmarks:
        for et in args.ets:
            total, exact, approx = run_one(bm, et, args)
            lbl, rw, vr = (_COUNTERS['labelling'], _COUNTERS['rewriting'],
                           _COUNTERS['verification'])
            totc = lbl[0] + rw[0] + vr[0]
            row = {'benchmark': bm, 'et': et, 'method': args.method,
                   'labelling_calls': lbl[0], 'labelling_s': round(lbl[1], 3),
                   'rewriting_calls': rw[0], 'rewriting_s': round(rw[1], 3),
                   'verification_calls': vr[0], 'verification_s': round(vr[1], 3),
                   'total_z3_calls': totc, 'total_s': round(total, 2),
                   'exact_area': round(exact, 4), 'approx_area': round(approx, 4)}
            w.writerow(row); f.flush()
            print(f"{bm:16}{et:>5}{args.method:>11}{lbl[0]:>7}{rw[0]:>6}{vr[0]:>6}"
                  f"{totc:>7}{total:>8.1f}{approx:>9.2f}")
    f.close()
    print(f'\nSaved {args.out}')


if __name__ == '__main__':
    main()
