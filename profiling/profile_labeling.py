"""Comprehensive profiler for the SubXPAT gate-labelling pipeline.

Produces two complementary outputs for each benchmark run:

1. Phase breakdown table (stdout + CSV)
   - setup   : Verilog export, GV conversion, Graph construction, Z3solver init
   - gen     : create_pruned_z3pyscript_approximate() for all gates
   - exec    : run_z3pyscript_labeling() — subprocess Z3 calls (the hot path)
   - import  : import_labels() — reading result CSVs
   - overhead: everything else inside labeling_explicit()

2. cProfile flat-list report (stdout + .prof file)
   Sorted by cumulative time so you see the full call stack cost.
   Load the .prof file in snakeviz for an interactive flame graph:
       pip install snakeviz
       snakeviz profiling/results/<name>.prof

Usage (from repo root, with venv active):
    python profiling/profile_labeling.py
    python profiling/profile_labeling.py --benchmarks adder_i4_o3 adder_i8_o5
    python profiling/profile_labeling.py --benchmarks adder_i8_o5 --parallel
    python profiling/profile_labeling.py --top 30 --no-cprofile
"""

import argparse
import contextlib
import cProfile
import csv
import io
import os
import pstats
import sys
import time
import tempfile
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import Z3Log_patched.z3solver as _patched_module
from Z3Log_patched.z3solver import Z3solver
from Z3Log_patched.verilog import Verilog
from Z3Log_patched.graph import Graph
from Z3Log_patched.utils import convert_verilog_to_gv
from Z3Log_patched.config.config import SINGLE, MAXIMIZE
from sxpat.specifications import Paths


# ---------------------------------------------------------------------------
# Phase-level instrumentation via monkey-patching
# ---------------------------------------------------------------------------

class _PhaseTimer:
    """Accumulates wall-clock time for named phases."""

    def __init__(self):
        self._totals: Dict[str, float] = {}
        self._starts: Dict[str, float] = {}

    @contextlib.contextmanager
    def measure(self, phase: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._totals[phase] = self._totals.get(phase, 0.0) + (time.perf_counter() - t0)

    def get(self, phase: str) -> float:
        return self._totals.get(phase, 0.0)

    def phases(self):
        return dict(self._totals)


def _install_patches(timer: _PhaseTimer):
    """Monkey-patch Z3solver methods to record time per phase."""

    original_gen = Z3solver.create_pruned_z3pyscript_approximate
    original_exec = Z3solver.run_z3pyscript_labeling
    original_import = Z3solver.import_labels

    def _timed_gen(self, gates, constant_value=False):
        with timer.measure('gen'):
            return original_gen(self, gates, constant_value)

    def _timed_exec(self):
        with timer.measure('exec'):
            return original_exec(self)

    def _timed_import(self, constant_value=False):
        with timer.measure('import'):
            return original_import(self, constant_value)

    Z3solver.create_pruned_z3pyscript_approximate = _timed_gen
    Z3solver.run_z3pyscript_labeling = _timed_exec
    Z3solver.import_labels = _timed_import

    return original_gen, original_exec, original_import


def _remove_patches(originals):
    original_gen, original_exec, original_import = originals
    Z3solver.create_pruned_z3pyscript_approximate = original_gen
    Z3solver.run_z3pyscript_labeling = original_exec
    Z3solver.import_labels = original_import


# ---------------------------------------------------------------------------
# Core profiling routine
# ---------------------------------------------------------------------------

def profile_one(benchmark: str, parallel: bool, run_cprofile: bool) -> Tuple[dict, str]:
    """Profile a single benchmark; return (phase_times, cprofile_report)."""

    verilog_path = os.path.join('input', 'ver', f'{benchmark}.v')
    run_paths = Paths.RunFiles(f'profile_{benchmark}', 'output')
    os.makedirs(run_paths.temporary, exist_ok=True)

    timer = _PhaseTimer()
    originals = _install_patches(timer)

    profiler = cProfile.Profile() if run_cprofile else None

    try:
        total_t0 = time.perf_counter()

        if profiler:
            profiler.enable()

        # ---- phase: setup ------------------------------------------------
        with timer.measure('setup'):
            verilog_obj_exact = Verilog(
                verilog_path,
                tmp_exact_v := os.path.join(run_paths.temporary, 'lbl_exact.v'),
                run_paths.temporary,
            )
            verilog_obj_exact.export_circuit()
            verilog_obj_approx = Verilog(
                verilog_path,
                tmp_current_v := os.path.join(run_paths.temporary, 'lbl_current.v'),
                run_paths.temporary,
            )
            verilog_obj_approx.export_circuit()

            convert_verilog_to_gv(
                tmp_exact_v,
                tmp_exact_gv := os.path.join(run_paths.temporary, 'lbl_exact.gv'),
                run_paths.temporary,
            )
            convert_verilog_to_gv(
                tmp_current_v,
                tmp_current_gv := os.path.join(run_paths.temporary, 'lbl_current.gv'),
                run_paths.temporary,
            )

            graph_obj_exact = Graph(tmp_exact_gv)
            graph_obj_current = Graph(tmp_current_gv)
            graph_obj_exact.export_graph()
            graph_obj_current.export_graph()

            z3py_obj = Z3solver(
                tmp_exact_gv, tmp_current_gv, run_paths.temporary,
                experiment=SINGLE, optimization=MAXIMIZE, style='max',
                partial=False, parallel=parallel,
            )

        # ---- phases: gen + exec + import (inside label_circuit) ----------
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
            z3py_obj.label_circuit(False, partial=False)

        total_elapsed = time.perf_counter() - total_t0

        if profiler:
            profiler.disable()

    finally:
        _remove_patches(originals)

    # Build phase dict; compute overhead as remainder
    phases = timer.phases()
    accounted = sum(phases.values())
    phases['overhead'] = max(0.0, total_elapsed - accounted)
    phases['total'] = total_elapsed

    # cProfile report
    cprofile_report = ''
    if profiler:
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats('cumulative')
        stats.print_stats(40)
        cprofile_report = stream.getvalue()

    return phases, cprofile_report


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

PHASE_ORDER = ['setup', 'gen', 'exec', 'import', 'overhead', 'total']
PHASE_LABELS = {
    'setup':    'Setup (verilog/graph/z3init)',
    'gen':      'Script generation',
    'exec':     'Subprocess execution (Z3)',
    'import':   'Label import (CSV read)',
    'overhead': 'Other overhead',
    'total':    'TOTAL',
}


def _fmt_phase_table(benchmark: str, phases: dict, parallel: bool) -> str:
    mode = 'parallel' if parallel else 'sequential'
    lines = [
        '',
        f'  Benchmark : {benchmark}  [{mode}]',
        f'  {"Phase":<35} {"Time (s)":>9}  {"% total":>8}',
        f'  {"-"*55}',
    ]
    total = phases.get('total', 1e-9)
    for key in PHASE_ORDER:
        v = phases.get(key, 0.0)
        pct = 100.0 * v / total if total > 0 else 0.0
        sep = '  ' + '-' * 55 if key == 'total' else ''
        if sep:
            lines.append(sep)
        lines.append(f'  {PHASE_LABELS[key]:<35} {v:>9.3f}s  {pct:>7.1f}%')
    return '\n'.join(lines)


CSV_HEADER = ['benchmark', 'mode', 'setup_s', 'gen_s', 'exec_s',
              'import_s', 'overhead_s', 'total_s',
              'exec_pct', 'gen_pct']


def _to_csv_row(benchmark: str, phases: dict, parallel: bool) -> dict:
    total = phases.get('total', 1e-9)
    return {
        'benchmark': benchmark,
        'mode': 'parallel' if parallel else 'sequential',
        'setup_s':    round(phases.get('setup', 0), 4),
        'gen_s':      round(phases.get('gen', 0), 4),
        'exec_s':     round(phases.get('exec', 0), 4),
        'import_s':   round(phases.get('import', 0), 4),
        'overhead_s': round(phases.get('overhead', 0), 4),
        'total_s':    round(total, 4),
        'exec_pct':   round(100.0 * phases.get('exec', 0) / total, 1),
        'gen_pct':    round(100.0 * phases.get('gen', 0) / total, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_BENCHMARKS = [
    'adder_i4_o3',
    'adder_i6_o4',
    'adder_i8_o5',
]
RESULTS_DIR = os.path.join('profiling', 'results')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--benchmarks', nargs='+', default=DEFAULT_BENCHMARKS,
                        metavar='BENCHMARK')
    parser.add_argument('--parallel', action='store_true',
                        help='Profile the parallel execution path')
    parser.add_argument('--both', action='store_true',
                        help='Profile both sequential and parallel, side by side')
    parser.add_argument('--top', type=int, default=25,
                        help='Number of top functions in cProfile report (default 25)')
    parser.add_argument('--no-cprofile', action='store_true',
                        help='Skip cProfile (faster; only phase breakdown)')
    parser.add_argument('--out-dir', default=RESULTS_DIR,
                        help=f'Output directory (default: {RESULTS_DIR})')
    args = parser.parse_args()

    results_dir = args.out_dir
    os.makedirs(results_dir, exist_ok=True)

    modes = []
    if args.both:
        modes = [False, True]
    else:
        modes = [args.parallel]

    all_rows = []
    csv_path = os.path.join(results_dir, 'phase_breakdown.csv')

    for benchmark in args.benchmarks:
        for parallel in modes:
            mode_tag = 'par' if parallel else 'seq'
            print(f'\n{"="*60}')
            print(f'  Profiling: {benchmark}  [{"parallel" if parallel else "sequential"}]')
            print(f'{"="*60}')

            phases, cprofile_report = profile_one(
                benchmark, parallel, run_cprofile=not args.no_cprofile
            )

            print(_fmt_phase_table(benchmark, phases, parallel))

            if cprofile_report:
                print(f'\n--- cProfile top-{args.top} (cumulative time) ---')
                # trim to top N lines of the stats table
                stat_lines = cprofile_report.splitlines()
                header_idx = next(
                    (i for i, l in enumerate(stat_lines) if 'cumtime' in l), 0
                )
                trimmed = '\n'.join(stat_lines[:header_idx + 1 + args.top + 5])
                print(trimmed)

                txt_path = os.path.join(results_dir, f'{benchmark}_{mode_tag}_cprofile.txt')
                with open(txt_path, 'w') as f:
                    f.write(cprofile_report)
                print(f'\n  Full cProfile report saved: {txt_path}')

            all_rows.append(_to_csv_row(benchmark, phases, parallel))

    # Write CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'\nPhase breakdown CSV saved: {csv_path}')

    # Summary comparison table when --both is used
    if args.both and len(args.benchmarks) > 0:
        print(f'\n{"="*70}')
        print(f'  Sequential vs Parallel comparison')
        print(f'  {"Benchmark":<20} {"exec_seq":>10} {"exec_par":>10} {"speedup":>9}')
        print(f'  {"-"*52}')
        rows_by_bm = {}
        for row in all_rows:
            rows_by_bm.setdefault(row['benchmark'], {})[row['mode']] = row
        for bm, modes_d in rows_by_bm.items():
            s = modes_d.get('sequential', {})
            p = modes_d.get('parallel', {})
            if s and p:
                speedup = s['exec_s'] / p['exec_s'] if p['exec_s'] > 0 else float('inf')
                print(f"  {bm:<20} {s['exec_s']:>10.3f}s {p['exec_s']:>10.3f}s {speedup:>8.2f}x")


if __name__ == '__main__':
    main()
