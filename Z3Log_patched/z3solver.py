from typing import Dict

import re
import os
import sys
import copy
import csv
import concurrent.futures
from os.path import join as path_join
from subprocess import PIPE, Popen

import networkx as nx

from .utils import get_pure_name
from .graph import Graph

from .config.config import (
    SINGLE, WAE, RANDOM, WRE, OPTIMIZE, MAXIMIZE, BISECTION,
    WHD, WCE, DEFAULT_STRATEGY, QOR, MONOTONIC,
)

# patched:
from Z3Log.z3solver import Z3solver as _Z3solver

__all__ = ['Z3solver']


class Z3solver(_Z3solver):
    def __init__(self,
                 circuit_in_graphviz_path: str,
                 approx_circuit_in_graphviz_path: str,
                 temporary_folder_path: str,
                 #
                 samples: list = [],
                 experiment: str = SINGLE,
                 pruned_percentage: int = None, pruned_gates=None, metric: str = WAE, precision: int = 4,
                 optimization: str = None, style: str = 'max',
                 parallel: bool = False, partial: bool = True,
                 prefilter: bool = False):
        """

        :param benchmark_name: the input benchmark in gv format
        :param approximate_benchmark_name: the approximate benchmark in gv format
        :param samples: number of samples for the mc evaluation; by defaults it is an empty list
        :param prefilter: if True, skip the SMT label call for any gate whose
            least-significant reachable output has weight greater than the error
            threshold (its minimum non-zero error is then guaranteed to exceed ET)
        """
        self.__circuit_name = circuit_name = get_pure_name(circuit_in_graphviz_path)
        self.__graph = Graph(circuit_in_graphviz_path, True)

        self.__pyscript_results_out_path = None

        self.__z3_log_path = path_join(temporary_folder_path, f'{circuit_name}_z3py.log')

        # added
        self.temporary_folder_path = temporary_folder_path

        self.__approximate_graph = None

        if approx_circuit_in_graphviz_path:
            self.__approximate_circuit_name = get_pure_name(approx_circuit_in_graphviz_path)
            self.__approximate_graph = Graph(approx_circuit_in_graphviz_path, True)

            self.relabel_approximate_graph()

            self.approximate_graph.set_input_dict(self.approximate_graph.extract_inputs())
            self.approximate_graph.set_output_dict(self.approximate_graph.extract_outputs())
            self.approximate_graph.set_gate_dict(self.approximate_graph.extract_gates())
            self.approximate_graph.set_constant_dict(self.approximate_graph.extract_constants())

            self.__labeling_graph = copy.deepcopy(self.approximate_graph)

        self.__experiment = experiment
        self.__pruned_percentage = None
        # TODO
        # Later create and internal method that can generate pruned gates
        self.__pruned_gates = None
        if experiment == RANDOM:
            self.__pruned_percentage = pruned_percentage
            self.__pruned_gates = pruned_gates

        self.__metric = metric
        self.__precision = precision

        self.__z3_report = None

        self.__samples = samples
        self.__sample_results = None

        self.__z3string = None

        self.__z3pyscript = None

        self.__strategy = None

        self.__optimization = optimization

        self.__pyscript_files_for_labeling: list = []

        self.__z3_out_path = None

        self.__style = style

        self.__parallel = parallel

        self.__labels: Dict = {}

        self.__partial: bool = partial

        self.__prefilter: bool = prefilter
        self.__prefilter_stats: Dict = {}
        self.__selective_stats: Dict = {}

        # per-gate warm-start upper bound for the minimisation (Strategy 2 /
        # simulation warm-start); None means use the default loose bound.
        self._warm_bound = None

    def set_warm_bound(self, value):
        self._warm_bound = value

    def express_monotonic_while_loop(self):
        loop = super().express_monotonic_while_loop()
        # Inject a tight initial upper bound from simulation: the exact minimum
        # is <= warm_bound, so constraining the search to [0, warm_bound] keeps
        # the optimum unchanged while shrinking the solver's search space.
        if self._warm_bound is not None and self.style == 'min':
            loop = loop.replace(
                '(f_error(exact_out, approx_out)) <= z3_abs(max)',
                f'(f_error(exact_out, approx_out)) <= z3_abs({int(self._warm_bound)})',
            )
        return loop

    @property
    def prefilter(self):
        return self.__prefilter

    def set_prefilter(self, value: bool):
        self.__prefilter = value

    @property
    def prefilter_stats(self):
        return self.__prefilter_stats

    def _compute_min_output_index(self, graph) -> Dict[str, int]:
        """Return {node_name: index of the least-significant primary output
        reachable from that node}.

        The circuit is a DAG with edges flowing input -> gate -> output, so the
        minimum reachable output index of a node is the minimum over its
        successors. Computing this in a single reverse-topological pass costs
        O(|V| + |E|). Nodes that reach no primary output are absent from the
        returned dict.
        """
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

    def _label_circuit_prefilter(self, constant_value: bool, et: int) -> Dict:
        """Structural pre-filter labelling.

        For every gate, look at the least-significant output it can reach. If the
        weight of that output (2**index) already exceeds the error threshold, then
        any perturbation of the gate must change an output of at least that weight,
        so its minimum non-zero error is guaranteed to exceed ET. Such gates are
        infeasible for approximation and are skipped without an SMT call. The
        remaining gates are labelled by the usual SMT optimisation.
        """
        min_out = self._compute_min_output_index(self.labeling_graph)

        skipped = 0
        labeled = 0

        def _consider(gate):
            nonlocal skipped, labeled
            lsb = min_out.get(gate)
            if lsb is not None and 2 ** lsb > et:
                skipped += 1
                return
            self.create_pruned_z3pyscript_approximate([gate], constant_value)
            labeled += 1

        for key in self.labeling_graph.gate_dict:
            _consider(self.labeling_graph.gate_dict[key])
        for key in self.labeling_graph.constant_dict:
            _consider(self.labeling_graph.constant_dict[key])

        self.__prefilter_stats = {
            'et': et,
            'labeled': labeled,
            'skipped': skipped,
            'total': labeled + skipped,
        }
        print(f'[prefilter] et={et} labeled={labeled} skipped={skipped} '
              f'(total={labeled + skipped})')

        if labeled == 0:  # nothing to solve; importing would fail on a missing dir
            return self.labels
        self.run_z3pyscript_labeling()
        self.import_labels(constant_value)
        return self.labels

    def label_circuit(self, constant_value: bool = False, partial: bool = False, et: int = -1):
        if self.prefilter and et != -1:
            self.experiment = SINGLE
            self.set_strategy(MONOTONIC)
            return self._label_circuit_prefilter(constant_value, et)
        return super().label_circuit(constant_value, partial=partial, et=et)

    @property
    def selective_stats(self):
        return self.__selective_stats

    def _compute_needs_relabel(self, graph, modified_gates) -> set:
        """Return the set of nodes whose label may have changed after a rewrite
        that modified ``modified_gates``.

        A gate's label is computed from the miter between that gate and the
        primary outputs. A rewrite of the gate set R can only affect gates whose
        fanout paths overlap the region influenced by R: the transitive fanout
        TFO(R), plus every gate that can reach a node in TFO(R). Both passes are
        plain BFS over the DAG, O(|V| + |E|) total.
        """
        g = graph.graph
        # 1) transitive fanout of the modified gates (including the gates themselves)
        tfo = set()
        stack = list(modified_gates)
        while stack:
            v = stack.pop()
            if v in tfo:
                continue
            tfo.add(v)
            stack.extend(g.successors(v))
        # 2) every node that can reach the affected region: TFI(TFO(R))
        needs = set(tfo)
        stack = list(tfo)
        while stack:
            v = stack.pop()
            for p in g.predecessors(v):
                if p not in needs:
                    needs.add(p)
                    stack.append(p)
        return needs

    def label_circuit_selective(self, modified_gates, constant_value: bool = False,
                                et: int = -1, previous_labels: Dict = None) -> Dict:
        """Fanout-cone selective relabelling.

        Only relabel gates whose miter could have changed after a rewrite of
        ``modified_gates`` (see ``_compute_needs_relabel``). If the prefilter
        toggle is set and ``et`` is given, the output-significance pre-filter is
        applied on top, skipping gates whose minimum non-zero error provably
        exceeds ET. Labels of skipped gates are taken from ``previous_labels``.
        """
        self.experiment = SINGLE
        self.set_strategy(MONOTONIC)

        needs = self._compute_needs_relabel(self.labeling_graph, modified_gates)
        min_out = (self._compute_min_output_index(self.labeling_graph)
                   if (self.prefilter and et != -1) else None)

        relabeled = 0
        skipped_cone = 0
        skipped_prefilter = 0

        def _consider(gate):
            nonlocal relabeled, skipped_cone, skipped_prefilter
            if gate not in needs:
                skipped_cone += 1
                return
            if min_out is not None:
                lsb = min_out.get(gate)
                if lsb is not None and 2 ** lsb > et:
                    skipped_prefilter += 1
                    return
            self.create_pruned_z3pyscript_approximate([gate], constant_value)
            relabeled += 1

        for key in self.labeling_graph.gate_dict:
            _consider(self.labeling_graph.gate_dict[key])
        for key in self.labeling_graph.constant_dict:
            _consider(self.labeling_graph.constant_dict[key])

        self.__selective_stats = {
            'et': et,
            'n_modified': len(modified_gates),
            'relabeled': relabeled,
            'skipped_cone': skipped_cone,
            'skipped_prefilter': skipped_prefilter,
            'total': relabeled + skipped_cone + skipped_prefilter,
        }
        print(f'[selective] modified={len(modified_gates)} et={et} '
              f'relabeled={relabeled} skipped_cone={skipped_cone} '
              f'skipped_prefilter={skipped_prefilter} '
              f'(total={relabeled + skipped_cone + skipped_prefilter})')

        if relabeled > 0:  # importing with zero scripts would fail on a missing dir
            self.run_z3pyscript_labeling()
            self.import_labels(constant_value)

        # gates outside the affected region keep their previous-iteration labels
        if previous_labels:
            for gate, label in previous_labels.items():
                self.append_label(gate, label)

        return self.labels

    @property
    def graph_in_path(self): raise RuntimeError('[DEPRECATED] talk with Marco if you need this')
    @property
    def approximate_in_path(self): raise RuntimeError('[DEPRECATED] talk with Marco if you need this')

    def import_labels(self, constant_value: bool = False) -> Dict:
        label_dict: Dict[str, int] = {}
        folder = path_join(self.temporary_folder_path, 'outreport')
        extension = 'csv'

        all_dirs = [f for f in os.listdir(folder)]
        # print(f'{all_dirs = }')
        relevant_dir = None
        for dir in all_dirs:
            if re.search(f'{self.approximate_benchmark}_{SINGLE}', dir) and os.path.isdir(f'{folder}/{dir}') and re.search(f'{constant_value}', dir):
                relevant_dir = f'{folder}/{dir}'

        all_csv = [f for f in os.listdir(relevant_dir)]
        for report in all_csv:
            if re.search(self.approximate_benchmark, report) and report.endswith(extension):
                gate_label = re.search('(g\d+)', report).group(1)

                with open(f'{relevant_dir}/{report}', 'r') as r:
                    csvreader = csv.reader(r)
                    for line in csvreader:
                        if re.search(WCE, line[0]):
                            gate_wce = int(line[1])

                            label_dict[gate_label] = gate_wce
                            self.append_label(gate_label, gate_wce)
        return label_dict

    def convert_gv_to_z3pyscript_test(self):
        self.set_z3_report(path_join(self.temporary_folder_path, f'outreport_{self.name}_test.csv'))
        self.set_out_path(path_join(self.temporary_folder_path, f'outz3_{self.name}_test.py'))
        self.set_pyscript_results_out_path(path_join(self.temporary_folder_path, f'testz3_{self.name}_test.py'))

        import_string = self.create_imports()
        abs_function = self.create_abs_function()
        exact_circuit_declaration = self.declare_original_circuit()
        exact_circuit_expression = self.express_original_circuit()
        # TODO: Fix Later
        if self.metric == WHD:
            output_declaration = ''
            print(f'ERROR!!! Right now testing is not possible on WHD!')
        else:
            output_declaration = self.declare_original_output()
        exact_function = self.declare_original_function()
        solver = self.declare_solver()
        sample_expression = self.express_samples()
        store_results = self.store_results()
        self.set_z3pyscript(import_string + abs_function + exact_circuit_declaration + exact_circuit_expression +
                            output_declaration + exact_function + solver + sample_expression + store_results)

    def convert_gv_to_z3pyscript_maxerror_qor(self, strategy: str = DEFAULT_STRATEGY):

        self.experiment = QOR
        self.set_strategy(strategy)

        if self.metric == WRE:
            folder = path_join(self.temporary_folder_path, 'outreport')
            extension = 'csv'
            if self.optimization == OPTIMIZE or self.optimization == MAXIMIZE and (self.strategy != BISECTION):
                self.set_z3_report(
                    f'{folder}/{self.approximate_benchmark}_{self.experiment}_{self.metric}_d{self.precision}_{self.strategy}_{self.optimization}.{extension}')
            else:
                self.set_z3_report(
                    f'{folder}/{self.approximate_benchmark}_{self.experiment}_{self.metric}_d{self.precision}_{self.strategy}.{extension}')

            folder = path_join(self.temporary_folder_path, 'outz3')
            extension = 'py'
            if self.optimization == OPTIMIZE or self.optimization == MAXIMIZE and (self.strategy != BISECTION):
                self.set_out_path(
                    f'{folder}/{self.approximate_benchmark}_{self.experiment}_{self.metric}_d{self.precision}_{self.strategy}_{self.optimization}.{extension}')
            else:
                self.set_out_path(
                    f'{folder}/{self.approximate_benchmark}_{self.experiment}_{self.metric}_d{self.precision}_{self.strategy}.{extension}')

        else:
            folder = path_join(self.temporary_folder_path, 'outreport')
            extension = 'csv'
            if self.optimization == OPTIMIZE or self.optimization == MAXIMIZE and (self.strategy != BISECTION):
                self.set_z3_report(
                    f'{folder}/{self.approximate_benchmark}_{self.experiment}_{self.metric}_{self.strategy}_{self.optimization}.{extension}')
            else:
                self.set_z3_report(
                    f'{folder}/{self.approximate_benchmark}_{self.experiment}_{self.metric}_{self.strategy}.{extension}')

            folder = path_join(self.temporary_folder_path, 'outz3')
            extension = 'py'
            if self.optimization == OPTIMIZE or self.optimization == MAXIMIZE and (self.strategy != BISECTION):
                self.set_out_path(
                    f'{folder}/{self.approximate_benchmark}_{self.experiment}_{self.metric}_{self.strategy}_{self.optimization}.{extension}')
            else:
                self.set_out_path(
                    f'{folder}/{self.approximate_benchmark}_{self.experiment}_{self.metric}_{self.strategy}.{extension}')

        import_string = self.create_imports()
        abs_function = self.create_abs_function()

        # exact_part
        original_circuit_declaration = self.declare_original_circuit()
        original_circuit_expression = self.express_original_circuit()
        if self.metric == WHD:
            original_output_declaration = 'Blah Blah Blah\n'
        else:
            original_output_declaration = self.declare_original_output()

        # approximate_part
        approximate_circuit_declaration = self.declare_approximate_circuit()
        approximate_circuit_expression = self.express_approximate_circuit()
        if self.metric == WHD:
            approximate_output_declaration = 'Blah Blah Blah\n'
        else:
            approximate_output_declaration = self.declare_approximate_output()

        # error distance function
        declare_error_distance_function = self.declare_error_distance_function()

        # strategy
        strategy = self.express_strategy()

        self.set_z3pyscript(import_string + abs_function + original_circuit_declaration + original_circuit_expression +
                            original_output_declaration + approximate_circuit_declaration + approximate_circuit_expression +
                            approximate_output_declaration + declare_error_distance_function + strategy)

    # TODO
    # Naming problems for more than one gate removal

    def create_pruned_z3pyscript(self, gates: list, constant_value: bool = False):
        self.create_pruned_graph_approximate(gates, constant_value)
        if self.experiment == SINGLE:
            gate = gates[0]
        # TODO
        elif self.experiment == RANDOM:
            gate = 'id0'

        folder = path_join(self.temporary_folder_path, 'outreport')
        extension = 'csv'
        if self.metric == WRE:
            if self.optimization == OPTIMIZE or self.optimization == MAXIMIZE and (self.strategy != BISECTION):
                folder = f'{folder}/{self.name}_{self.experiment}_{self.metric}_d{self.precision}_{self.strategy}_{self.optimization}'
                os.makedirs(folder, exist_ok=True)
                self.set_z3_report(
                    f'{folder}/{self.name}_{self.experiment}_{self.metric}_d{self.precision}_{self.strategy}_{self.optimization}_{gate}.{extension}')
            else:
                folder = f'{folder}/{self.name}_{self.experiment}_{self.metric}_d{self.precision}_{self.strategy}'
                os.makedirs(folder, exist_ok=True)
                self.set_z3_report(
                    f'{folder}/{self.name}_{self.experiment}_{self.metric}_d{self.precision}_{self.strategy}_{gate}.{extension}')
        else:

            if self.optimization == OPTIMIZE or self.optimization == MAXIMIZE and (self.strategy != BISECTION):
                folder = f'{folder}/{self.name}_{self.experiment}_{self.metric}_{self.strategy}_{self.optimization}'
                os.makedirs(folder, exist_ok=True)
                self.set_z3_report(
                    f'{folder}/{self.name}_{self.experiment}_{self.metric}_{self.strategy}_{self.optimization}_{gate}.{extension}')
            else:
                folder = f'{folder}/{self.name}_{self.experiment}_{self.metric}_{self.strategy}'
                os.makedirs(folder, exist_ok=True)
                self.set_z3_report(
                    f'{folder}/{self.name}_{self.experiment}_{self.metric}_{self.strategy}_{gate}.{extension}')

        folder = path_join(self.temporary_folder_path, 'outz3')
        extension = 'py'
        if self.metric == WRE:
            if self.optimization == OPTIMIZE or self.optimization == MAXIMIZE and (self.strategy != BISECTION):
                folder = f'{folder}/{self.name}_{self.experiment}_{self.metric}_d{self.precision}_{self.strategy}_{self.optimization}'
                os.makedirs(folder, exist_ok=True)
                self.set_out_path(
                    f'{folder}/{self.name}_{self.experiment}_{self.metric}_d{self.precision}_{self.strategy}_{self.optimization}_{gate}.{extension}')
            else:
                folder = f'{folder}/{self.name}_{self.experiment}_{self.metric}_d{self.precision}_{self.strategy}'
                os.makedirs(folder, exist_ok=True)
                self.set_out_path(
                    f'{folder}/{self.name}_{self.experiment}_{self.metric}_d{self.precision}_{self.strategy}_{gate}.{extension}')
        else:
            if self.optimization == OPTIMIZE or self.optimization == MAXIMIZE and (self.strategy != BISECTION):
                folder = f'{folder}/{self.name}_{self.experiment}_{self.metric}_{self.strategy}_{self.optimization}'
                os.makedirs(folder, exist_ok=True)
                self.set_out_path(
                    f'{folder}/{self.name}_{self.experiment}_{self.metric}_{self.strategy}_{self.optimization}_{gate}.{extension}')
            else:
                folder = f'{folder}/{self.name}_{self.experiment}_{self.metric}_{self.strategy}'
                os.makedirs(folder, exist_ok=True)
                self.set_out_path(
                    f'{folder}/{self.name}_{self.experiment}_{self.metric}_{self.strategy}_{gate}.{extension}')

        self.append_pyscript_files_for_labeling(self.out_path)

        import_string = self.create_imports()
        abs_function = self.create_abs_function()

        # exact_part
        original_circuit_declaration = self.declare_original_circuit()
        original_circuit_expression = self.express_original_circuit()
        if self.metric == WHD:
            original_output_declaration = f'\n'
        else:
            original_output_declaration = self.declare_original_output()

        approximate_circuit_declaration = self.declare_approximate_circuit()
        approximate_circuit_expression = self.express_approximate_circuit()

        if self.metric == WHD:
            xor_miter_declaration = self.declare_xor_miter()
        else:
            approximate_output_declaration = self.declare_approximate_output()

        # error distance function
        declare_error_distance_function = self.declare_error_distance_function()

        # strategy
        strategy = self.express_strategy()

        if self.metric == WHD:
            self.set_z3pyscript(
                import_string + abs_function + original_circuit_declaration + original_circuit_expression +
                approximate_circuit_expression + xor_miter_declaration + declare_error_distance_function + strategy)
        else:
            self.set_z3pyscript(
                import_string + abs_function + original_circuit_declaration + original_circuit_expression +
                original_output_declaration + approximate_circuit_declaration + approximate_circuit_expression +
                approximate_output_declaration + declare_error_distance_function + strategy)

        self.export_z3pyscript()

    def create_pruned_z3pyscript_approximate(self, gates: list, constant_value: bool = False):
        self.create_pruned_graph_approximate(gates, constant_value)
        if self.experiment == SINGLE:
            gate = gates[0]
        # TODO
        elif self.experiment == RANDOM:
            gate = 'id0'

        folder = path_join(self.temporary_folder_path, 'outreport')
        extension = 'csv'
        if self.metric == WRE:
            if self.optimization == OPTIMIZE or self.optimization == MAXIMIZE and (self.strategy != BISECTION):
                folder = f'{folder}/{self.approximate_benchmark}_{self.experiment}_{self.metric}_d{self.precision}_{self.strategy}_{self.optimization}_{constant_value}'
                os.makedirs(folder, exist_ok=True)
                self.set_z3_report(
                    f'{folder}/{self.approximate_benchmark}_{self.experiment}_{self.metric}_d{self.precision}_{self.strategy}_{self.optimization}_{gate}.{extension}')
            else:
                folder = f'{folder}/{self.approximate_benchmark}_{self.experiment}_{self.metric}_d{self.precision}_{self.strategy}_{constant_value}'
                os.makedirs(folder, exist_ok=True)
                self.set_z3_report(
                    f'{folder}/{self.approximate_benchmark}_{self.experiment}_{self.metric}_d{self.precision}_{self.strategy}_{gate}.{extension}')
        else:

            if self.optimization == OPTIMIZE or self.optimization == MAXIMIZE and (self.strategy != BISECTION):
                folder = f'{folder}/{self.approximate_benchmark}_{self.experiment}_{self.metric}_{self.strategy}_{self.optimization}_{constant_value}'
                os.makedirs(folder, exist_ok=True)
                self.set_z3_report(
                    f'{folder}/{self.approximate_benchmark}_{self.experiment}_{self.metric}_{self.strategy}_{self.optimization}_{gate}.{extension}')
            else:
                folder = f'{folder}/{self.approximate_benchmark}_{self.experiment}_{self.metric}_{self.strategy}_{constant_value}'
                os.makedirs(folder, exist_ok=True)
                self.set_z3_report(
                    f'{folder}/{self.approximate_benchmark}_{self.experiment}_{self.metric}_{self.strategy}_{gate}.{extension}')

        folder = path_join(self.temporary_folder_path, 'outz3')
        extension = 'py'
        if self.metric == WRE:
            if self.optimization == OPTIMIZE or self.optimization == MAXIMIZE and (self.strategy != BISECTION):
                folder = f'{folder}/{self.approximate_benchmark}_{self.experiment}_{self.metric}_d{self.precision}_{self.strategy}_{self.optimization}_{constant_value}'
                os.makedirs(folder, exist_ok=True)
                self.set_out_path(
                    f'{folder}/{self.approximate_benchmark}_{self.experiment}_{self.metric}_d{self.precision}_{self.strategy}_{self.optimization}_{gate}.{extension}')
            else:
                folder = f'{folder}/{self.approximate_benchmark}_{self.experiment}_{self.metric}_d{self.precision}_{self.strategy}_{constant_value}'
                os.makedirs(folder, exist_ok=True)
                self.set_out_path(
                    f'{folder}/{self.approximate_benchmark}_{self.experiment}_{self.metric}_d{self.precision}_{self.strategy}_{gate}.{extension}')
        else:
            if self.optimization == OPTIMIZE or self.optimization == MAXIMIZE and (self.strategy != BISECTION):
                folder = f'{folder}/{self.approximate_benchmark}_{self.experiment}_{self.metric}_{self.strategy}_{self.optimization}_{constant_value}'
                os.makedirs(folder, exist_ok=True)
                self.set_out_path(
                    f'{folder}/{self.approximate_benchmark}_{self.experiment}_{self.metric}_{self.strategy}_{self.optimization}_{gate}.{extension}')
            else:
                folder = f'{folder}/{self.approximate_benchmark}_{self.experiment}_{self.metric}_{self.strategy}_{constant_value}'
                os.makedirs(folder, exist_ok=True)
                self.set_out_path(
                    f'{folder}/{self.approximate_benchmark}_{self.experiment}_{self.metric}_{self.strategy}_{gate}.{extension}')

        self.append_pyscript_files_for_labeling(self.out_path)

        import_string = self.create_imports()
        abs_function = self.create_abs_function()

        # exact_part
        original_circuit_declaration = self.declare_original_circuit()
        original_circuit_expression = self.express_original_circuit()
        if self.metric == WHD:
            original_output_declaration = f'\n'
        else:
            original_output_declaration = self.declare_original_output()

        approximate_circuit_declaration = self.declare_approximate_circuit()
        approximate_circuit_expression = self.express_approximate_circuit()

        if self.metric == WHD:
            xor_miter_declaration = self.declare_xor_miter()
        else:
            approximate_output_declaration = self.declare_approximate_output()

        # error distance function
        declare_error_distance_function = self.declare_error_distance_function()

        # strategy
        strategy = self.express_strategy()

        if self.metric == WHD:
            self.set_z3pyscript(
                import_string + abs_function + original_circuit_declaration + original_circuit_expression +
                approximate_circuit_expression + xor_miter_declaration + declare_error_distance_function + strategy)
        else:
            self.set_z3pyscript(
                import_string + abs_function + original_circuit_declaration + original_circuit_expression +
                original_output_declaration + approximate_circuit_declaration + approximate_circuit_expression +
                approximate_output_declaration + declare_error_distance_function + strategy)

        self.export_z3pyscript()

    def run_z3pyscript_labeling(self):
        """Override the base-class pool with a proper concurrent.futures executor.

        The base class uses ``active_procs.pop(0)`` which always waits for the
        *oldest* process rather than whichever finishes first.  On workloads
        where gate solve-times vary this serialises the entire pool.  We replace
        it with ``ThreadPoolExecutor`` + ``as_completed`` so the next script is
        dispatched the moment any worker slot becomes free.
        """
        _python = sys.executable

        def _run(script):
            proc = Popen([_python, script], stderr=PIPE, stdout=PIPE)
            proc.communicate()

        scripts = list(self.pyscript_files_for_labeling)
        if not scripts:
            return

        if self.parallel:
            n_workers = os.cpu_count() or 1
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_run, s): s for s in scripts}
                for _ in concurrent.futures.as_completed(futures):
                    pass
        else:
            for s in scripts:
                _run(s)

    def __repr__(self):
        return (
            f'An object of class Z3solver\n'
            f'{self.name = }\n'
            # f'{self.graph_in_path = }\n'
            f'{self.out_path = }\n'
        )
