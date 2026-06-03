# name: str: TSPEvaluation
# Parameters:
# timeout_seconds: int: 20
# end
from __future__ import annotations

from typing import Any
import numpy as np
from llm4ad.base import Evaluation
from llm4ad.task.optimization.bi_tsp_semo.get_instance import GetData
from llm4ad.task.optimization.bi_tsp_semo.template import template_program, task_description
from llm4ad.task.optimization.hv_utils import scale_hypervolume
from pymoo.indicators.hv import HV 
import random
import time 

__all__ = ['BITSPEvolution']


def tour_cost(instance, solution, problem_size):

        cost_1 = 0
        cost_2 = 0
        
        for j in range(problem_size - 1):
            node1, node2 = int(solution[j]), int(solution[j + 1])
            
            coord_1_node1, coord_2_node1 = instance[node1][:2], instance[node1][2:]
            coord_1_node2, coord_2_node2 = instance[node2][:2], instance[node2][2:]

            cost_1 += np.linalg.norm(coord_1_node1 - coord_1_node2)
            cost_2 += np.linalg.norm(coord_2_node1 - coord_2_node2)
        
        node_first, node_last = int(solution[0]), int(solution[-1])
        
        coord_1_first, coord_2_first = instance[node_first][:2], instance[node_first][2:]
        coord_1_last, coord_2_last = instance[node_last][:2], instance[node_last][2:]

        cost_1 += np.linalg.norm(coord_1_last - coord_1_first)
        cost_2 += np.linalg.norm(coord_2_last - coord_2_first)

        return cost_1, cost_2  
    

def dominates(a, b):
        """True if a dominates b (minimization)."""
        return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

def random_solution(problem_size):
        sol = list(range(problem_size))
        random.shuffle(sol)
        return np.array(sol)


def estimate_reference_point(instance_data, problem_size, samples_per_instance: int = 128, margin: float = 1.05):
        rng = np.random.default_rng(2025 + 17 * problem_size + len(instance_data))
        objectives = []
        for instance, _, _ in instance_data:
            for _ in range(samples_per_instance):
                sol = rng.permutation(problem_size)
                objectives.append(tour_cost(instance, sol, problem_size))
        ref = np.max(np.array(objectives, dtype=float), axis=0)
        return ref * margin


def check_constraint(solution, problem_size):
    sol = list(solution)
    if len(sol) != problem_size:
        return False
    if len(set(sol)) != problem_size:
        return False
    if not all(0 <= x < problem_size for x in solution):
        return False
    return True
         



def _archive_objectives(archive):
        return [list(map(float, obj)) for _, obj in archive]


def evaluate(
        instance_data,
        n_instance,
        problem_size,
        ref_point,
        eva: callable,
        eval_seed: int | None = None,
        *,
        return_mo_trace: bool = False,
        trace_points: int = 21,
):
        if eval_seed is not None:
            random.seed(eval_seed)
            np.random.seed(eval_seed)
        obj_1 = np.ones(n_instance)
        obj_2 = np.ones(n_instance)
        n_ins = 0
        final_list = []
        archive_trajectories = []
        total_iterations = 2000
        checkpoints = set(np.linspace(0, total_iterations, trace_points, dtype=int).tolist())
        for _, (instance, distance_matrix_1, distance_matrix_2) in enumerate(instance_data):
            start = time.time()
            s = [random_solution(problem_size) for _ in range(100)]
            Archive = [(s_, tour_cost(instance, s_, problem_size)) for s_ in s]
            trajectory = []
            if 0 in checkpoints:
                trajectory.append(_archive_objectives(Archive))
            for iteration in range(1, total_iterations + 1):
                archive_arg = [(sol.copy(), obj) for sol, obj in Archive]
                s_prime = eva(
                    archive_arg,
                    instance.copy(),
                    distance_matrix_1.copy(),
                    distance_matrix_2.copy(),
                )
                if not check_constraint(s_prime, problem_size):
                    if iteration in checkpoints:
                        trajectory.append(_archive_objectives(Archive))
                    continue
                f_s_prime = tour_cost(instance, s_prime, problem_size)

                # Nếu không bị thống trị
                if not any(dominates(f_a, f_s_prime) for _, f_a in Archive):
                    # Loại bỏ các phần tử bị thống trị bởi f_s_prime
                    Archive = [(a, f_a) for a, f_a in Archive if not dominates(f_s_prime, f_a)]
                    # Thêm nghiệm mới
                    Archive.append((s_prime, f_s_prime))
                if iteration in checkpoints:
                    trajectory.append(_archive_objectives(Archive))
            end = time.time()
            objs = np.array([obj for _, obj in Archive])
            # Convert to list of lists
            final_list.append(objs.tolist())
            archive_trajectories.append(trajectory)
            # Tính HV
            hv_indicator = HV(ref_point=ref_point)
            hv_value = hv_indicator(objs)
            obj_1[n_ins] = -scale_hypervolume(hv_value, ref_point)
            obj_2[n_ins] = end - start
            n_ins += 1
        if return_mo_trace:
            return {
                "objective_num": 2,
                "fronts": final_list,
                "archive_trajectories": archive_trajectories,
                "legacy_score": [float(np.mean(obj_1)), float(np.mean(obj_2))],
            }
        return np.mean(obj_1), np.mean(obj_2)
            





class BITSPEvaluation(Evaluation):
    """Evaluator for the Bi-objective Traveling Salesman Problem (TSP) using a custom algorithm."""

    def __init__(
        self,
        *,
        n_instance: int = 4,
        problem_size: int = 20,
        seed: int = 2025,
        eval_seed: int | None = None,
        timeout_seconds: int = 60,
        data_dir: str | None = None,
        return_mo_trace: bool = False,
        **kwargs,
    ):

        """
            Args:
                None
            Raises:
                AttributeError: If the data key does not exist.
                FileNotFoundError: If the specified data file is not found.
        """

        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds
        )

        self.n_instance = n_instance
        self.problem_size = problem_size
        self.eval_seed = eval_seed
        self.return_mo_trace = return_mo_trace
        self.objective_num = 2
        self.objective_labels = ("distance_1", "distance_2")
        getData = GetData(self.n_instance, self.problem_size, seed=seed, data_dir=data_dir)
        self._datasets = getData.generate_instances()
        self.ref_point = estimate_reference_point(self._datasets, self.problem_size)

    def evaluate_program(self, program_str: str, callable_func: callable):
        return evaluate(
            self._datasets,
            self.n_instance,
            self.problem_size,
            self.ref_point,
            callable_func,
            self.eval_seed,
            return_mo_trace=self.return_mo_trace,
        )
    

import numpy as np
from typing import List, Tuple
import random
import json
import multiprocessing
import os
import warnings
warnings.filterwarnings("ignore")

def run_exec_and_eval(code_str, result_queue):
    try:
        local_vars = {}
        exec(code_str, globals(), local_vars)
        select_neighbor_func = local_vars["select_neighbor"]
        tsp = BITSPEvaluation()
        cst, tme = tsp.evaluate_program('_', select_neighbor_func)
        result_queue.put([cst, tme])
    except Exception as e:
        result_queue.put(f"Error: {e}")

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    with open(f"Illustration/Bi TSP /LLM PFG/population_0/pop_8.json", "r") as f:
        data = json.load(f)
    for k in range(len(data)):
        if k == 1:
            for _ in range(1):
                # print(data[k]["function"])
                select_neighbor_code = data[k]["function"]
                result_queue = multiprocessing.Queue()
                p = multiprocessing.Process(target=run_exec_and_eval, args=(select_neighbor_code, result_queue))
                p.start()
                p.join(timeout=3600)
                if p.is_alive():
                    print(f"Timeout on code {k+1}, skipping.")
                    p.terminate()
                    p.join()
                    data[k]["score"] = data[k-1]["score"]
                    continue
                result = result_queue.get()
                if isinstance(result, str) and result.startswith("Error"):
                    print(f"Error on code {k+1}: {result}")
                    continue
                result[0] = result[0] / (35*35)
                data[k]["score"] = result 
                print(f"Evaluating with code {k+1}...", result)







