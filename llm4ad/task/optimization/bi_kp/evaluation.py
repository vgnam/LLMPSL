from __future__ import annotations

from typing import Any
import numpy as np
from llm4ad.base import Evaluation
from llm4ad.task.optimization.bi_kp.get_instance import GetData
from llm4ad.task.optimization.bi_kp.template import template_program, task_description
from llm4ad.task.optimization.hv_utils import DEFAULT_SEARCH_ITERATIONS, scale_hypervolume
from pymoo.indicators.hv import HV
import random
import time

__all__ = ['BIKPEvaluation', 'fixed_ideal_point']

FIXED_IDEAL_MAGNITUDE_BY_SIZE = {
    20: 11.0,
    50: 25.0,
    100: 48.0,
    150: 60.0,
    200: 70.0,
    300: 115.0,
    400: 135.0,
}


def knapsack_value(solution: np.ndarray, weight_lst: np.ndarray, value1_lst: np.ndarray, value2_lst: np.ndarray, capacity: float):
    if np.sum(solution * weight_lst) > capacity:
        return -1e10, -1e10  # Penalize infeasible solutions
    # check if the solution is feasible
    if not np.all(np.isin(solution, [0, 1])):
        return -1e10, -1e10
    if len(solution) != len(weight_lst):
        return -1e10, -1e10
    total_val1 = np.sum(solution * value1_lst)
    total_val2 = np.sum(solution * value2_lst)
    return total_val1, total_val2


def dominates(a, b):
    """True if a dominates b (maximization)."""
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def random_solution(weight_lst, capacity, problem_size):
    # Generate a permutation of the problem size and then select a subset of items in order with a probability of 0.5 till reaching the capacity
    # This is a simple random solution generator for the knapsack problem
    sol = list(range(problem_size))
    random.shuffle(sol)
    selected_items = []
    total_weight = 0
    for item in sol:
        if total_weight + weight_lst[item] <= capacity:
            selected_items.append(item)
            total_weight += weight_lst[item]
    return np.array([1 if i in selected_items else 0 for i in range(problem_size)])


def _random_solution_rng(weight_lst, capacity, problem_size, rng: np.random.Generator):
    order = rng.permutation(problem_size)
    selected_items = []
    total_weight = 0.0
    for item in order:
        if total_weight + weight_lst[item] <= capacity:
            selected_items.append(int(item))
            total_weight += weight_lst[item]
    return np.array([1 if i in selected_items else 0 for i in range(problem_size)])


def estimate_reference_point(instance_data, capacity, problem_size, samples_per_instance: int = 128, margin: float = 0.05):
    rng = np.random.default_rng(4049 + problem_size + len(instance_data))
    objectives = []
    for weight_lst, value1_lst, value2_lst in instance_data:
        for _ in range(samples_per_instance):
            sol = _random_solution_rng(weight_lst, capacity, problem_size, rng)
            value = knapsack_value(sol, weight_lst, value1_lst, value2_lst, capacity)
            objectives.append([-float(value[0]), -float(value[1])])
    arr = np.array(objectives, dtype=float)
    nadir = np.max(arr, axis=0)
    span = np.maximum(np.max(arr, axis=0) - np.min(arr, axis=0), 1e-6)
    return nadir + margin * span


def fixed_ideal_point(problem_size, objective_num=2):
    """Return the method- and run-independent BI-KP normalization ideal."""
    magnitude = FIXED_IDEAL_MAGNITUDE_BY_SIZE.get(problem_size, float(problem_size))
    return np.full(objective_num, -magnitude, dtype=float)



    


def _archive_objectives(archive):
    # PBC-LLM uses minimization objectives; knapsack values are maximized.
    return [[-float(obj[0]), -float(obj[1])] for _, obj in archive]


def evaluate(
    instance_data,
    n_instance,
    problem_size,
    ref_point,
    ideal_point,
    capacity,
    eva: callable,
    eval_seed: int | None = None,
    *,
    return_mo_trace: bool = False,
    trace_points: int = 21,
    total_iterations: int = DEFAULT_SEARCH_ITERATIONS,
):
    if eval_seed is not None:
        random.seed(eval_seed)
        np.random.seed(eval_seed)
    obj_1 = np.ones(n_instance)
    obj_2 = np.ones(n_instance)
    n_ins = 0
    final_list = []
    archive_trajectories = []
    checkpoints = set(np.linspace(0, total_iterations, trace_points, dtype=int).tolist())
    for weight_lst, value1_lst, value2_lst in instance_data:
        start = time.time()
        s = [random_solution(weight_lst, capacity, problem_size) for _ in range(20)]
        Archive = [(s_, knapsack_value(s_, weight_lst, value1_lst, value2_lst, capacity)) for s_ in s if knapsack_value(s_, weight_lst, value1_lst, value2_lst, capacity)[0] > -1e5]
        trajectory = []
        if 0 in checkpoints:
            trajectory.append(_archive_objectives(Archive))
        for iteration in range(1, total_iterations + 1):
            archive_arg = [(sol.copy(), obj) for sol, obj in Archive]
            s_prime = np.array(
                eva(
                    archive_arg,
                    weight_lst.copy(),
                    value1_lst.copy(),
                    value2_lst.copy(),
                    capacity,
                )
            )
            f_s_prime = knapsack_value(s_prime, weight_lst, value1_lst, value2_lst, capacity)

            if f_s_prime[0] < -1e5:
                if iteration in checkpoints:
                    trajectory.append(_archive_objectives(Archive))
                continue  # Skip infeasible

            if not any(dominates(f_a, f_s_prime) for _, f_a in Archive):
                Archive = [(a, f_a) for a, f_a in Archive if not dominates(f_s_prime, f_a)]
                Archive.append((s_prime, f_s_prime))
            if iteration in checkpoints:
                trajectory.append(_archive_objectives(Archive))
        end = time.time()
        objs = np.array([obj for _, obj in Archive]) * (-1)
        final_list.append(objs.tolist())
        archive_trajectories.append(trajectory)
        hv_indicator = HV(ref_point=ref_point)
        hv_value = hv_indicator(objs)
        obj_1[n_ins] = -scale_hypervolume(hv_value, ref_point, ideal_point)
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


class BIKPEvaluation(Evaluation):
    """Evaluator for the Bi-objective Knapsack Problem (BI-KP) using a custom algorithm."""

    def __init__(
        self,
        *,
        n_instance: int = 8,
        problem_size: int = 200,
        seed: int = 2025,
        eval_seed: int | None = None,
        timeout_seconds: int = 90,
        data_dir: str | None = None,
        return_mo_trace: bool = False,
        **kwargs,
    ):
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
        self.search_iterations = DEFAULT_SEARCH_ITERATIONS
        self.objective_num = 2
        self.objective_labels = ("negative_value_1", "negative_value_2")
        getData = GetData(self.n_instance, self.problem_size, seed=seed, data_dir=data_dir)
        self._datasets, self.cap = getData.generate_instances() 
        self.ref_point = estimate_reference_point(self._datasets, self.cap, self.problem_size) 
        self.ideal_point = fixed_ideal_point(self.problem_size, self.objective_num)
        self.hv_normalization_policy = "fixed_by_problem_size"
        self.hv_fixed_ideal_magnitude_by_size = FIXED_IDEAL_MAGNITUDE_BY_SIZE.copy()
        self.normalization_ideal = self.ideal_point.copy()
        self.normalization_nadir = self.ref_point.copy()

    def evaluate_program(self, program_str: str, callable_func: callable):
        return evaluate(
            self._datasets,
            self.n_instance,
            self.problem_size,
            self.ref_point,
            self.ideal_point,
            self.cap,
            callable_func,
            self.eval_seed,
            return_mo_trace=self.return_mo_trace,
            total_iterations=self.search_iterations,
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
        tsp = BIKPEvaluation()
        cst, tme = tsp.evaluate_program('_', select_neighbor_func)
        result_queue.put([cst, tme])
    except Exception as e:
        result_queue.put(f"Error: {e}")

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    with open(f"Illustration/Bi KP 50/EoH/population_0/pop_1.json", "r") as f:
        data = json.load(f)
    for k in range(len(data)):
        if k == 9:
            for _ in range(1):
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
                result[0] = result[0] / (45*45)
                data[k]["score"] = result
                print(f"Evaluating with code {k+1}...", result)




