from __future__ import annotations

from typing import Any
import numpy as np
from llm4ad.base import Evaluation
from llm4ad.task.optimization.bi_cvrp.get_instance import GetData
from llm4ad.task.optimization.bi_cvrp.template import template_program, task_description
from llm4ad.task.optimization.hv_utils import scale_hypervolume
from pymoo.indicators.hv import HV 
import random
import time

__all__ = ['BICVRPEvaluation']

def compute_route_length(route: np.ndarray, distance_matrix: np.ndarray) -> float:
    if len(route) <= 1:
        return 0.0
    return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))

def evaluate_solution(routes: list[np.ndarray], distance_matrix: np.ndarray):
    total_distance = 0.0
    longest_route = 0.0
    for route in routes:
        d = compute_route_length(route, distance_matrix)
        total_distance += d
        longest_route = max(longest_route, d)
    return total_distance, longest_route

def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

def random_solution(num_customers: int, capacity: float, demand: np.ndarray) -> list[np.ndarray]:
    customers = list(range(1, num_customers + 1))
    random.shuffle(customers)
    routes = []
    current_route = [0]
    current_load = 0.0
    for customer in customers:
        if current_load + demand[customer] <= capacity:
            current_route.append(customer)
            current_load += demand[customer]
        else:
            current_route.append(0)
            routes.append(np.array(current_route))
            current_route = [0, customer]
            current_load = demand[customer]
    current_route.append(0)
    routes.append(np.array(current_route))
    return routes


def _random_solution_rng(num_customers: int, capacity: float, demand: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    customers = np.arange(1, num_customers + 1)
    rng.shuffle(customers)
    routes = []
    current_route = [0]
    current_load = 0.0
    for customer in customers:
        if current_load + demand[customer] <= capacity:
            current_route.append(int(customer))
            current_load += demand[customer]
        else:
            current_route.append(0)
            routes.append(np.array(current_route))
            current_route = [0, int(customer)]
            current_load = demand[customer]
    current_route.append(0)
    routes.append(np.array(current_route))
    return routes


def estimate_reference_point(instance_data, capacity, samples_per_instance: int = 128, margin: float = 1.05):
    rng = np.random.default_rng(3031 + len(instance_data))
    objectives = []
    for _, demand, distance_matrix in instance_data:
        num_customers = len(demand) - 1
        for _ in range(samples_per_instance):
            routes = _random_solution_rng(num_customers, capacity, demand, rng)
            objectives.append(evaluate_solution(routes, distance_matrix))
    ref = np.max(np.array(objectives, dtype=float), axis=0)
    return ref * margin

def is_feasible_solution(routes: List[np.ndarray], demand: np.ndarray, capacity: float) -> bool:
    """
    Check if all routes satisfy the vehicle capacity constraint, visit each customer exactly once,
    and have no duplicate visits. Routes are numpy arrays starting and ending with depot index 0.
    """
    all_customers = set(range(1, len(demand)))
    visited = []
    # Check capacity and collect visited customers
    for route in routes:
        # depot at start and end
        if route[0] != 0 or route[-1] != 0:
            return False
        if len(route) == 2:
            continue
        # compute load excluding depot
        customers = route[1:-1]
        load = sum(demand[customers])
        if load > capacity:
            return False
        visited.extend(customers.tolist())
    # Check for duplicates and completeness
    visited_set = set(visited)
    if len(visited) != len(visited_set):
        return False
    if visited_set != all_customers:
        return False
    return True

def _copy_routes(routes: list[np.ndarray]) -> list[np.ndarray]:
    return [route.copy() for route in routes]


def _archive_objectives(archive):
    return [list(map(float, obj)) for _, obj in archive]


def evaluate(
    instance_data,
    n_instance,
    ref_point,
    capacity,
    evaluate_func: callable,
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
    final_list = []
    archive_trajectories = []
    total_iterations = 6000
    checkpoints = set(np.linspace(0, total_iterations, trace_points, dtype=int).tolist())
    for i, (coords, demand, distance_matrix) in enumerate(instance_data):
        start = time.time()
        init_solutions = [random_solution(len(demand)-1, capacity, demand) for _ in range(10)]
        archive = [(s, evaluate_solution(s, distance_matrix)) for s in init_solutions]
        trajectory = []
        if 0 in checkpoints:
            trajectory.append(_archive_objectives(archive))
        for iteration in range(1, total_iterations + 1):
            archive_arg = [(_copy_routes(sol), obj) for sol, obj in archive]
            s_prime = evaluate_func(
                archive_arg,
                coords.copy(),
                demand.copy(),
                distance_matrix.copy(),
                capacity,
            )
            if not is_feasible_solution(s_prime, demand, capacity):
                if iteration in checkpoints:
                    trajectory.append(_archive_objectives(archive))
                continue
            f_prime = evaluate_solution(s_prime, distance_matrix)
            if not any(dominates(f, f_prime) for _, f in archive):
                archive = [(s, f) for s, f in archive if not dominates(f_prime, f)]
                archive.append((s_prime, f_prime))
            if iteration in checkpoints:
                trajectory.append(_archive_objectives(archive))
        end = time.time()
        objs = np.array([f for _, f in archive])
        final_list.append(objs.tolist())
        archive_trajectories.append(trajectory)
        hv_indicator = HV(ref_point=ref_point)
        hv_value = hv_indicator(objs)
        obj_1[i] = -scale_hypervolume(hv_value, ref_point)
        obj_2[i] = end - start
    if return_mo_trace:
        return {
            "objective_num": 2,
            "fronts": final_list,
            "archive_trajectories": archive_trajectories,
            "legacy_score": [float(np.mean(obj_1)), float(np.mean(obj_2))],
        }
    return np.mean(obj_1), np.mean(obj_2)

class BICVRPEvaluation(Evaluation):
    def __init__(
        self,
        *,
        n_instance: int = 8,
        problem_size: int = 100,
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
        self.objective_num = 2
        self.objective_labels = ("total_distance", "longest_route")
        getData = GetData(self.n_instance, self.problem_size, seed=seed, data_dir=data_dir)
        self._datasets, self.cap = getData.generate_instances()
        self.ref_point = estimate_reference_point(self._datasets, self.cap)

    def evaluate_program(self, program_str: str, callable_func: callable):
        return evaluate(
            self._datasets,
            self.n_instance,
            self.ref_point,
            self.cap,
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
        tsp = BICVRPEvaluation()
        cst, tme = tsp.evaluate_program('_', select_neighbor_func)
        result_queue.put([cst, tme])
    except Exception as e:
        result_queue.put(f"Error: {e}")


import numpy as np
from typing import List, Tuple
import random
import numpy as np
import random
from typing import List, Tuple

def is_feasible(route: np.ndarray, demand: np.ndarray, capacity: float) -> bool:
    load = sum(demand[node] for node in route[1:-1])  # Exclude depot
    return load <= capacity

def select_neighbor(archive: List[Tuple[np.ndarray, Tuple[float, float]]], 
                       coords: np.ndarray, 
                       demand: np.ndarray, 
                       distance_matrix: np.ndarray, 
                       capacity: float) -> np.ndarray:
    """
    Select a promising solution from the archive and generate a neighbor solution using advanced local search strategies.
    
    Args:
        archive: A list of tuples containing solutions and their corresponding objectives.
        coords: A numpy array of shape (n_nodes, 2) for node coordinates.
        demand: A numpy array where demand[i] is the demand of node i.
        distance_matrix: A numpy array for the distances between nodes.
        capacity: A float for the maximum capacity of each vehicle.

    Returns:
        A new neighbor solution as a list of routes.
    """
    
    # Adaptive selection based on distance and makespan
    total_distance = sum(obj[0] for _, obj in archive)
    total_makespan = sum(obj[1] for _, obj in archive)
    probabilities = [(1 / (obj[0] + 1e-6)) * (1 / (obj[1] + 1e-6)) for _, obj in archive]
    selected_index = np.random.choice(len(archive), p=probabilities / np.sum(probabilities))
    base_solution = archive[selected_index][0].copy()
    
    new_solution = [route.copy() for route in base_solution]
    
    # Prioritize routes with higher loads
    vehicle_loads = [sum(demand[node] for node in route[1:-1]) for route in new_solution]
    prioritized_indices = np.argsort(vehicle_loads)[::-1]  # Sort indices by load descending

    # Store modifications for acceptance check
    modifications = []
    
    # Generate diverse neighbor solutions
    for selected_vehicle_index in prioritized_indices:
        route = new_solution[selected_vehicle_index]

        if len(route) > 3:  # At least one customer should be present to modify
            action_type = random.choices(['swap', 'shift', 'insert', 'reverse'], 
                                          weights=[0.4, 0.2, 0.2, 0.2])[0]
            modified_route = route.copy()  # Create a copy for modification
            
            if action_type == 'swap':
                idx1, idx2 = random.sample(range(1, len(modified_route) - 1), 2)  # Exclude depot
                modified_route[idx1], modified_route[idx2] = modified_route[idx2], modified_route[idx1]
            elif action_type == 'shift':
                customer_idx = random.randint(1, len(modified_route) - 2)
                if random.random() < 0.5 and customer_idx < len(modified_route) - 2:  # Shift right
                    new_position = customer_idx + 1
                    if new_position < len(modified_route) - 1:  # Ensure not going out of bounds
                        modified_route[customer_idx], modified_route[new_position] = modified_route[new_position], modified_route[customer_idx]
                else:  # Shift left
                    new_position = customer_idx - 1
                    if new_position > 0:  # Ensure not going out of bounds
                        modified_route[customer_idx], modified_route[new_position] = modified_route[new_position], modified_route[customer_idx]
            elif action_type == 'insert':
                customer_idx = random.randint(1, len(modified_route) - 2)
                new_vehicle_idx = random.randint(0, len(new_solution) - 1)
                if new_vehicle_idx != selected_vehicle_index:  # Don't insert into the same vehicle
                    modified_route = np.delete(modified_route, customer_idx)  # Remove from old route
                    new_solution[new_vehicle_idx] = np.insert(new_solution[new_vehicle_idx], -1, route[customer_idx])  # Insert before depot
            elif action_type == 'reverse':
                start_idx = random.randint(1, len(modified_route) - 2)
                end_idx = random.randint(start_idx, len(modified_route) - 2)
                modified_route[start_idx:end_idx + 1] = modified_route[start_idx:end_idx + 1][::-1]  # Reverse segment

            # Check feasibility after modification
            if is_feasible(modified_route, demand, capacity):
                modifications.append((selected_vehicle_index, modified_route))

    # If multiple valid modifications exist, select one randomly
    if modifications:
        selected_modification = random.choice(modifications)
        new_solution[selected_modification[0]] = selected_modification[1]

    return new_solution


if __name__ == '__main__':
    import numpy as np
    from typing import List, Tuple
    import random 


    
    tsp = BICVRPEvaluation()
    # print("def select_neighbor(archive: List[Tuple[np.ndarray, Tuple[float, float]]], coords: np.ndarray, demand: np.ndarray, distance_matrix: np.ndarray, capacity: float) -> np.ndarray:\n    best_solution = min(archive, key=lambda x: x[1][1])  # Minimize makespan\n    routes = best_solution[0]\n    \n    # Step 2: Create a copy of the selected solution to modify\n    neighbor_solution = [np.copy(route) for route in routes]\n    \n    # Step 3: Identify two routes to swap segments from\n    route_indices = np.random.choice(len(neighbor_solution), 2, replace=False)\n    route1, route2 = neighbor_solution[route_indices[0]], neighbor_solution[route_indices[1]]\n\n    # Step 4: Choose a segment to swap from each route\n    if len(route1) > 2 and len(route2) > 2:  # Ensure both routes have at least one customer\n        # Randomly select a segment from route1 and route2\n        swap_start1, swap_end1 = np.random.randint(1, len(route1)-1), np.random.randint(1, len(route1)-1)\n        swap_start2, swap_end2 = np.random.randint(1, len(route2)-1), np.random.randint(1, len(route2)-1)\n\n        # Ensure proper ordering\n        if swap_start1 > swap_end1:\n            swap_start1, swap_end1 = swap_end1, swap_start1\n        if swap_start2 > swap_end2:\n            swap_start2, swap_end2 = swap_end2, swap_start2\n            \n        # Extract segments to swap\n        segment1 = route1[swap_start1:swap_end1 + 1]\n        segment2 = route2[swap_start2:swap_end2 + 1]\n\n        # Swap segments\n        new_route1 = np.concatenate((route1[:swap_start1], segment2, route1[swap_end1 + 1:]))\n        new_route2 = np.concatenate((route2[:swap_start2], segment1, route2[swap_end2 + 1:]))\n\n        # Step 5: Check feasibility and adjust if necessary\n        def adjust_route(route):\n            total_demand = sum(demand[route])\n            while total_demand > capacity:\n                # Remove the last customer until the route is feasible\n                route = route[:-1]\n                total_demand = sum(demand[route])\n            return route\n\n        new_route1 = adjust_route(new_route1)\n        new_route2 = adjust_route(new_route2)\n\n        # Step 6: Update the neighbor solution\n        neighbor_solution[route_indices[0]] = new_route1\n        neighbor_solution[route_indices[1]] = new_route2\n\n    return neighbor_solution\n\n")
    for _ in range(5):
        cst, tme = tsp.evaluate_program('_',select_neighbor)
        print("Cost:", -cst / (80*8))
        print("Time:", tme)
        print("--------------------------------------------------")






