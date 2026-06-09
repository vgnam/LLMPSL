from __future__ import annotations

import math
from threading import Lock
from typing import List
import numpy as np
import traceback
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from ...base import *


def _crowding_distance(front_values: np.ndarray) -> np.ndarray:
    front_values = np.asarray(front_values, dtype=float)
    n_points = len(front_values)
    if n_points == 0:
        return np.array([], dtype=float)
    if n_points <= 2:
        return np.full(n_points, float("inf"), dtype=float)

    distances = np.zeros(n_points, dtype=float)
    n_objectives = front_values.shape[1]
    for obj_idx in range(n_objectives):
        order = np.argsort(front_values[:, obj_idx])
        distances[order[0]] = float("inf")
        distances[order[-1]] = float("inf")
        min_value = front_values[order[0], obj_idx]
        max_value = front_values[order[-1], obj_idx]
        span = max_value - min_value
        if span <= 0:
            continue
        for pos in range(1, n_points - 1):
            if np.isinf(distances[order[pos]]):
                continue
            prev_value = front_values[order[pos - 1], obj_idx]
            next_value = front_values[order[pos + 1], obj_idx]
            distances[order[pos]] += (next_value - prev_value) / span
    return distances


class Population:
    def __init__(self, pop_size, generation=0, pop: List[Function] | Population | None = None):
        if pop is None:
            self._population = []
        elif isinstance(pop, list):
            self._population = pop
        else:
            self._population = pop._population

        self._pop_size = pop_size
        self._lock = Lock()
        self._next_gen_pop = []
        self._elitist = []
        self._generation = generation

    def __len__(self):
        return len(self._population)

    def __getitem__(self, item) -> Function:
        return self._population[item]

    def __setitem__(self, key, value):
        self._population[key] = value

    @property
    def population(self):
        return self._population

    @property
    def elitist(self):
        return self._elitist

    @property
    def generation(self):
        return self._generation

    def register_function(self, func: Function):
        # we only accept valid functions
        if func.score is None:
            return
        try:
            self._lock.acquire()
            # register to next_gen
            if not self.has_duplicate_function(func):
                self._next_gen_pop.append(func)

            # update: perform survival if reach the pop size
            if len(self._next_gen_pop) >= self._pop_size or (len(self._next_gen_pop) >= self._pop_size//5 and self._generation == 0):
                pop = self._population + self._next_gen_pop

                pop_elitist = pop + self._elitist
                objs = [ind.score for ind in pop_elitist]
                objs_array = np.array(objs)
                nondom_idx = NonDominatedSorting().do(objs_array, only_non_dominated_front=True)
                self._elitist = []
                for idx in nondom_idx.tolist():
                    self._elitist.append(pop_elitist[idx])

                # modified from pymoo.algorithms.moo.nsga2
                # get the objective space values and objects
                objs = [ind.score for ind in pop]
                F = np.array(objs)

                # the final indices of surviving individuals
                survivors = []

                # do the non-dominated sorting until splitting front
                fronts = NonDominatedSorting().do(F, n_stop_if_ranked=self._pop_size)

                for k, front in enumerate(fronts):

                    I = np.arange(len(front))

                    # current front sorted by crowding distance if splitting
                    if len(survivors) + len(I) > self._pop_size // 5:

                        # Define how many will be removed
                        n_remove = len(survivors) + len(front) - self._pop_size // 5

                        crowding_of_front = _crowding_distance(F[front, :])
                        I = np.argsort(-crowding_of_front)[:len(front) - n_remove]

                    # otherwise take the whole front unsorted
                    else:
                        crowding_of_front = _crowding_distance(F[front, :])

                    # save rank and crowding in the individual class
                    for j, i in enumerate(front):
                        pop[i].rank = k
                        pop[i].crowding = crowding_of_front[j]

                    # extend the survivors by all or selected individuals
                    survivors.extend(front[I])

                self._population = [pop[i] for i in survivors]
                self._next_gen_pop = []
                self._generation += 1
        except Exception as e:
            traceback.print_exc()
            return
        finally:
            self._lock.release()

    def has_duplicate_function(self, func: str | Function) -> bool:
        if func.score is None:
            return True

        for i in range(len(self._population)):
            f = self._population[i]
            if str(f) == str(func):
                if func.score[0] < f.score[0]:
                    self._population[i] = func
                    return True
                if func.score[0] == f.score[0] and func.score[1] < f.score[1]:
                    self._population[i] = func
                    return True

        for i in range(len(self._next_gen_pop)):
            f = self._next_gen_pop[i]
            if str(f) == str(func):
                if func.score[0] < f.score[0]:
                    self._next_gen_pop[i] = func
                    return True
                if func.score[0] == f.score[0] and func.score[1] < f.score[1]:
                    self._next_gen_pop[i] = func
                    return True
        return False

    def selection(self) -> Function:
        # funcs = [f for f in self._population if not math.isinf(f.score)]
        funcs = [f for f in self._population if not np.isinf(np.array(f.score)).any()]
        if len(funcs) > 1:
            parents = np.random.choice(funcs, size=2, replace=False)
            if parents[0].rank < parents[1].rank:
                return parents[0]
            elif parents[0].rank > parents[1].rank:
                return parents[1]
            elif parents[0].crowding > parents[1].crowding:
                return parents[0]
            else:
                return parents[1]
        else:
            return funcs[0]
