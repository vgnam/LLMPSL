from __future__ import annotations

import json
import os.path

from .eoh import MPaGE
from .profiler import EoHProfiler
from .population import Population
from ...base import TextFunctionProgramConverter as tfpc


def _order_from_name(name: str, prefix: str) -> int | None:
    if not name.startswith(prefix) or not name.endswith('.json'):
        return None
    try:
        return int(name[len(prefix):-5].split('~')[0])
    except ValueError:
        return None


def _load_json_records(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _get_latest_pop_json(log_path: str):
    path = os.path.join(log_path, 'population')
    if not os.path.isdir(path):
        raise FileNotFoundError(f'Missing population directory: {path}')
    orders = [
        order for p in os.listdir(path)
        if (order := _order_from_name(p, 'pop_')) is not None
    ]
    if not orders:
        raise FileNotFoundError(f'No population checkpoint found in {path}')
    max_o = max(orders)
    return os.path.join(path, f'pop_{max_o}.json'), max_o


def _get_sample_records(log_path: str):
    path = os.path.join(log_path, 'samples')
    if not os.path.isdir(path):
        raise FileNotFoundError(f'Missing samples directory: {path}')

    files = []
    for name in os.listdir(path):
        if name == 'samples_best.json':
            continue
        order = _order_from_name(name, 'samples_')
        if order is not None:
            files.append((order, name))
    files.sort()

    records = []
    for _, name in files:
        file_name = os.path.join(path, name)
        records.extend(_load_json_records(file_name))
    records.sort(key=lambda item: int(item.get('sample_order', 0) or 0))
    return records


def _get_all_samples_and_scores(log_path: str):
    records = _get_sample_records(log_path)
    all_func = []
    all_score = []
    max_o = 0
    for idx, sample in enumerate(records, start=1):
        sample_order = sample.get('sample_order', idx)
        try:
            max_o = max(max_o, int(sample_order))
        except (TypeError, ValueError):
            max_o = max(max_o, idx)
        all_func.append(sample.get('function'))
        score = sample.get('score')
        all_score.append(score if score else float('-inf'))
    return all_func, all_score, max_o


def _resume_pop(log_path: str, pop_size) -> Population:
    path, max_gen = _get_latest_pop_json(log_path)
    print(f'RESUME EoH: Generations: {max_gen}.', flush=True)
    data = _load_json_records(path)
    pop = Population(pop_size=pop_size)
    for d in data:
        func = d['function']
        func = tfpc.text_to_function(func)
        if func is None:
            continue
        score = d['score']
        algorithm = d.get('algorithm')
        func.score = score
        func.algorithm = algorithm
        pop.register_function(func)
    if not pop.population:
        raise ValueError(f'Could not restore any valid functions from {path}')
    pop._generation = max_gen
    return pop


def _resume_pf(log_path: str, pf: EoHProfiler):
    _, db_max_order = _get_latest_pop_json(log_path)
    _, _, sample_max_order = _get_all_samples_and_scores(log_path)
    print(f'RESUME EoH: Sample order: {sample_max_order}.', flush=True)
    pf.__class__._prog_db_order = db_max_order
    pf.__class__._num_samples = sample_max_order
    pf.__class__._cur_gen = db_max_order


def resume_eoh(eoh: MPaGE):
    eoh._resume_mode = True
    pf = eoh._profiler
    log_path = pf._log_dir
    # resume program database
    pop = _resume_pop(log_path, eoh._pop_size)
    eoh._population = pop
    # resume profiler
    _resume_pf(log_path, pf)
    # resume eoh
    _, _, sample_max_order = _get_all_samples_and_scores(log_path)
    eoh._tot_sample_nums = sample_max_order
    if eoh._max_sample_nums is not None and eoh._max_sample_nums <= sample_max_order:
        print(
            f'RESUME EoH: max_sample_nums={eoh._max_sample_nums} is not greater than '
            f'the saved sample order {sample_max_order}; no new samples will be generated.',
            flush=True,
        )
