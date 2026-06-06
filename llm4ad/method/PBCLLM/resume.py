from __future__ import annotations

import copy
import json
import os
import shutil

from ...base import TextFunctionProgramConverter as tfpc
from .population import analyze_mo_result
from .profiler import PBCProfiler


_FULL_PBC_FIELDS = (
    "objective_num",
    "fronts",
    "pbt",
    "preference_performance",
    "evaluation_seeds",
    "instances_per_seed",
)


def _order_from_name(name: str, prefix: str) -> int | None:
    if not name.startswith(prefix) or not name.endswith(".json"):
        return None
    try:
        return int(name[len(prefix):-5].split("~")[0])
    except ValueError:
        return None


def _load_json_records(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _get_latest_pop_json(log_path: str) -> tuple[str, int] | None:
    path = os.path.join(log_path, "population")
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Missing population directory: {path}")
    files = [
        (order, name)
        for name in os.listdir(path)
        if (order := _order_from_name(name, "pop_")) is not None
    ]
    if not files:
        return None
    max_order, name = max(files)
    return os.path.join(path, name), max_order


def _sample_order(record: dict, fallback: int) -> int:
    try:
        return int(record.get("sample_order", fallback))
    except (TypeError, ValueError):
        return fallback


def _get_sample_records(log_path: str) -> list[dict]:
    path = os.path.join(log_path, "samples")
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Missing samples directory: {path}")

    files = []
    for name in os.listdir(path):
        if name == "samples_best.json":
            continue
        order = _order_from_name(name, "samples_")
        if order is not None:
            files.append((order, name))
    files.sort()

    records = []
    for _, name in files:
        records.extend(_load_json_records(os.path.join(path, name)))
    records.sort(key=lambda item: _sample_order(item, 0))
    return records


def _validate_sample_orders(records: list[dict]) -> int:
    orders = [_sample_order(record, idx) for idx, record in enumerate(records, start=1)]
    if not orders:
        return 0
    if orders != list(range(1, max(orders) + 1)):
        raise ValueError(
            "PBCLLM sample history must contain unique contiguous sample_order values "
            "to resume safely."
        )
    return orders[-1]


def _has_full_pbc_state(record: dict) -> bool:
    pbc = record.get("pbc")
    return isinstance(pbc, dict) and all(pbc.get(field) is not None for field in _FULL_PBC_FIELDS)


def _validate_full_pbc_state(record: dict, pbcllm, source: str) -> None:
    pbc = record["pbc"]
    if int(pbc["objective_num"]) != pbcllm._objective_num:
        raise ValueError(
            f"PBCLLM log objective_num={pbc['objective_num']} in {source} does not match "
            f"the current objective_num={pbcllm._objective_num}."
        )
    saved_seeds = tuple(int(seed) for seed in pbc["evaluation_seeds"])
    current_seeds = tuple(int(seed) for seed in pbcllm._evaluation_seeds)
    if saved_seeds != current_seeds:
        raise ValueError(
            f"PBCLLM log evaluation seeds {saved_seeds} in {source} do not match "
            f"the current evaluation seeds {current_seeds}."
        )


def _rebuild_pbc(func, pbcllm, source: str) -> dict:
    program = tfpc.function_to_program(func, pbcllm._template_program)
    if program is None:
        raise ValueError(f"Could not create a program while rebuilding PBCLLM state from {source}.")

    result, eval_time = pbcllm._evaluate_program_across_seeds(program)
    pbc = analyze_mo_result(
        result,
        pbcllm._preference_vectors,
        rho=pbcllm._rho,
        normalization_ideal=pbcllm._normalization_ideal,
        normalization_nadir=pbcllm._normalization_nadir,
        hv_ref_point=pbcllm._hv_ref_point,
    )
    if pbc is None:
        raise ValueError(f"Could not rebuild valid PBCLLM behavior state from {source}.")
    expected_front_count = sum((result or {}).get("instances_per_seed", []))
    if pbc.get("front_count") != expected_front_count:
        raise ValueError(f"Incomplete PBCLLM behavior state rebuilt from {source}.")

    func.evaluate_time = eval_time
    func.score = [-pbc["individual_hv"], pbc["coverage_loss"]]
    return pbc


def _restore_function(record: dict, pbcllm, source: str, pbc_cache: dict[str, dict]):
    func_text = record.get("function")
    if not func_text:
        raise ValueError(f"Missing function text in PBCLLM resume record: {source}")
    func = tfpc.text_to_function(func_text)
    if func is None:
        raise ValueError(f"Could not parse function in PBCLLM resume record: {source}")

    rebuilt = False
    if func_text in pbc_cache:
        func.pbc = copy.deepcopy(pbc_cache[func_text])
    elif _has_full_pbc_state(record):
        _validate_full_pbc_state(record, pbcllm, source)
        func.pbc = copy.deepcopy(record["pbc"])
        pbc_cache[func_text] = copy.deepcopy(func.pbc)
    else:
        print(f"RESUME PBCLLM: Re-evaluating old log record from {source}.", flush=True)
        func.pbc = _rebuild_pbc(func, pbcllm, source)
        pbc_cache[func_text] = copy.deepcopy(func.pbc)
        rebuilt = True

    func.algorithm = record.get("algorithm")
    if not rebuilt:
        func.score = record.get("score") or [
            -func.pbc["individual_hv"],
            func.pbc["coverage_loss"],
        ]
    return func, rebuilt


def _write_population_checkpoint(path: str, pop, profiler: PBCProfiler) -> None:
    records = [profiler._record_payload(func) for func in pop.population]
    backup_path = f"{path[:-5]}.pre_resume.json"
    if os.path.isfile(path) and not os.path.exists(backup_path):
        shutil.copy2(path, backup_path)
        print(f"RESUME PBCLLM: Saved original checkpoint backup {backup_path}.", flush=True)
    temp_path = f"{path}.resume.tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)
    os.replace(temp_path, path)


def _resume_population(log_path: str, pbcllm, pbc_cache: dict[str, dict]) -> tuple[int, int, int]:
    latest = _get_latest_pop_json(log_path)
    pop = pbcllm._population
    pop._population = []
    pop._next_gen_pop = []

    if latest is None:
        pop._generation = 0
        pop._refresh_population_metrics()
        print("RESUME PBCLLM: No population checkpoint; restoring from generation 0.", flush=True)
        return 0, 0, 0

    path, max_gen = latest
    records = _load_json_records(path)
    if len(records) != pop._pop_size:
        raise ValueError(
            f"PBCLLM checkpoint population size {len(records)} does not match "
            f"the current pop_size={pop._pop_size}."
        )

    pop._generation = max_gen
    rebuilt_count = 0
    for idx, record in enumerate(records, start=1):
        func, rebuilt = _restore_function(record, pbcllm, f"{path} record {idx}", pbc_cache)
        pop._population.append(func)
        rebuilt_count += int(rebuilt)
    pop._refresh_population_metrics()

    if rebuilt_count:
        _write_population_checkpoint(path, pop, pbcllm._profiler)
        print(f"RESUME PBCLLM: Upgraded existing checkpoint {path}.", flush=True)
    print(f"RESUME PBCLLM: Restored generation={max_gen}.", flush=True)
    return max_gen, max_gen * pop._pop_size, rebuilt_count


def _resume_pending_samples(
    records: list[dict],
    pbcllm,
    checkpoint_sample_order: int,
    pbc_cache: dict[str, dict],
) -> tuple[int, int]:
    restored = 0
    rebuilt_count = 0
    profiler = pbcllm._profiler
    for idx, record in enumerate(records, start=1):
        order = _sample_order(record, idx)
        if order <= checkpoint_sample_order:
            continue
        old_generation = pbcllm._population.generation
        func, rebuilt = _restore_function(record, pbcllm, f"sample_order={order}", pbc_cache)
        pbcllm._population.register_function(func)
        restored += 1
        rebuilt_count += int(rebuilt)
        if pbcllm._population.generation > old_generation:
            profiler.register_population(pbcllm._population)
    return restored, rebuilt_count


def resume_pbcllm(pbcllm):
    if not isinstance(pbcllm._profiler, PBCProfiler):
        raise ValueError("PBCLLM resume requires a PBCProfiler.")

    log_path = pbcllm._profiler._log_dir
    pbc_cache: dict[str, dict] = {}
    max_gen, checkpoint_sample_order, rebuilt_population = _resume_population(
        log_path,
        pbcllm,
        pbc_cache,
    )

    records = _get_sample_records(log_path)
    max_sample_order = _validate_sample_orders(records)
    if max_sample_order < checkpoint_sample_order:
        raise ValueError(
            f"PBCLLM sample history ends at {max_sample_order}, before the latest "
            f"population checkpoint boundary {checkpoint_sample_order}."
        )

    pbcllm._profiler._cur_gen = max_gen
    pending_count, rebuilt_pending = _resume_pending_samples(
        records,
        pbcllm,
        checkpoint_sample_order,
        pbc_cache,
    )
    pbcllm._profiler.__class__._num_samples = max_sample_order
    pbcllm._profiler._cur_gen = pbcllm._population.generation
    pbcllm._tot_sample_nums = max_sample_order

    print(
        f"RESUME PBCLLM: Restored {pending_count} pending functions; "
        f"re-evaluated {rebuilt_population + rebuilt_pending} old-log functions; "
        f"sample_order={max_sample_order}.",
        flush=True,
    )
    if pbcllm._max_sample_nums is not None and pbcllm._max_sample_nums <= max_sample_order:
        print(
            f"RESUME PBCLLM: max_sample_nums={pbcllm._max_sample_nums} is not greater than "
            f"the saved sample order {max_sample_order}; no new samples will be generated.",
            flush=True,
        )
