from __future__ import annotations

import json
import re
from pathlib import Path

from ..base import TextFunctionProgramConverter as tfpc


def _numeric_suffix(path: Path, prefix: str) -> int | None:
    match = re.fullmatch(rf"{re.escape(prefix)}_(\d+)\.json", path.name)
    return int(match.group(1)) if match else None


def _load_json_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def load_sample_records(log_path: str | Path) -> list[dict]:
    samples_dir = Path(log_path) / "samples"
    if not samples_dir.is_dir():
        raise FileNotFoundError(f"Missing samples directory: {samples_dir}")

    records = []
    for path in samples_dir.glob("samples_*.json"):
        if path.name == "samples_best.json":
            continue
        records.extend(_load_json_records(path))

    def sample_order(record):
        try:
            return int(record["sample_order"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Resume sample record is missing a valid sample_order.") from exc

    records.sort(key=sample_order)
    orders = [sample_order(record) for record in records]
    if orders and orders != list(range(1, orders[-1] + 1)):
        raise ValueError(
            "Resume sample history must contain unique contiguous sample_order values."
        )
    return records


def _function_from_record(record: dict):
    function_text = record.get("function")
    if not isinstance(function_text, str) or not function_text.strip():
        return None
    function = tfpc.text_to_function(function_text)
    if function is None:
        return None
    function.score = record.get("score")
    function.algorithm = record.get("algorithm")
    return function


def _latest_checkpoint_generation(log_path: str | Path) -> int:
    population_dir = Path(log_path) / "population"
    if not population_dir.is_dir():
        return 0
    generations = [
        generation
        for path in population_dir.glob("pop_*.json")
        if (generation := _numeric_suffix(path, "pop")) is not None
    ]
    return max(generations, default=0)


def _set_resume_state(method, sample_order: int) -> None:
    method._resume_mode = True
    method._tot_sample_nums = sample_order
    profiler = method._profiler
    if profiler is None:
        return
    try:
        profiler.__class__._num_samples = sample_order
    except (AttributeError, TypeError):
        profiler._num_samples = sample_order
    if hasattr(profiler, "_cur_gen") and hasattr(method, "_population"):
        profiler._cur_gen = method._population.generation


def _budget_exhausted(method, sample_order: int) -> bool:
    max_samples = getattr(method, "_max_sample_nums", None)
    exhausted = max_samples is not None and max_samples <= sample_order
    if exhausted:
        print(
            f"RESUME: max_sample_nums={max_samples} is not greater than "
            f"the saved sample order {sample_order}; no new samples will be generated.",
            flush=True,
        )
    return exhausted


def resume_population_method(method, log_path: str | Path | None = None, *, label: str) -> None:
    log_path = Path(log_path or method._profiler._log_dir)
    records = load_sample_records(log_path)
    population_cls = method._population.__class__
    population = population_cls(pop_size=method._pop_size)

    restored = 0
    for record in records:
        function = _function_from_record(record)
        if function is None:
            continue
        population.register_function(function)
        restored += 1

    checkpoint_generation = _latest_checkpoint_generation(log_path)
    population._generation = max(population.generation, checkpoint_generation)
    method._population = population

    sample_order = int(records[-1]["sample_order"]) if records else 0
    _set_resume_state(method, sample_order)
    budget_exhausted = _budget_exhausted(method, sample_order)
    if not population.population and not budget_exhausted:
        method._resume_mode = False
    print(
        f"RESUME {label}: restored {restored} functions; "
        f"generation={population.generation}; sample_order={sample_order}.",
        flush=True,
    )


def resume_funsearch_method(method, log_path: str | Path | None = None) -> None:
    from .funsearch.programs_database import ProgramsDatabase

    log_path = Path(log_path or method._profiler._log_dir)
    records = load_sample_records(log_path)
    database = ProgramsDatabase(
        method.db_config,
        method._template_program,
        method._function_to_evolve_name,
    )

    restored = 0
    for record in records:
        function = _function_from_record(record)
        score = record.get("score")
        if function is None or score is None:
            continue
        island_id = record.get("island_id")
        try:
            island_id = None if island_id is None else int(island_id)
        except (TypeError, ValueError):
            island_id = None
        if island_id is not None and not 0 <= island_id < len(database.islands):
            island_id = None
        database.register_function(function, island_id=island_id, score=score)
        restored += 1

    method._database = database
    sample_order = int(records[-1]["sample_order"]) if records else 0
    _set_resume_state(method, sample_order)
    _budget_exhausted(method, sample_order)

    profiler = method._profiler
    if profiler is not None and hasattr(profiler, "_prog_db_order"):
        prog_db_dir = log_path / "prog_db"
        orders = [
            order
            for path in prog_db_dir.glob("db_*.json")
            if (order := _numeric_suffix(path, "db")) is not None
        ] if prog_db_dir.is_dir() else []
        profiler._prog_db_order = max(orders, default=0)

    print(
        f"RESUME FunSearch: restored {restored} functions; sample_order={sample_order}.",
        flush=True,
    )
