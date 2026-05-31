from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path


@dataclass(frozen=True)
class ProblemConfig:
    name: str
    aliases: tuple[str, ...]
    evaluation_module: str
    evaluation_class: str
    default_n_instance: int
    default_problem_size: int
    size_label: str
    data_dir: Path


_OPT_DIR = Path(__file__).resolve().parent

PROBLEM_CONFIGS: dict[str, ProblemConfig] = {
    "bi_tsp": ProblemConfig(
        name="bi_tsp",
        aliases=("bi_tsp", "bi_tsp_semo", "tsp2"),
        evaluation_module="llm4ad.task.optimization.bi_tsp_semo",
        evaluation_class="BITSPEvaluation",
        default_n_instance=10,
        default_problem_size=20,
        size_label="cities",
        data_dir=_OPT_DIR / "bi_tsp_semo" / "data",
    ),
    "tri_tsp": ProblemConfig(
        name="tri_tsp",
        aliases=("tri_tsp", "tri_tsp_semo", "tsp3"),
        evaluation_module="llm4ad.task.optimization.tri_tsp_semo",
        evaluation_class="TRITSPEvaluation",
        default_n_instance=10,
        default_problem_size=20,
        size_label="cities",
        data_dir=_OPT_DIR / "tri_tsp_semo" / "data",
    ),
    "bi_cvrp": ProblemConfig(
        name="bi_cvrp",
        aliases=("bi_cvrp", "cvrp"),
        evaluation_module="llm4ad.task.optimization.bi_cvrp",
        evaluation_class="BICVRPEvaluation",
        default_n_instance=10,
        default_problem_size=20,
        size_label="customers",
        data_dir=_OPT_DIR / "bi_cvrp" / "data",
    ),
    "bi_kp": ProblemConfig(
        name="bi_kp",
        aliases=("bi_kp", "kp"),
        evaluation_module="llm4ad.task.optimization.bi_kp",
        evaluation_class="BIKPEvaluation",
        default_n_instance=10,
        default_problem_size=20,
        size_label="items",
        data_dir=_OPT_DIR / "bi_kp" / "data",
    ),
}


def normalize_problem_name(problem: str) -> str:
    problem = problem.strip().lower()
    for name, config in PROBLEM_CONFIGS.items():
        if problem in config.aliases:
            return name
    choices = ", ".join(sorted(PROBLEM_CONFIGS))
    raise ValueError(f"Unknown problem={problem!r}. Choose one of: {choices}.")


def get_problem_config(problem: str) -> ProblemConfig:
    return PROBLEM_CONFIGS[normalize_problem_name(problem)]


def build_problem(
    problem: str,
    *,
    n_instance: int | None = None,
    problem_size: int | None = None,
    seed: int = 2025,
    eval_seed: int | None = None,
    timeout_seconds: int | None = None,
):
    config = get_problem_config(problem)
    module = import_module(config.evaluation_module)
    evaluation_cls = getattr(module, config.evaluation_class)
    kwargs = {
        "n_instance": config.default_n_instance if n_instance is None else n_instance,
        "problem_size": config.default_problem_size if problem_size is None else problem_size,
        "seed": seed,
        "eval_seed": eval_seed,
        "data_dir": str(config.data_dir),
    }
    if timeout_seconds is not None:
        kwargs["timeout_seconds"] = timeout_seconds
    return evaluation_cls(**kwargs)
