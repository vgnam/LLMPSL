import argparse
import os
import pytz
import numpy as np
from datetime import datetime

# from llm4ad.tools.llm.llm_api_https import HttpsApi
# from llm4ad.tools.llm.llm_api_openai import HttpsApiOpenAI
# from llm4ad.tools.llm.llm_api_openai_cluster import HttpsApiOpenAI4Cluster
from llm4ad.tools.llm.llm_api_litellm import HttpsApiLiteLLM
from llm4ad.tools.llm.llm_api_litellm import HttpsApiLiteLLM4Cluster
# LLMPFG baseline:
from llm4ad.method.LLMPFG import MPaGE, EoHProfiler as MPaGEProfiler
from llm4ad.method.LLMPFG.resume import resume_eoh as resume_llmpfg

# PBC-LLM method:
from llm4ad.method.PBCLLM import PBCLLM, PBCProfiler, resume_pbcllm
from llm4ad.method.eoh import EoH, EoHProfiler, resume_eoh
from llm4ad.method.funsearch import FunSearch, resume_funsearch
from llm4ad.method.funsearch.profiler import FunSearchProfiler
from llm4ad.method.reevo import ReEvo, ReEvoProfiler, resume_reevo
from llm4ad.method.meoh import MEoH, MEoHProfiler, resume_meoh
from llm4ad.method.nsga2 import NSGA2, NSGA2Profiler, resume_nsga2
from llm4ad.method.moead import MOEAD, MOEADProfiler, resume_moead
from llm4ad.task.optimization.registry import PROBLEM_CONFIGS, build_problem
from llm4ad.tools.evaluate_population_front import (
    evaluate_population_front_all_sizes,
    format_population_front_table,
    write_population_front_report,
)
from llm4ad.tools.evaluate_behavior_novelty import (
    evaluate_population_behavior_novelty,
    write_behavior_novelty_report,
)


LLM_PROFILES = {
    "gpt-4o-mini": {
        "base_url": "https://api.vectorengine.ai/v1",
        "model": "openai/gpt-4o-mini",
        "api_key_env": "LLM_API_KEY",
        "api_key_file": "secret.txt",
        "cluster_api_key_env": "LLM_CLUSTER_API_KEY",
        "cluster_api_key_file": "secret_cluster.txt",
    },
    "codestral-2508": {
        "base_url": "https://api.mistral.ai/v1",
        "model": "mistral/codestral-2508",
        "api_key_env": "MISTRAL_API_KEY",
        "api_key_file": "secret_mistral.txt",
        "cluster_api_key_env": "MISTRAL_CLUSTER_API_KEY",
        "cluster_api_key_file": "secret_mistral_cluster.txt",
    },
    "gpt-oss-120b": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": "openai/openai/gpt-oss-120b",
        "api_key_env": "NVIDIA_API_KEY",
        "api_key_file": "secret_nvidia.txt",
        "cluster_api_key_env": "NVIDIA_CLUSTER_API_KEY",
        "cluster_api_key_file": "secret_nvidia_cluster.txt",
    },
}

METHOD_LOG_LABELS = {
    "mpage": "LLMPFG",
    "llmpfg": "LLMPFG",
    "pbcllm": "PBCLLM",
    "eoh": "EoH",
    "funsearch": "FunSearch",
    "reevo": "ReEvo",
    "meoh": "MEoH",
    "nsga2": "NSGA2",
    "moead": "MOEAD",
}

SCALAR_HV_BASELINES = {"eoh", "funsearch", "reevo"}
VECTOR_QT_BASELINES = {"meoh", "nsga2", "moead"}
RESUME_HANDLERS = {
    "mpage": lambda method, _: resume_llmpfg(method),
    "llmpfg": lambda method, _: resume_llmpfg(method),
    "pbcllm": lambda method, _: resume_pbcllm(method),
    "eoh": resume_eoh,
    "funsearch": resume_funsearch,
    "reevo": resume_reevo,
    "meoh": resume_meoh,
    "nsga2": resume_nsga2,
    "moead": resume_moead,
}


class ScoreProjectionEvaluation:
    """Project the MOCO evaluator score into the baseline's training objective."""

    def __init__(self, base_evaluation, projection: str):
        if projection not in {"negative_hv", "negative_hv_runtime"}:
            raise ValueError(f"Unknown score projection: {projection}")
        self._base_evaluation = base_evaluation
        self.score_projection = projection

        self.template_program = base_evaluation.template_program
        self.task_description = base_evaluation.task_description
        self.use_numba_accelerate = base_evaluation.use_numba_accelerate
        self.use_protected_div = base_evaluation.use_protected_div
        self.protected_div_delta = base_evaluation.protected_div_delta
        self.random_seed = base_evaluation.random_seed
        self.timeout_seconds = base_evaluation.timeout_seconds
        self.exec_code = base_evaluation.exec_code
        self.safe_evaluate = base_evaluation.safe_evaluate
        self.daemon_eval_process = base_evaluation.daemon_eval_process

    def __getattr__(self, name):
        base_evaluation = self.__dict__.get("_base_evaluation")
        if base_evaluation is None:
            raise AttributeError(name)
        return getattr(base_evaluation, name)

    def evaluate_program(self, program_str: str, callable_func: callable):
        score = self._base_evaluation.evaluate_program(program_str, callable_func)
        if score is None:
            return None
        if isinstance(score, dict):
            score = score.get("legacy_score")
        if not isinstance(score, (list, tuple, np.ndarray)) or len(score) == 0:
            return None
        values = [float(value) for value in score]
        if not all(np.isfinite(values)):
            return None
        if self.score_projection == "negative_hv":
            return values[0]
        if len(values) < 2:
            return None
        return np.array([values[0], values[1]], dtype=float)


def read_api_key(env_name, file_name, fallback=None):
    value = os.getenv(env_name)
    if value:
        return value.strip()
    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            return f.readline().strip()
    if fallback is not None:
        return fallback
    raise FileNotFoundError(
        f"Missing API key. Set environment variable {env_name} or create {file_name}."
    )


def selected_llm_profile():
    provider = os.getenv("LLM_PROVIDER", "gpt-4o-mini").strip()
    if provider not in LLM_PROFILES:
        choices = ", ".join(sorted(LLM_PROFILES))
        raise ValueError(f"Unknown LLM_PROVIDER={provider!r}. Choose one of: {choices}.")
    return provider, LLM_PROFILES[provider]


def build_llms():
    provider, profile = selected_llm_profile()
    llm_api_key = read_api_key(profile["api_key_env"], profile["api_key_file"])
    llm_api_key_cluster = read_api_key(
        profile["cluster_api_key_env"],
        profile["cluster_api_key_file"],
        fallback=llm_api_key,
    )

    print(
        f"Using LLM_PROVIDER={provider} with model={profile['model']} "
        f"and base_url={profile['base_url']}"
    )
    llm = HttpsApiLiteLLM(
        base_url=profile["base_url"],
        api_key=llm_api_key,
        model=profile["model"],
        timeout=30,
        temperature=0.7,
    )
    llm_cluster = HttpsApiLiteLLM4Cluster(
        base_url=profile["base_url"],
        api_key=llm_api_key_cluster,
        model=profile["model"],
        timeout=30,
    )
    return llm, llm_cluster


def parse_args():
    parser = argparse.ArgumentParser(description="Run MPaGE heuristic search.")
    parser.add_argument(
        "--method",
        choices=tuple(METHOD_LOG_LABELS) + ("both",),
        default="mpage",
        help="Search method to run. llmpfg=MPaGE baseline, pbcllm=Pareto Behavior Coevolution. Defaults to mpage.",
    )
    parser.add_argument(
        "--problem",
        choices=sorted(PROBLEM_CONFIGS),
        default="bi_tsp",
        help="Optimization problem to run.",
    )
    parser.add_argument("--problem-size", type=int, default=None, help="Training instance size.")
    parser.add_argument("--n-instance", type=int, default=None, help="Number of training instances.")
    parser.add_argument("--seed", type=int, default=2025, help="Dataset seed.")
    parser.add_argument(
        "--max-sample-nums",
        type=int,
        default=None,
        help="Override the method sample budget. When resuming, set this higher than the saved sample count.",
    )
    parser.add_argument(
        "--resume-log-dir",
        default=None,
        help="Resume a supported method from an existing log directory, or from its run_log.txt file.",
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume from the latest log directory for the selected method and problem.",
    )
    post_eval_group = parser.add_mutually_exclusive_group()
    post_eval_group.add_argument(
        "--evaluate-all-sizes",
        action="store_true",
        help="After training, reevaluate the final population on all available sizes for the selected problem.",
    )
    post_eval_group.add_argument(
        "--evaluate-sizes",
        type=int,
        nargs="+",
        default=None,
        help="After training, reevaluate the final population only on these problem sizes.",
    )
    parser.add_argument(
        "--post-eval-top-k",
        type=int,
        default=0,
        help="0 means evaluate the whole final population; otherwise evaluate the top-k by training score.",
    )
    parser.add_argument("--post-eval-seed", type=int, default=2025)
    parser.add_argument("--post-eval-timeout-seconds", type=int, default=None)
    parser.add_argument(
        "--pbcllm-eval-seeds",
        type=int,
        nargs="+",
        default=[2025],
        help="Fixed common evaluation seeds used for every PBCLLM heuristic.",
    )
    parser.add_argument(
        "--pbcllm-parent-selection",
        choices=("complementary_behavior", "legacy_role"),
        default="complementary_behavior",
        help=(
            "PBCLLM parent selection policy. complementary_behavior selects parents by weak-region anchor, "
            "marginal population-HV complement, and behavior-trajectory complement; legacy_role uses the old "
            "best-HV/contribution/diversity roles."
        ),
    )
    parser.add_argument(
        "--pbcllm-debug-output",
        action="store_true",
        help=(
            "Print PBCLLM raw LLM responses, extracted thoughts, extracted functions, "
            "reconstructed programs, evaluation results, and PBC analysis."
        ),
    )
    return parser.parse_args()


def _looks_like_log_dir(path):
    samples_dir = os.path.join(path, "samples")
    population_dir = os.path.join(path, "population")
    return os.path.isdir(samples_dir) and os.path.isdir(population_dir)


def _latest_resume_log_dir(method_name, problem):
    label = METHOD_LOG_LABELS.get(method_name)
    if label is None:
        raise ValueError(f"--resume-latest is not supported for method={method_name!r}.")
    search_roots = [
        os.path.join("logs", label, problem),
        os.path.join("logs", label),
    ]

    candidates = []
    seen = set()
    for base_dir in search_roots:
        if not os.path.isdir(base_dir):
            continue
        for root, _, _ in os.walk(base_dir):
            if not _looks_like_log_dir(root):
                continue
            abs_root = os.path.abspath(root)
            if abs_root in seen:
                continue
            seen.add(abs_root)
            candidates.append(abs_root)
    if not candidates:
        roots = ", ".join(search_roots)
        raise FileNotFoundError(f"No resumable log directory found under: {roots}.")
    return max(candidates, key=lambda path: os.stat(path).st_mtime)


def resolve_resume_log_dir(args):
    if args.resume_log_dir and args.resume_latest:
        raise ValueError("Use either --resume-log-dir or --resume-latest, not both.")
    if not args.resume_log_dir and not args.resume_latest:
        return None
    if args.method == "both":
        raise ValueError("Resume can be used with one method at a time, not --method both.")
    if args.method not in RESUME_HANDLERS:
        raise ValueError(f"Resume is not supported for method={args.method!r}.")

    log_dir = _latest_resume_log_dir(args.method, args.problem) if args.resume_latest else args.resume_log_dir
    log_dir = os.path.abspath(log_dir)
    if os.path.isfile(log_dir):
        log_dir = os.path.dirname(log_dir)
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(f"Resume log directory does not exist: {log_dir}")
    for child in ("samples", "population"):
        child_path = os.path.join(log_dir, child)
        if not os.path.isdir(child_path):
            raise FileNotFoundError(f"Resume log directory is missing {child_path}")
    return log_dir


def _timestamped_log_dir(method_name, problem, problem_size, resume_log_dir=None):
    if resume_log_dir:
        return resume_log_dir
    timestamp = datetime.now(pytz.timezone("Asia/Bangkok")).strftime("%Y%m%d_%H%M%S")
    size_str = str(problem_size) if problem_size is not None else "default"
    return os.path.join(
        "logs",
        METHOD_LOG_LABELS[method_name],
        problem,
        f"{timestamp}_{size_str}",
    )


def _profiler_kwargs(method_name, args):
    label = METHOD_LOG_LABELS[method_name]
    final_log_dir = _timestamped_log_dir(method_name, args.problem, args.problem_size, args.resume_log_dir)
    return {
        "log_dir": os.path.join("logs", label, args.problem),
        "evaluation_name": args.problem,
        "method_name": label,
        "log_style": "complex",
        "final_log_dir": final_log_dir,
    }


def _report_score_vector(score):
    if score is None:
        return []
    if isinstance(score, np.ndarray):
        score = score.tolist()
    elif not isinstance(score, (list, tuple)):
        score = [score]
    try:
        values = [float(value) for value in score]
    except (TypeError, ValueError):
        return []
    if not values or not all(np.isfinite(values)):
        return []
    return values


def build_method(method_name, llm, llm_cluster, task, args):
    if method_name in ("mpage", "llmpfg"):
        profiler_kwargs = {
            "log_dir": os.path.join("logs", "LLMPFG", args.problem),
            "evaluation_name": args.problem,
            "log_style": "complex",
        }
        if args.resume_log_dir:
            profiler_kwargs["final_log_dir"] = args.resume_log_dir
        return MPaGE(llm=llm,
                     llm_cluster=llm_cluster,
                     profiler=MPaGEProfiler(**profiler_kwargs),
                     evaluation=task,
                     max_sample_nums=args.max_sample_nums if args.max_sample_nums is not None else 200,
                     max_generations=None,
                     pop_size=10,
                     num_samplers=1,
                     num_evaluators=1,
                     resume_mode=bool(args.resume_log_dir),
                     # llm_review=True
                  )

    if method_name == "pbcllm":
        return PBCLLM(llm=llm,
                      llm_cluster=llm_cluster,
                      profiler=PBCProfiler(
                           log_dir=os.path.join("logs", "PBCLLM", args.problem),
                           evaluation_name=args.problem,
                           log_style='complex',
                           final_log_dir=_timestamped_log_dir(method_name, args.problem, args.problem_size, args.resume_log_dir),
                      ),
                      evaluation=task,
                      max_sample_nums=args.max_sample_nums if args.max_sample_nums is not None else 200,
                      pop_size=10,
                      selection_num=3,
                      num_samplers=1,
                      num_evaluators=1,
                      evaluation_seeds=args.pbcllm_eval_seeds,
                      behavior_novelty_weight=0.2,
                      cluster_distance=0.35,
                      elites_per_preference=2,
                      parent_selection_strategy=args.pbcllm_parent_selection,
                      debug_output=args.pbcllm_debug_output,
                    )

    max_samples = args.max_sample_nums if args.max_sample_nums is not None else 200
    if method_name in SCALAR_HV_BASELINES:
        task = ScoreProjectionEvaluation(task, "negative_hv")
    elif method_name in VECTOR_QT_BASELINES:
        task = ScoreProjectionEvaluation(task, "negative_hv_runtime")

    if method_name == "eoh":
        return EoH(llm=llm,
                   profiler=EoHProfiler(**_profiler_kwargs(method_name, args)),
                   evaluation=task,
                   max_sample_nums=max_samples,
                   max_generations=None,
                   pop_size=10,
                   num_samplers=1,
                   num_evaluators=1,
                   max_evaluation_retries=1)

    if method_name == "funsearch":
        return FunSearch(llm=llm,
                         profiler=FunSearchProfiler(**_profiler_kwargs(method_name, args)),
                         evaluation=task,
                         max_sample_nums=max_samples,
                         samples_per_prompt=1,
                         num_samplers=1,
                         num_evaluators=1)

    if method_name == "reevo":
        return ReEvo(llm=llm,
                     profiler=ReEvoProfiler(**_profiler_kwargs(method_name, args)),
                     evaluation=task,
                     max_sample_nums=max_samples,
                     pop_size=10,
                     num_samplers=1,
                     num_evaluators=1)

    if method_name == "meoh":
        return MEoH(llm=llm,
                    profiler=MEoHProfiler(**_profiler_kwargs(method_name, args)),
                    evaluation=task,
                    max_sample_nums=max_samples,
                    max_generations=None,
                    pop_size=80,
                    selection_num=2,
                    num_samplers=1,
                    num_evaluators=1,
                    num_objs=2)

    if method_name == "nsga2":
        return NSGA2(llm=llm,
                     profiler=NSGA2Profiler(**_profiler_kwargs(method_name, args)),
                     evaluation=task,
                     max_sample_nums=max_samples,
                     max_generations=None,
                     pop_size=80,
                     selection_num=5,
                     num_samplers=1,
                     num_evaluators=1,
                     num_objs=2)

    if method_name == "moead":
        return MOEAD(llm=llm,
                     profiler=MOEADProfiler(**_profiler_kwargs(method_name, args)),
                     evaluation=task,
                     max_sample_nums=max_samples,
                     max_generations=None,
                     pop_size=80,
                     selection_num=5,
                     num_samplers=1,
                     num_evaluators=1,
                     num_objs=2)

    raise ValueError(f"Unknown method={method_name!r}")


def main():
    args = parse_args()
    args.resume_log_dir = resolve_resume_log_dir(args)

    llm, llm_cluster = build_llms()
    if args.method == "both":
        method_names = ["llmpfg", "pbcllm"]
    else:
        method_names = [args.method]
    population_front_reports = []

    for method_name in method_names:
        task = build_problem(
            args.problem,
            n_instance=args.n_instance,
            problem_size=args.problem_size,
            seed=args.seed,
        )
        method = build_method(method_name, llm, llm_cluster, task, args)
        print(f"Using method={method_name}, problem={args.problem}")
        if args.resume_log_dir:
            print(f"Resuming {method_name} from {args.resume_log_dir}")
            RESUME_HANDLERS[method_name](method, args.resume_log_dir)
        method.run()

        # Report training summary from final population
        pop = getattr(method, "_population", None)
        if pop and pop.population:
            if method_name == "pbcllm":
                preference_matrix = [
                    f.score for f in pop.population
                    if getattr(f, "score", None)
                ]
                population_hv = max(
                    (
                        float((getattr(f, "pbc", None) or {}).get("population_hv", 0.0))
                        for f in pop.population
                    ),
                    default=0.0,
                )
                best_individual_hv = max(
                    (
                        float((getattr(f, "pbc", None) or {}).get("individual_hv", 0.0))
                        for f in pop.population
                    ),
                    default=0.0,
                )
                best_contribution = max(
                    (
                        float((getattr(f, "pbc", None) or {}).get("population_hv_contribution", 0.0))
                        for f in pop.population
                    ),
                    default=0.0,
                )
                diversities = [
                    float((getattr(f, "pbc", None) or {}).get("behavior_diversity", 0.0))
                    for f in pop.population
                ]
                print(f"[Report] Population-front train HV: {population_hv:.6f}")
                print(f"[Report] Best individual train HV: {best_individual_hv:.6f}")
                print(f"[Report] Best train HV contribution: {best_contribution:.6f}")
                if diversities:
                    print(f"[Report] Mean behavior diversity: {sum(diversities) / len(diversities):.6f}")
                if preference_matrix:
                    best_by_preference = [min(col) for col in zip(*preference_matrix)]
                    print(f"[Report] Best score vector values: {best_by_preference}")
            else:
                valid_scores = [
                    (func, _report_score_vector(getattr(func, "score", None)))
                    for func in pop.population
                ]
                valid_scores = [(func, score) for func, score in valid_scores if score]
                if valid_scores:
                    best_func, best_score = min(valid_scores, key=lambda item: item[1][0])
                    print(f"[Report] Best individual train score: {getattr(best_func, 'score', best_score)}")
                    print(f"[Report] Best train HV: {-best_score[0]:.6f}")

        profiler = getattr(method, "_profiler", None)
        log_dir = getattr(profiler, "_log_dir", None)
        if method_name in {"mpage", "llmpfg"} and log_dir and pop and pop.population:
            behavior_report = evaluate_population_behavior_novelty(
                pop.population,
                task,
                method=method_name,
                problem=args.problem,
                log_dir=log_dir,
                eval_seed=args.post_eval_seed,
            )
            write_behavior_novelty_report(behavior_report)

        if (args.evaluate_all_sizes or args.evaluate_sizes) and log_dir:
            print(f"\n{'='*60}")
            if args.evaluate_sizes:
                print(f"Post-evaluation on problem sizes {args.evaluate_sizes} for {method_name}...")
            else:
                print(f"Post-evaluation on all available instance sizes for {method_name}...")
            print(f"{'='*60}")
            report = evaluate_population_front_all_sizes(
                log_dir,
                method=method_name,
                problem=args.problem,
                top_k=args.post_eval_top_k,
                seed=args.seed,
                eval_seed=args.post_eval_seed,
                timeout_seconds=args.post_eval_timeout_seconds,
                problem_sizes=args.evaluate_sizes,
            )
            write_population_front_report(report)
            population_front_reports.append(report)
            print(f"\n{'-'*60}")
            print(f"Population-front results for {method_name} on {args.problem}")
            print(f"{'-'*60}")
            print(
                f"{'Size':>8} | {'Instances':>10} | {'Valid':>8} | "
                f"{'Pop HV':>12} | {'HV std':>10} | {'Points':>10}"
            )
            for item in report.get("sizes", []):
                size = item.get("problem_size", "?")
                n_ins = item.get("n_instance", "?")
                valid = f"{item.get('num_valid',0)}/{item.get('num_evaluated',0)}"
                pop_hv = item.get("population_front_hv")
                hv_std = item.get("population_front_hv_std")
                points = item.get("mean_population_front_points")
                if pop_hv is None:
                    print(
                        f"{size:>8} | {n_ins:>10} | {valid:>8} | "
                        f"{'N/A':>12} | {'N/A':>10} | {'N/A':>10}"
                    )
                else:
                    print(
                        f"{size:>8} | {n_ins:>10} | {valid:>8} | "
                        f"{pop_hv:>12.6f} | {hv_std:>10.6f} | {points:>10.2f}"
                    )
            print(f"{'-'*60}\n")

    if population_front_reports:
        print(format_population_front_table(population_front_reports))


if __name__ == '__main__':
    main()
