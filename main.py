import argparse
import os
import pytz
from datetime import datetime

# from llm4ad.tools.llm.llm_api_https import HttpsApi
# from llm4ad.tools.llm.llm_api_openai import HttpsApiOpenAI
# from llm4ad.tools.llm.llm_api_openai_cluster import HttpsApiOpenAI4Cluster
from llm4ad.tools.llm.llm_api_litellm import HttpsApiLiteLLM
from llm4ad.tools.llm.llm_api_litellm import HttpsApiLiteLLM4Cluster
# LLMPFG baseline:
from llm4ad.method.LLMPFG import MPaGE, EoHProfiler as MPaGEProfiler
from llm4ad.method.LLMPFG.resume import resume_eoh

# PBC-LLM method:
from llm4ad.method.PBCLLM import PBCLLM, PBCProfiler, resume_pbcllm
from llm4ad.task.optimization.registry import PROBLEM_CONFIGS, build_problem
from llm4ad.tools.evaluate_all_sizes import (
    evaluate_log_all_sizes,
    format_reports_table,
    write_all_size_report,
)
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
        choices=("mpage", "llmpfg", "pbcllm", "both"),
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
        help="Resume MPaGE/LLMPFG/PBCLLM from an existing log directory, or from its run_log.txt file.",
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume from the latest log directory for the selected method and problem.",
    )
    parser.add_argument(
        "--evaluate-all-sizes",
        action="store_true",
        help="After training, reevaluate the final population on all available sizes for the selected problem.",
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
    return parser.parse_args()


def _looks_like_log_dir(path):
    samples_dir = os.path.join(path, "samples")
    population_dir = os.path.join(path, "population")
    return os.path.isdir(samples_dir) and os.path.isdir(population_dir)


def _latest_resume_log_dir(method_name, problem):
    if method_name in ("mpage", "llmpfg"):
        search_roots = [
            os.path.join("logs", "LLMPFG", problem),
            os.path.join("logs", "LLMPFG"),
        ]
    elif method_name == "pbcllm":
        search_roots = [
            os.path.join("logs", "PBCLLM", problem),
            os.path.join("logs", "PBCLLM"),
        ]
    else:
        raise ValueError(f"--resume-latest is not supported for method={method_name!r}.")

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
        timestamp = datetime.now(pytz.timezone("Asia/Bangkok")).strftime("%Y%m%d_%H%M%S")
        size_str = str(args.problem_size) if args.problem_size is not None else "default"
        final_log_dir = args.resume_log_dir or os.path.join(
            "logs",
            "PBCLLM",
            args.problem,
            f"{timestamp}_{size_str}",
        )
        return PBCLLM(llm=llm,
                      llm_cluster=llm_cluster,
                      profiler=PBCProfiler(
                           log_dir=os.path.join("logs", "PBCLLM", args.problem),
                           evaluation_name=args.problem,
                           log_style='complex',
                           final_log_dir=final_log_dir,
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
                    )

    raise ValueError(f"Unknown method={method_name!r}")


def main():
    args = parse_args()
    args.resume_log_dir = resolve_resume_log_dir(args)

    llm, llm_cluster = build_llms()
    if args.method == "both":
        method_names = ["llmpfg", "pbcllm"]
    else:
        method_names = [args.method]
    post_eval_reports = []
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
            if method_name == "pbcllm":
                resume_pbcllm(method)
            else:
                resume_eoh(method)
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
                best_func = min(
                    pop.population,
                    key=lambda f: f.score[0]
                    if getattr(f, "score", None) and len(f.score) > 0
                    else float("inf"),
                )
                if getattr(best_func, "score", None) and len(best_func.score) > 0:
                    print(f"[Report] Best individual train score: {best_func.score}")
                    print(f"[Report] Best train HV: {-best_func.score[0]:.6f}")

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

        if args.evaluate_all_sizes and log_dir:
            print(f"\n{'='*60}")
            print(f"Post-evaluation on all available instance sizes for {method_name}...")
            print(f"{'='*60}")
            if method_name in {"mpage", "llmpfg", "pbcllm"}:
                report = evaluate_population_front_all_sizes(
                    log_dir,
                    method=method_name,
                    problem=args.problem,
                    top_k=args.post_eval_top_k,
                    seed=args.seed,
                    eval_seed=args.post_eval_seed,
                    timeout_seconds=args.post_eval_timeout_seconds,
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
            else:
                report = evaluate_log_all_sizes(
                    log_dir,
                    method=method_name,
                    problem=args.problem,
                    top_k=args.post_eval_top_k,
                    seed=args.seed,
                    eval_seed=args.post_eval_seed,
                    timeout_seconds=args.post_eval_timeout_seconds,
                )
                write_all_size_report(report)
                post_eval_reports.append(report)
                # Print summary immediately
                print(f"\n{'-'*60}")
                print(f"Results for {method_name} on {args.problem}")
                print(f"{'-'*60}")
                print(f"{'Size':>8} | {'Instances':>10} | {'Valid':>8} | {'Best HV':>12} | {'Mean HV':>12}")
                for item in report.get("sizes", []):
                    size = item.get("problem_size", "?")
                    n_ins = item.get("n_instance", "?")
                    valid = f"{item.get('num_valid',0)}/{item.get('num_evaluated',0)}"
                    best_hv = item.get("best_inner_hv")
                    mean_hv = item.get("mean_inner_hv")
                    print(f"{size:>8} | {n_ins:>10} | {valid:>8} | {best_hv:>12.6f} | {mean_hv:>12.6f}" if best_hv is not None else f"{size:>8} | {n_ins:>10} | {valid:>8} | {'N/A':>12} | {'N/A':>12}")
                print(f"{'-'*60}\n")

    if post_eval_reports:
        print(format_reports_table(post_eval_reports))
    if population_front_reports:
        print(format_population_front_table(population_front_reports))


if __name__ == '__main__':
    main()
