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

# PBC-LLM method:
from llm4ad.method.PBCLLM import PBCLLM, PBCProfiler
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
    return parser.parse_args()


def build_method(method_name, llm, llm_cluster, task, args):
    if method_name in ("mpage", "llmpfg"):
        return MPaGE(llm=llm,
                     llm_cluster=llm_cluster,
                     profiler=MPaGEProfiler(log_dir='logs/LLMPFG', log_style='complex'),
                     evaluation=task,
                     max_sample_nums=200,
                     max_generations=None,
                     pop_size=10,
                     num_samplers=1,
                     num_evaluators=1,
                     # llm_review=True
                  )

    if method_name == "pbcllm":
        timestamp = datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
        size_str = str(args.problem_size) if args.problem_size is not None else "default"
        final_log_dir = f"logs/PBCLLM/{timestamp}_{size_str}"
        return PBCLLM(llm=llm,
                      llm_cluster=llm_cluster,
                      profiler=PBCProfiler(
                           log_dir='logs/PBCLLM',
                           evaluation_name=args.problem,
                           log_style='complex',
                           final_log_dir=final_log_dir,
                      ),
                      evaluation=task,
                      max_sample_nums=80,
                      pop_size=10,
                      selection_num=3,
                      num_samplers=1,
                      num_evaluators=1,
                      behavior_novelty_weight=0.2,
                      cluster_distance=0.35,
                      elites_per_preference=2,
                    )

    raise ValueError(f"Unknown method={method_name!r}")


def main():
    args = parse_args()

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
