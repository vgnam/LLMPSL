import argparse
import os

# from llm4ad.tools.llm.llm_api_https import HttpsApi
# from llm4ad.tools.llm.llm_api_openai import HttpsApiOpenAI
# from llm4ad.tools.llm.llm_api_openai_cluster import HttpsApiOpenAI4Cluster
from llm4ad.tools.llm.llm_api_litellm import HttpsApiLiteLLM
from llm4ad.tools.llm.llm_api_litellm import HttpsApiLiteLLM4Cluster
# LLMPFG baseline:
from llm4ad.method.LLMPFG import MPaGE, EoHProfiler as MPaGEProfiler

# LLMPSL method:
from llm4ad.method.LLMPSL import LLMPSL, EoHProfiler as LLMPSLProfiler

# If you want to run the bi_tsp_semo example, uncomment the following line:
from llm4ad.task.optimization.bi_tsp_semo import BITSPEvaluation as ProblemEvaluation

# If you want to run the bi_tsp_semo example, uncomment the following line:
# from llm4ad.task.optimization.tri_tsp_semo import TRITSPEvaluation as ProblemEvaluation

# If you want to run the bi_cvrp example, uncomment the following line:
# from llm4ad.task.optimization.bi_cvrp import BICVRPEvaluation as ProblemEvaluation

# If you want to run the bi_kp example, uncomment the following line:
# from llm4ad.task.optimization.bi_kp import BIKPEvaluation as ProblemEvaluation


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
    parser = argparse.ArgumentParser(description="Run MPaGE or LLMPSL heuristic search.")
    parser.add_argument(
        "--method",
        choices=("llmpsl", "mpage"),
        default="llmpsl",
        help="Search method to run. Defaults to llmpsl.",
    )
    return parser.parse_args()


def build_method(method_name, llm, llm_cluster, task):
    if method_name == "mpage":
        return MPaGE(llm=llm,
                     llm_cluster=llm_cluster,
                     profiler=MPaGEProfiler(log_dir='logs/MPaGE', log_style='complex'),
                     evaluation=task,
                     max_sample_nums=200,
                     max_generations=None,
                     pop_size=10,
                     num_samplers=1,
                     num_evaluators=1,
                     # llm_review=True
                  )

    return LLMPSL(llm=llm,
                  llm_cluster=llm_cluster,
                  profiler=LLMPSLProfiler(log_dir='logs/LLMPSL', log_style='complex'),
                  evaluation=task,
                  max_sample_nums=200,
                  max_generations=None,
                  pop_size=10,
                  num_samplers=1,
                  num_evaluators=1,
                  objective_num=3,
                  novelty_k=8,
                  novelty_threshold=0.9,
                  prefer_codebleu=True,
                  tchebycheff_rho=0.05,
                  tchebycheff_set_size=3,
                  smooth_set_scalarization=True,
                  smooth_mu=0.05,
                  use_psl_interpolation=True,
                  use_novelty_repair=False,
                  # llm_review=True
               )


def main():
    args = parse_args()

    llm, llm_cluster = build_llms()
    task = ProblemEvaluation()
    method = build_method(args.method, llm, llm_cluster, task)
    print(f"Using method={args.method}")

    method.run()


if __name__ == '__main__':
    main()
