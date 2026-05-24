import os

# from llm4ad.tools.llm.llm_api_https import HttpsApi
# from llm4ad.tools.llm.llm_api_openai import HttpsApiOpenAI
# from llm4ad.tools.llm.llm_api_openai_cluster import HttpsApiOpenAI4Cluster
from llm4ad.tools.llm.llm_api_litellm import HttpsApiLiteLLM
from llm4ad.tools.llm.llm_api_litellm import HttpsApiLiteLLM4Cluster
# LLMPFG baseline:
# from llm4ad.method.LLMPFG import MPaGE
# from llm4ad.method.LLMPFG import EoHProfiler

# LLMPSL method:
from llm4ad.method.LLMPSL import LLMPSL
from llm4ad.method.LLMPSL import EoHProfiler

# If you want to run the bi_tsp_semo example, uncomment the following line:
from llm4ad.task.optimization.bi_tsp_semo import BITSPEvaluation as ProblemEvaluation

# If you want to run the bi_tsp_semo example, uncomment the following line:
# from llm4ad.task.optimization.tri_tsp_semo import TRITSPEvaluation as ProblemEvaluation

# If you want to run the bi_cvrp example, uncomment the following line:
# from llm4ad.task.optimization.bi_cvrp import BICVRPEvaluation as ProblemEvaluation

# If you want to run the bi_kp example, uncomment the following line:
# from llm4ad.task.optimization.bi_kp import BIKPEvaluation as ProblemEvaluation

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


llm_api_key = read_api_key("LLM_API_KEY", "secret.txt")
llm_api_key_cluster = read_api_key("LLM_CLUSTER_API_KEY", "secret_cluster.txt", fallback=llm_api_key)


def main():

    # LiteLLM provider. For an OpenAI-compatible GFI endpoint, keep the
    # `openai/` prefix and route through api_base.
    llm = HttpsApiLiteLLM(base_url='https://api.vectorengine.ai/v1',
                          api_key=llm_api_key,
                          model='openai/gpt-4o-mini',
                          timeout=30)
    llm_cluster = HttpsApiLiteLLM4Cluster(base_url='https://api.vectorengine.ai/v1',
                                          api_key=llm_api_key_cluster,
                                          model='openai/gpt-4o-mini',
                                          timeout=30)
    task = ProblemEvaluation()

    # LLMPFG baseline:
    # method = MPaGE(llm=llm,
    #                 llm_cluster=llm_cluster,
    #                 profiler=EoHProfiler(log_dir='logs', log_style='complex'),
    #                 evaluation=task,
    #                 max_sample_nums=200,
    #                 max_generations=20,
    #                 pop_size=6,
    #                 num_samplers=1,
    #                 num_evaluators=1,
    #                 # llm_review=True
    #              )

    method = LLMPSL(llm=llm,
                    llm_cluster=llm_cluster,
                    profiler=EoHProfiler(log_dir='logs', log_style='complex'),
                    evaluation=task,
                    max_sample_nums=200,
                    max_generations=20,
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

    method.run()


if __name__ == '__main__':
    main()


