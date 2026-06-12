from __future__ import annotations

import unittest
from unittest import mock

from llm4ad.base import Function
from llm4ad.method.PBCLLM.population import (
    BEHAVIOR_DISTANCE_DTW,
    BEHAVIOR_DISTANCE_EUCLIDEAN,
    Population,
    SURVIVOR_SELECTION_BEHAVIOR_ONLY,
    SURVIVOR_SELECTION_HV_ONLY,
    PARENT_ROLE_BEHAVIOR_COMPLEMENT,
    PARENT_ROLE_HV_COMPLEMENT,
    PARENT_ROLE_WEAK_ANCHOR,
    behavior_distance,
    default_preference_vectors,
)


def _function(name: str, *, contribution: float, diversity: float, pbt) -> Function:
    func = Function(name=name, args="x", body="    return x")
    func.score = [-contribution, -diversity]
    func.pbc = {
        "selection_hv_gain": contribution,
        "population_hv_contribution": contribution,
        "behavior_diversity": diversity,
        "behavior_novelty": diversity,
        "pbt": [pbt],
    }
    return func


class PBCLLMAblationTests(unittest.TestCase):
    def test_cli_ablation_configs_cover_all_requested_variants(self):
        from main import PBCLLM_ABLATION_CONFIGS

        self.assertEqual(
            set(PBCLLM_ABLATION_CONFIGS),
            {
                "none",
                "only_hv_contribution",
                "only_behavior_diversity",
                "euclidean_instead_of_dtw",
                "no_weak_anchor",
                "no_hv_complement",
                "no_behavior_complement",
            },
        )
        self.assertEqual(
            PBCLLM_ABLATION_CONFIGS["no_weak_anchor"]["excluded_parent_role"],
            PARENT_ROLE_WEAK_ANCHOR,
        )
        self.assertEqual(
            PBCLLM_ABLATION_CONFIGS["no_hv_complement"]["excluded_parent_role"],
            PARENT_ROLE_HV_COMPLEMENT,
        )
        self.assertEqual(
            PBCLLM_ABLATION_CONFIGS["no_behavior_complement"]["excluded_parent_role"],
            PARENT_ROLE_BEHAVIOR_COMPLEMENT,
        )

    def test_only_hv_survivor_selection_ignores_behavior_diversity(self):
        pop = Population(
            2,
            default_preference_vectors(2),
            survivor_selection_strategy=SURVIVOR_SELECTION_HV_ONLY,
        )
        pool = [
            _function("highest_behavior", contribution=0.1, diversity=100.0, pbt=[[0.0]]),
            _function("highest_hv", contribution=0.9, diversity=0.0, pbt=[[1.0]]),
            _function("second_hv", contribution=0.8, diversity=0.1, pbt=[[2.0]]),
        ]

        with mock.patch.object(pop, "_annotate_pool_metrics"):
            selected = pop._select_by_contribution_diversity(pool)

        self.assertEqual([func.name for func in selected], ["highest_hv", "second_hv"])

    def test_only_behavior_survivor_selection_ignores_hv_contribution(self):
        pop = Population(
            2,
            default_preference_vectors(2),
            survivor_selection_strategy=SURVIVOR_SELECTION_BEHAVIOR_ONLY,
        )
        pool = [
            _function("highest_hv", contribution=100.0, diversity=0.1, pbt=[[0.0]]),
            _function("highest_behavior", contribution=0.0, diversity=0.9, pbt=[[1.0]]),
            _function("second_behavior", contribution=0.1, diversity=0.8, pbt=[[2.0]]),
        ]

        with mock.patch.object(pop, "_annotate_pool_metrics"):
            selected = pop._select_by_contribution_diversity(pool)

        self.assertEqual([func.name for func in selected], ["highest_behavior", "second_behavior"])

    def test_euclidean_ablation_uses_direct_checkpoint_distance(self):
        first = _function(
            "first",
            contribution=0.0,
            diversity=0.0,
            pbt=[[0.0], [1.0], [2.0]],
        )
        second = _function(
            "second",
            contribution=0.0,
            diversity=0.0,
            pbt=[[0.0], [0.0], [1.0]],
        )

        euclidean = behavior_distance(first, second, metric=BEHAVIOR_DISTANCE_EUCLIDEAN)
        dtw = behavior_distance(first, second, metric=BEHAVIOR_DISTANCE_DTW)

        self.assertAlmostEqual(euclidean, 2.0 / 3.0)
        self.assertNotAlmostEqual(euclidean, dtw)

    def test_parent_role_ablation_rejects_legacy_parent_policy(self):
        with self.assertRaisesRegex(ValueError, "complementary_behavior"):
            Population(
                2,
                default_preference_vectors(2),
                parent_selection_strategy="legacy_role",
                excluded_parent_role=PARENT_ROLE_WEAK_ANCHOR,
            )


if __name__ == "__main__":
    unittest.main()
