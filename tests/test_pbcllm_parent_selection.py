from __future__ import annotations

import copy
import unittest

import numpy as np

from llm4ad.base import Function
from llm4ad.method.PBCLLM.population import (
    PARENT_ROLE_BEHAVIOR_COMPLEMENT,
    PARENT_ROLE_HV_COMPLEMENT,
    PARENT_ROLE_WEAK_ANCHOR,
    Population,
    default_preference_vectors,
)
from llm4ad.method.PBCLLM.prompt import PBCPrompt
from llm4ad.method.LLMPFG.prompt import EoHPrompt


def _function(
    name: str,
    *,
    front,
    preference_performance,
    pbt_offset: float,
    individual_hv: float,
    contribution: float,
    diversity: float,
) -> Function:
    func = Function(
        name=name,
        args="archive",
        body=f"    return archive  # {name}",
    )
    func.algorithm = f"{{{name} algorithm}}"
    func.score = [-contribution, -diversity]
    func.pbc = {
        "fronts": [front],
        "preference_performance": preference_performance,
        "pbt": [
            [
                [pbt_offset, pbt_offset, 1.0],
                [pbt_offset + 0.1, pbt_offset + 0.1, 1.0],
            ]
        ],
        "individual_hv": individual_hv,
        "population_hv_contribution": contribution,
        "behavior_diversity": diversity,
        "coverage_loss": float(np.mean(preference_performance) + np.std(preference_performance)),
    }
    return func


class PBCLLMParentSelectionTests(unittest.TestCase):
    def _population(self, *, excluded_parent_role: str | None = None) -> tuple[Population, dict[str, Function]]:
        pop = Population(
            4,
            default_preference_vectors(2),
            hv_ref_point=np.array([10.0, 10.0]),
            normalization_ideal=np.array([0.0, 0.0]),
            normalization_nadir=np.array([10.0, 10.0]),
            parent_selection_strategy="complementary_behavior",
            excluded_parent_role=excluded_parent_role,
        )
        funcs = {
            "high_individual_hv": _function(
                "high_individual_hv",
                front=[[1.0, 9.0]],
                preference_performance=[0.8, 0.8, 0.8, 0.8, 0.8],
                pbt_offset=0.1,
                individual_hv=2.0,
                contribution=0.0,
                diversity=0.1,
            ),
            "weak_region_anchor": _function(
                "weak_region_anchor",
                front=[[1.0, 9.0]],
                preference_performance=[0.1, 0.8, 0.8, 0.8, 0.8],
                pbt_offset=0.0,
                individual_hv=0.5,
                contribution=0.1,
                diversity=0.1,
            ),
            "hv_complement": _function(
                "hv_complement",
                front=[[9.0, 1.0]],
                preference_performance=[0.4, 0.4, 0.4, 0.4, 0.4],
                pbt_offset=0.2,
                individual_hv=0.5,
                contribution=0.2,
                diversity=0.2,
            ),
            "behavior_complement": _function(
                "behavior_complement",
                front=[[1.0, 9.0]],
                preference_performance=[0.6, 0.6, 0.6, 0.6, 0.6],
                pbt_offset=10.0,
                individual_hv=0.6,
                contribution=0.0,
                diversity=10.0,
            ),
        }
        pop._population = list(funcs.values())
        return pop, funcs

    def test_complementary_behavior_selection_uses_three_explicit_roles(self):
        pop, funcs = self._population()

        parents = pop.select_parents(target_preference=0, selection_num=3)

        self.assertIs(parents[0], funcs["weak_region_anchor"])
        self.assertIs(parents[1], funcs["hv_complement"])
        self.assertIs(parents[2], funcs["behavior_complement"])
        self.assertEqual(parents[0].pbc["parent_selection_role"], "weak_preference_anchor")
        self.assertEqual(parents[1].pbc["parent_selection_role"], "hv_complement")
        self.assertEqual(parents[2].pbc["parent_selection_role"], "behavior_complement")
        self.assertGreater(parents[1].pbc["parent_selection_hv_delta"], 0.0)
        self.assertGreater(parents[2].pbc["parent_selection_min_behavior_distance"], 0.0)

    def test_no_weak_anchor_returns_only_hv_and_behavior_complements(self):
        pop, _ = self._population(excluded_parent_role=PARENT_ROLE_WEAK_ANCHOR)

        parents = pop.select_parents(target_preference=0, selection_num=3)

        self.assertEqual(
            [parent.pbc["parent_selection_role"] for parent in parents],
            [PARENT_ROLE_HV_COMPLEMENT, PARENT_ROLE_BEHAVIOR_COMPLEMENT],
        )

    def test_no_hv_complement_returns_only_weak_anchor_and_behavior_complement(self):
        pop, _ = self._population(excluded_parent_role=PARENT_ROLE_HV_COMPLEMENT)

        parents = pop.select_parents(target_preference=0, selection_num=3)

        self.assertEqual(
            [parent.pbc["parent_selection_role"] for parent in parents],
            [PARENT_ROLE_WEAK_ANCHOR, PARENT_ROLE_BEHAVIOR_COMPLEMENT],
        )

    def test_no_behavior_complement_returns_only_weak_anchor_and_hv_complement(self):
        pop, _ = self._population(excluded_parent_role=PARENT_ROLE_BEHAVIOR_COMPLEMENT)

        parents = pop.select_parents(target_preference=0, selection_num=3)

        self.assertEqual(
            [parent.pbc["parent_selection_role"] for parent in parents],
            [PARENT_ROLE_WEAK_ANCHOR, PARENT_ROLE_HV_COMPLEMENT],
        )

    def test_parent_ablation_does_not_remove_mutation_parent(self):
        pop, funcs = self._population(excluded_parent_role=PARENT_ROLE_WEAK_ANCHOR)

        parents = pop.select_parents(target_preference=0, selection_num=1)

        self.assertEqual(parents, [funcs["weak_region_anchor"]])

    def test_pbcllm_prompt_matches_llmpfg_prompt(self):
        pop, _ = self._population()
        parents = pop.select_parents(target_preference=0, selection_num=3)
        template = Function(name="select_neighbor", args="archive", body="")

        pbc_prompt = PBCPrompt.get_prompt_e1(
            "Task",
            copy.deepcopy(parents),
            copy.deepcopy(template),
        )
        llmpfg_prompt = EoHPrompt.get_prompt_e1(
            "Task",
            copy.deepcopy(parents),
            copy.deepcopy(template),
        )

        self.assertEqual(pbc_prompt, llmpfg_prompt)
        self.assertNotIn("PBCLLM parent-selection context", pbc_prompt)
        self.assertNotIn("parent-selection", pbc_prompt)


if __name__ == "__main__":
    unittest.main()
