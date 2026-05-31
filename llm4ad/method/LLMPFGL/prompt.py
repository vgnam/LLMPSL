from __future__ import annotations

import copy
from typing import List

import numpy as np

from ...base import Function


class PFGLPrompt:
    """Prompt templates for LLM-PFGL.

    Every prompt can optionally receive a ``geometry_context`` dict produced
    by ``_geometry_context()`` in ``eoh.py``.  When present, the prompt
    prepends a *Pareto Front Geometry Report* so that the LLM is always aware
    of the current trade-off landscape, no matter which evolutionary operator
    is being used.
    """

    @staticmethod
    def _geometry_report_text(geo: dict | None) -> str:
        """Return a formatted geometry report block, or empty string if None."""
        if geo is None:
            return ""
        target_str = ", ".join(f"{v:.2f}" for v in geo["target_obj"])
        return f"""=== PARETO FRONT GEOMETRY REPORT ===

- Current focus bearing angle: {geo['theta_gap_deg']:.1f} degrees
- Target objective vector: [{target_str}]
- Front shape: {geo['shape']}
- Confidence: {geo['confidence']}

"""

    @classmethod
    def get_prompt_i1(
        cls,
        task_prompt: str,
        template_function: Function,
        geometry_context: dict | None = None,
    ):
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ""
        geom = cls._geometry_report_text(geometry_context)
        if geom:
            focus = (
                "Your heuristic should help explore the full Pareto front, "
                f"with special attention to the bearing {geometry_context['theta_gap_deg']:.1f}° region."
            )
        else:
            focus = "Your heuristic should help explore the full Pareto front."
        prompt_content = f"""{task_prompt}

{geom}{focus}
1. First, describe your new algorithm and main steps in one long, detail sentence. The description must be inside within boxed {{}}. 
2. Next, implement the following Python function:
{str(temp_func)} \n
Check syntax, code carefully before returning the final function. Do not give additional explanations."""
        return prompt_content

    @classmethod
    def get_prompt_gap_infill(
        cls,
        task_prompt: str,
        template_function: Function,
        theta_gap_deg: float,
        target_obj: tuple[float, ...],
        confidence: str,
        shape: str,
        parent_a: Function | None,
        parent_b: Function | None,
    ) -> str:
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ""

        target_str = ", ".join(f"{v:.2f}" for v in target_obj)

        parents_section = ""
        if parent_a is not None and parent_b is not None:
            parents_section = f"""Existing heuristics near this region:
- Left neighbor heuristic:
{parent_a.algorithm}
- Right neighbor heuristic:
{parent_b.algorithm}
"""
        else:
            parents_section = ""

        prompt_content = f"""{task_prompt}

=== PARETO FRONT GEOMETRY REPORT ===

Target Trade-off Region:
- Bearing angle: {theta_gap_deg:.1f} degrees
- Target objective vector: [{target_str}]
- Front shape: {shape}
- Confidence: {confidence}

{parents_section}INSTRUCTION:
Synthesize a heuristic that specifically targets this trade-off region.
The Pareto front geometry indicates this area needs densification.
1. First, describe your new algorithm and main steps in one long, detail sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Check syntax, code carefully before returning the final function. Do not give additional explanations."""
        return prompt_content

    @classmethod
    def get_prompt_suggestions_only(cls, task_prompt: str, indivs: List[Function], template_function: Function):
        for indi in indivs:
            assert hasattr(indi, 'algorithm')
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f'No. {i + 1} algorithm and the corresponding code are:\n{indi.algorithm}\n{str(indi)}'
        prompt_content = f'''
I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}
\n Please carefully analyze all of the above algorithms. Your task is to synthesize their ideas, 
identify recurring patterns, and point out opportunities for improvement.\n\n
Your output should be a **Suggestions** section, where you:\n
- Summarize key strengths shared across the implementations.\n 
- Identify limitations or blind spots that appear in multiple codes.\n
- Propose hybrid or improved strategies that integrate strengths and overcome shortcomings, in a feasible running time.\n\n
Output format:\n 
---\n
Suggestions:\n(Write only one propose hybrid or improved strategie that integrate strengths and overcome shortcomings here.)\n
Do not include any explanations, summaries, or new algorithms outside of this section. '''
        return prompt_content

    @classmethod
    def get_prompt_e1(
        cls,
        task_prompt: str,
        indivs: List[Function],
        template_function: Function,
        suggestions = None,
        geometry_context: dict | None = None,
    ):
        for indi in indivs:
            assert hasattr(indi, 'algorithm')
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f'No. {i + 1} algorithm and the corresponding code are:\n{indi.algorithm}\n{str(indi)}'
        geom = cls._geometry_report_text(geometry_context)
        if suggestions is None:
            prompt_content = f'''{task_prompt}

{geom}I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}

Analyze the logic of all the given code snippets carefully. Then identify the two code snippets whose logic is most different from each other
and create a new algorithm that totally different in logic and form from both of them.
1. First, describe your new algorithm and main steps in one long, detail sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Check syntax, code carefully before returning the final function. Do not give additional explanations.'''
        else:
            prompt_content = f'''{task_prompt}

{geom}\n Here are some suggestions you can refer to:\n
---\n
Suggestions:\n + {suggestions} + \n
---\n\n 
Please help me create a new algorithm based on the above suggestions. 
1. First, describe your new algorithm and main steps in one long, detail sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Check syntax, code carefully before returning the final function. Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_e2(
        cls,
        task_prompt: str,
        indivs: List[Function],
        template_function: Function,
        suggestions = None,
        geometry_context: dict | None = None,
    ):
        for indi in indivs:
            assert hasattr(indi, 'algorithm')
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f'No. {i + 1} algorithm and the corresponding code are:\n{indi.algorithm}\n{str(indi)}'
        geom = cls._geometry_report_text(geometry_context)
        if suggestions is not None:
            prompt_content = f'''{task_prompt}

{geom}\n Here are some suggestions you can refer to:\n
---\n
Suggestions:\n + {suggestions} + \n
---\n\n 
Please help me create a new algorithm based on the above suggestions.
1. Firstly, identify the common backbone idea in the provided algorithms. 
2. Secondly, based on the backbone idea describe your new algorithm. The description must be inside within boxed {{}}.
3. Thirdly, implement the following Python function:
{str(temp_func)}
Check syntax, code carefully before returning the final function. Do not give additional explanations.'''
        else:
            prompt_content = f'''{task_prompt}

{geom}I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}
Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them.
1. Firstly, identify the common backbone idea in the provided algorithms. 
2. Secondly, based on the backbone idea describe your new algorithm in one long, detail sentence. The description must be inside within boxed {{}}.
3. Thirdly, implement the following Python function:
{str(temp_func)}
Check syntax, code carefully before returning the final function. Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_m1(
        cls,
        task_prompt: str,
        indi: Function,
        template_function: Function,
        geometry_context: dict | None = None,
    ):
        assert hasattr(indi, 'algorithm')
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        geom = cls._geometry_report_text(geometry_context)
        if geom:
            note = (
                f"When modifying this heuristic, keep in mind that the current "
                f"Pareto front focus is at {geometry_context['theta_gap_deg']:.1f}° "
                f"({geometry_context['shape']}, {geometry_context['confidence']})."
            )
        else:
            note = ""
        prompt_content = f'''{task_prompt}

{geom}{note}
I have one algorithm with its code as follows. Algorithm description:
{indi.algorithm}
Code:
{str(indi)}
Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided. You may focus on refining either the selection phase or the neighborhood search phase.
1. First, describe your new algorithm and main steps in one long, detail sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Check syntax, code carefully before returning the final function. Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_m2(
        cls,
        task_prompt: str,
        indi: Function,
        template_function: Function,
        geometry_context: dict | None = None,
    ):
        assert hasattr(indi, 'algorithm')
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        geom = cls._geometry_report_text(geometry_context)
        if geom:
            note = (
                f"When tuning parameters, keep in mind that the current "
                f"Pareto front focus is at {geometry_context['theta_gap_deg']:.1f}° "
                f"({geometry_context['shape']}, {geometry_context['confidence']})."
            )
        else:
            note = ""
        prompt_content = f'''{task_prompt}

{geom}{note}
I have one algorithm with its code as follows. Algorithm description:
{indi.algorithm}
Code:
{str(indi)}
Please identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided. You may focus on refining either the selection phase or the neighborhood search phase
1. First, describe your new algorithm and main steps  in one long, detail sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Check syntax, code carefully before returning the final function. Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_cluster(cls, task_prompt: str, indivs: List[Function], template_function: Function):
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ""
        indivs_prompt = ""
        for i, indi in enumerate(indivs):
            indi.docstring = ""
            indivs_prompt += f"Code {i}: \n{indi.algorithm}\n{str(indi)}\n\n"
        prompt_content = f"""I have {len(indivs)} existing algorithms with their codes as follows: \n
{indivs_prompt}
Group them into clusters, where:
- Each cluster contains code snippets with similar logic.
- Different clusters should have maximally different logic.
- Return a JSON with a key "Group", whose value is a list of sublists.
- Each sublist contains the code indices (starting from 0) that belong to that cluster.

Return format example:
  "Group": [
    [0, 2],
    [1, 4],
    [3]
  ]
  """
        return prompt_content
