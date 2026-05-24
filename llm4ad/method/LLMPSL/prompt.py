from __future__ import annotations

import copy
import re
from typing import Dict, List, Sequence

from ...base import Function


class LLMPSLPrompt:
    @classmethod
    def create_instruct_prompt(cls, prompt: str) -> List[Dict]:
        return [
            {
                "role": "system",
                "message": "You are an expert in optimization heuristics and multi-objective Pareto set learning.",
            },
            {"role": "user", "message": prompt},
        ]

    @classmethod
    def get_system_prompt(cls) -> str:
        return ""

    @classmethod
    def _preference_block(cls, preference: Sequence[float] | None) -> str:
        if preference is None:
            return ""

        weights = [float(x) for x in preference]
        labels = [
            "quality objective -HV, where lower means a larger final solution-archive hypervolume",
            "runtime objective, where lower means faster execution",
            "code novelty objective -CodeBLEU-novelty, where lower means more structurally novel code",
        ]
        lines = ["Target Pareto preference for this generation:"]
        for i, weight in enumerate(weights):
            label = labels[i] if i < len(labels) else f"objective {i}"
            lines.append(f"- lambda_{i} = {weight:.3f}: {label}.")

        lines.append(
            "Interpretation: lambda_0 controls the emphasis on final solution-archive hypervolume, "
            "lambda_1 controls the emphasis on runtime efficiency, and lambda_2 controls the emphasis on code-level novelty."
        )
        lines.append("A larger lambda means the generated heuristic should pay more attention to that objective.")
        lines.append(
            "Do not merely rename variables or reformat code for novelty; make a real algorithmic change in selection, move generation, repair, scoring, or randomization logic."
        )
        return "\n".join(lines)

    @classmethod
    def _empty_template(cls, template_function: Function) -> Function:
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ""
        return temp_func

    @classmethod
    def _format_individuals(cls, indivs: List[Function]) -> str:
        indivs_prompt = ""
        for i, indi in enumerate(indivs):
            indi = copy.deepcopy(indi)
            indi.docstring = ""
            score = getattr(indi, "score", None)
            novelty = getattr(indi, "code_novelty", None)
            novelty_line = f"\nCode novelty: {novelty}" if novelty is not None else ""
            indivs_prompt += (
                f"No. {i + 1} algorithm, score, and code are:\n"
                f"Algorithm: {getattr(indi, 'algorithm', '')}\n"
                f"Score: {score}{novelty_line}\n"
                f"{str(indi)}\n"
            )
        return indivs_prompt

    @classmethod
    def _clean_suggestions(cls, suggestions) -> str:
        if suggestions is None:
            return ""
        text = str(suggestions).strip()
        text = re.sub(
            r"Target Pareto preference for this generation:.*?(?=\n\s*(?:Suggestions:|Please |Create |1\.|$))",
            "",
            text,
            flags=re.DOTALL,
        )
        text = re.sub(
            r"(?:Design intent:|Interpretation:).*?(?=\n\s*(?:Suggestions:|Please |Create |1\.|$))",
            "",
            text,
            flags=re.DOTALL,
        )
        text = text.replace("---", "").strip()
        if text.lower().startswith("suggestions:"):
            text = text.split(":", 1)[1].strip()
        return text

    @classmethod
    def get_prompt_i1(cls, task_prompt: str, template_function: Function, preference: Sequence[float] | None = None):
        temp_func = cls._empty_template(template_function)
        pref_block = cls._preference_block(preference)
        return f"""{task_prompt}

{pref_block}

Create one new heuristic program for the target preference above.
1. First, describe your new algorithm and main steps in one long, detailed sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Check syntax and code carefully before returning the final function. Do not give additional explanations."""

    @classmethod
    def get_prompt_suggestions_only(
        cls,
        task_prompt: str,
        indivs: List[Function],
        template_function: Function,
        preference: Sequence[float] | None = None,
    ):
        indivs_prompt = cls._format_individuals(indivs)
        pref_block = cls._preference_block(preference)
        return f"""
I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}

{pref_block}

Please analyze the algorithms under the target preference and output a Suggestions section only:
- Identify which mechanisms are useful for the target trade-off.
- Identify limitations or runtime risks.
- Propose one concrete hybrid or improved strategy that is feasible in running time and structurally different from the given code.

Output format:
---
Suggestions:
(Write only one strategy here.)
Do not include explanations outside this section."""

    @classmethod
    def get_prompt_e1(
        cls,
        task_prompt: str,
        indivs: List[Function],
        template_function: Function,
        suggestions=None,
        preference: Sequence[float] | None = None,
    ):
        temp_func = cls._empty_template(template_function)
        pref_block = cls._preference_block(preference)
        if suggestions is None:
            indivs_prompt = cls._format_individuals(indivs)
            return f"""{task_prompt}

{pref_block}

I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}

Analyze the mechanisms of the given algorithms. Create a new heuristic for the target preference by combining useful mechanisms while making a real algorithmic change from all parents.
1. First, describe your new algorithm and main steps in one long, detailed sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Check syntax and code carefully before returning the final function. Do not give additional explanations."""

        suggestions = cls._clean_suggestions(suggestions)
        return f"""{task_prompt}

{pref_block}

Here are suggestions derived from existing algorithms:
---
Suggestions:
{suggestions}
---

Create a new heuristic based on these suggestions and the target Pareto preference.
1. First, describe your new algorithm and main steps in one long, detailed sentence. The description must be inside within boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Check syntax and code carefully before returning the final function. Do not give additional explanations."""

    @classmethod
    def get_prompt_cluster(cls, task_prompt: str, indivs: List[Function], template_function: Function, suggestions=None):
        indivs_prompt = ""
        for i, indi in enumerate(indivs):
            indi = copy.deepcopy(indi)
            indi.docstring = ""
            indivs_prompt += f"Code {i}:\n{getattr(indi, 'algorithm', '')}\n{str(indi)}\n\n"
        return f"""
I have {len(indivs)} existing algorithms with their codes as follows:

{indivs_prompt}
Group them into clusters, where:
- Each cluster contains code snippets with similar algorithmic logic.
- Different clusters should have maximally different logic.
- Return a JSON with a key "Group", whose value is a list of sublists.
- Each sublist contains the code indices starting from 0.

Return format example:
  "Group": [
    [0, 2],
    [1, 4],
    [3]
  ]
"""

    @classmethod
    def get_prompt_e2(
        cls,
        task_prompt: str,
        indivs: List[Function],
        template_function: Function,
        suggestions=None,
        preference: Sequence[float] | None = None,
    ):
        temp_func = cls._empty_template(template_function)
        pref_block = cls._preference_block(preference)
        indivs_prompt = cls._format_individuals(indivs)
        if suggestions is not None:
            suggestions = cls._clean_suggestions(suggestions)
            return f"""{task_prompt}

{pref_block}

Here are suggestions you can refer to:
---
Suggestions:
{suggestions}
---

Create a new heuristic for the target preference.
1. Firstly, identify the useful backbone idea in the suggestions.
2. Secondly, describe your new algorithm in one long, detailed sentence inside boxed {{}}.
3. Thirdly, implement the following Python function:
{str(temp_func)}
Check syntax and code carefully before returning the final function. Do not give additional explanations."""

        return f"""{task_prompt}

{pref_block}

I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}

Create a new algorithm that is motivated by the parents but has a different form and targets the preference above.
1. Firstly, identify the common backbone idea in the provided algorithms.
2. Secondly, describe your new algorithm in one long, detailed sentence inside boxed {{}}.
3. Thirdly, implement the following Python function:
{str(temp_func)}
Check syntax and code carefully before returning the final function. Do not give additional explanations."""

    @classmethod
    def get_prompt_m1(
        cls,
        task_prompt: str,
        indi: Function | List[Function],
        template_function: Function,
        preference: Sequence[float] | None = None,
    ):
        if isinstance(indi, list):
            indi = indi[0]
        temp_func = cls._empty_template(template_function)
        pref_block = cls._preference_block(preference)
        return f"""{task_prompt}

{pref_block}

I have one algorithm with its code as follows:
Algorithm description:
{getattr(indi, 'algorithm', '')}
Score: {getattr(indi, 'score', None)}
Code:
{str(indi)}

Create a modified heuristic for the target preference. Focus on changing either the solution selection policy, neighborhood move, repair logic, or scoring formula.
1. First, describe your new algorithm and main steps in one long, detailed sentence inside boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Check syntax and code carefully before returning the final function. Do not give additional explanations."""

    @classmethod
    def get_prompt_m2(
        cls,
        task_prompt: str,
        indi: Function | List[Function],
        template_function: Function,
        preference: Sequence[float] | None = None,
    ):
        if isinstance(indi, list):
            indi = indi[0]
        temp_func = cls._empty_template(template_function)
        pref_block = cls._preference_block(preference)
        return f"""{task_prompt}

{pref_block}

I have one algorithm with its code as follows:
Algorithm description:
{getattr(indi, 'algorithm', '')}
Score: {getattr(indi, 'score', None)}
Code:
{str(indi)}

Identify the main algorithm parameters or thresholds and create a new version with different parameterization for the target preference.
1. First, describe your new algorithm and main steps in one long, detailed sentence inside boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Check syntax and code carefully before returning the final function. Do not give additional explanations."""

    @classmethod
    def get_prompt_interpolate(
        cls,
        task_prompt: str,
        anchors: List[Function],
        template_function: Function,
        preference: Sequence[float],
    ):
        temp_func = cls._empty_template(template_function)
        pref_block = cls._preference_block(preference)
        anchors_prompt = cls._format_individuals(anchors)
        return f"""{task_prompt}

{pref_block}

The following anchor heuristics represent different Pareto trade-off regions:
{anchors_prompt}

Perform Pareto interpolation in program space: synthesize a new heuristic that lies between these anchors under the target preference. Preserve the quality-improving mechanism from quality-oriented anchors, preserve speed-saving mechanisms from runtime-oriented anchors, and introduce a real algorithmic difference to improve CodeBLEU novelty.
1. First, describe your new algorithm and main steps in one long, detailed sentence inside boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Check syntax and code carefully before returning the final function. Do not give additional explanations."""

    @classmethod
    def get_prompt_extrapolate(
        cls,
        task_prompt: str,
        anchor: Function,
        template_function: Function,
        preference: Sequence[float],
    ):
        temp_func = cls._empty_template(template_function)
        pref_block = cls._preference_block(preference)
        return f"""{task_prompt}

{pref_block}

The following anchor heuristic is the current champion near this preference:
Algorithm description:
{getattr(anchor, 'algorithm', '')}
Score: {getattr(anchor, 'score', None)}
Code:
{str(anchor)}

Perform Pareto extrapolation: move this heuristic further toward the target preference by changing its algorithmic mechanism, not just its constants or variable names.
1. First, describe your new algorithm and main steps in one long, detailed sentence inside boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Check syntax and code carefully before returning the final function. Do not give additional explanations."""

    @classmethod
    def get_prompt_novelty_repair(
        cls,
        task_prompt: str,
        candidate: Function,
        template_function: Function,
        preference: Sequence[float],
    ):
        temp_func = cls._empty_template(template_function)
        pref_block = cls._preference_block(preference)
        return f"""{task_prompt}

{pref_block}

The following generated heuristic is too similar to previous programs:
Algorithm description:
{getattr(candidate, 'algorithm', '')}
Code:
{str(candidate)}

Rewrite it into a genuinely different algorithmic strategy while preserving the same function signature and target preference. Change control flow, selection logic, move operator, or repair logic; do not merely rename variables.
1. First, describe your revised algorithm and main steps in one long, detailed sentence inside boxed {{}}.
2. Next, implement the following Python function:
{str(temp_func)}
Check syntax and code carefully before returning the final function. Do not give additional explanations."""
