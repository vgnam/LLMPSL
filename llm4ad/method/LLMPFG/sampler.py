from __future__ import annotations

import re
from typing import Tuple, List, Dict

from .prompt import EoHPrompt
from ...base import LLM, SampleTrimmer, Function, Program


class EoHSampler:
    def __init__(self, sampler: LLM, template_program: str | Program):
        self._sampler = sampler
        self._template_program = template_program
        self.last_response = None
        self.last_thought = None
        self.last_extracted_code = None
        self.last_function = None

    def get_thought(self, prompt: str):
        response = self._sampler.draw_sample(prompt)
        self.last_response = response
        self.last_thought = response
        self.last_extracted_code = None
        self.last_function = None
        return response

    def get_thought_and_function(self, prompt: str) -> Tuple[str, Function]:
        response = self._sampler.draw_sample(prompt)
        self.last_response = response
        thought = self.__class__.trim_thought_from_response(response)
        code = SampleTrimmer.trim_preface_of_function(response)
        function = SampleTrimmer.sample_to_function(code, self._template_program)
        self.last_thought = thought
        self.last_extracted_code = code
        self.last_function = function
        return thought, function

    @classmethod
    def trim_thought_from_response(cls, response: str) -> str:
        if not isinstance(response, str):
            return "{Generated heuristic.}"

        match = re.search(r'\{.*?\}', response, flags=re.DOTALL)
        if match:
            return match.group(0)

        preface = response.split("def ", 1)[0]
        preface = re.sub(r"```(?:python)?", "", preface, flags=re.IGNORECASE).strip()
        if preface:
            return "{" + " ".join(preface.split()) + "}"
        return "{Generated heuristic.}"
