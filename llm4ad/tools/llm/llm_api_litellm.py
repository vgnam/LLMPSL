from __future__ import annotations

import json
import re
from typing import Any

from ...base import LLM


class HttpsApiLiteLLM(LLM):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout=30,
        temperature=0.7,
        max_tokens=8192,
        **kwargs,
    ):
        super().__init__()
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._kwargs = kwargs

    def _messages(self, prompt: str | Any, system_prompt: str):
        if isinstance(prompt, str):
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        if isinstance(prompt, list):
            return prompt
        raise ValueError("Unsupported prompt format")

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        try:
            from litellm import completion
        except ImportError as exc:
            raise ImportError(
                "LiteLLM is not installed. Install it with `python -m pip install litellm` "
                "or install project requirements."
            ) from exc

        messages = self._messages(
            prompt,
            "You are an expert in the domain of optimization heuristics helping to design heuristics that can effectively solve optimization problems.",
        )
        response = completion(
            model=self._model,
            messages=messages,
            api_key=self._api_key,
            api_base=self._base_url,
            timeout=self._timeout,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            **self._kwargs,
        )
        return response.choices[0].message.content


class HttpsApiLiteLLM4Cluster(HttpsApiLiteLLM):
    def _extract_group(self, content: str):
        try:
            data = json.loads(content)
        except Exception:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if not match:
                return None
            try:
                data = json.loads(match.group(0))
            except Exception:
                return None
        if isinstance(data, dict):
            return data.get("Group")
        return None

    def draw_sample(self, prompt: str | Any, *args, **kwargs):
        try:
            from litellm import completion
        except ImportError as exc:
            raise ImportError(
                "LiteLLM is not installed. Install it with `python -m pip install litellm` "
                "or install project requirements."
            ) from exc

        messages = self._messages(
            prompt,
            "You are an expert in program analysis and logic abstraction. Return valid JSON only when JSON is requested.",
        )
        response = completion(
            model=self._model,
            messages=messages,
            api_key=self._api_key,
            api_base=self._base_url,
            timeout=self._timeout,
            temperature=0.2,
            max_tokens=self._max_tokens,
            **self._kwargs,
        )
        content = response.choices[0].message.content
        group = self._extract_group(content)
        return group if group is not None else content
