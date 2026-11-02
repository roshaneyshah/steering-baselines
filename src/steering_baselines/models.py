"""Model specs: bedrock:<id>, anthropic:<id>, openai:<id> (OPENAI_BASE_URL for other servers), ollama:<id>."""

from __future__ import annotations

import os
from collections.abc import Callable

from strands.models.model import Model

UPSTREAM_MODEL = "bedrock:us.anthropic.claude-haiku-4-5-20251001-v1:0"


def make_model(spec: str, temperature: float | None = None, max_tokens: int = 1024) -> Model:
    provider, sep, model_id = spec.partition(":")
    if not sep or not model_id:
        raise ValueError(f"model spec must look like provider:model_id, got {spec!r}")
    provider = provider.lower()

    if provider == "bedrock":
        from strands.models.bedrock import BedrockModel

        kwargs = {"model_id": model_id, "max_tokens": max_tokens}
        if temperature is not None:
            kwargs["temperature"] = temperature
        return BedrockModel(**kwargs)

    if provider == "anthropic":
        from strands.models.anthropic import AnthropicModel

        params = {"temperature": temperature} if temperature is not None else {}
        return AnthropicModel(
            client_args={"api_key": os.environ.get("ANTHROPIC_API_KEY")},
            model_id=model_id,
            max_tokens=max_tokens,
            params=params,
        )

    if provider == "openai":
        from strands.models.openai import OpenAIModel

        client_args = {"api_key": os.environ.get("OPENAI_API_KEY", "not-needed")}
        if os.environ.get("OPENAI_BASE_URL"):
            client_args["base_url"] = os.environ["OPENAI_BASE_URL"]
        params: dict = {"max_tokens": max_tokens}
        if temperature is not None:
            params["temperature"] = temperature
        return OpenAIModel(client_args=client_args, model_id=model_id, params=params)

    if provider == "ollama":
        from strands.models.ollama import OllamaModel

        kwargs = {"model_id": model_id, "max_tokens": max_tokens}
        if temperature is not None:
            kwargs["temperature"] = temperature
        return OllamaModel(os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"), **kwargs)

    raise ValueError(f"unknown provider {provider!r}; use bedrock, anthropic, openai or ollama")


def model_factory(spec: str, temperature: float | None = None) -> Callable[[], Model]:
    return lambda: make_model(spec, temperature)
