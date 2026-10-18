"""OpenAI-compatible chat client, with usage and cost tracking.

OpenAI, Groq, Together AI and OpenRouter all serve the same chat-completions
API, so one client covers them; only the base URL, key and a few request
fields differ.
"""

import json
import os
from dataclasses import asdict, dataclass

import openai

from . import config

PROVIDERS = {
    "groq": {
        "label": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "env": "GROQ_API_KEY",
        "models": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"],
        "key_url": "https://console.groq.com/keys",
        # The free tier rejects requests over 8K tokens per minute, so the agent
        # stops searching once its conversation nears this size.
        "max_prompt_tokens": 6500,
    },
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "env": "OPENAI_API_KEY",
        "models": ["gpt-5-mini", "gpt-5-nano", "gpt-4.1-mini"],
        "key_url": "https://platform.openai.com/api-keys",
    },
    "together": {
        "label": "Together AI",
        "base_url": "https://api.together.xyz/v1",
        "env": "TOGETHER_API_KEY",
        "models": ["openai/gpt-oss-120b", "Qwen/Qwen3.5-9B"],
        "key_url": "https://api.together.ai/settings/api-keys",
    },
    "openrouter": {
        "label": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "env": "OPENROUTER_API_KEY",
        "models": ["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
        "key_url": "https://openrouter.ai/settings/keys",
    },
}

# USD per million tokens (input, output), list prices checked October 2026.
# Used only for reporting; the provider's bill is the source of truth.
PRICES = {
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
    "qwen/qwen3.8-27b": (0.80, 4.00),
    "Qwen/Qwen3.5-9B": (0.17, 0.25),
}


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, response) -> None:
        self.calls += 1
        if response.usage:
            self.input_tokens += response.usage.prompt_tokens or 0
            self.output_tokens += response.usage.completion_tokens or 0

    def cost(self, model: str) -> float:
        inp, out = PRICES.get(model, (0.0, 0.0))
        return (self.input_tokens * inp + self.output_tokens * out) / 1e6

    def to_dict(self, model: str) -> dict:
        return {**asdict(self), "cost_usd": round(self.cost(model), 5)}


@dataclass
class Backend:
    """A provider + model + key. The key is held only in this object."""

    provider: str = config.PROVIDER
    model: str = config.RATER_MODEL
    api_key: str | None = None
    effort: str = config.EFFORT
    max_prompt_tokens: int | None = None  # None: provider default, 0: no limit

    def __post_init__(self):
        spec = PROVIDERS[self.provider]
        if self.max_prompt_tokens is None:
            env = config.MAX_PROMPT_TOKENS
            self.max_prompt_tokens = env if env is not None else spec.get("max_prompt_tokens", 0)
        key = self.api_key or os.environ.get(spec["env"])
        if not key:
            raise RuntimeError(f"No API key: set {spec['env']} in .env or pass one in.")
        self._client = openai.OpenAI(api_key=key, base_url=spec["base_url"], max_retries=6, timeout=120)

    def extra_body(self) -> dict:
        """Provider- and model-specific reasoning settings."""
        if self.provider == "openrouter":
            return {"reasoning": {"effort": self.effort}}
        if "gpt-oss" in self.model or self.model.startswith("gpt-5"):
            return {"reasoning_effort": self.effort}
        if self.provider == "groq" and "qwen" in self.model:
            return {"reasoning_format": "hidden"}  # required with tool calls on Groq
        return {}

    def chat(self, messages: list[dict], tools: list[dict], tool_choice) -> object:
        return self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_completion_tokens=8000,
            extra_body=self.extra_body(),
        )


def function_tool(name: str, description: str, parameters: dict) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters, "strict": True},
    }


def force(name: str) -> dict:
    return {"type": "function", "function": {"name": name}}


def assistant_message(message) -> dict:
    """Echo an assistant turn back in the next request (reasoning text is dropped)."""
    out = {"role": "assistant", "content": message.content or ""}
    if message.tool_calls:
        out["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in message.tool_calls
        ]
    return out


def parse_args(call) -> dict:
    return json.loads(call.function.arguments or "{}")
