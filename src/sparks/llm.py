"""LLM client — uses Claude Code CLI as backend. No API key needed."""

from __future__ import annotations

import json
import subprocess
import os
import re
from typing import Any, Optional, Type

from pydantic import BaseModel

from sparks.cost import CostTracker

# Model mapping: our routing names → claude CLI model flags
MODEL_MAP = {
    "claude-opus-4-20250514": "opus",
    "claude-sonnet-4-20250514": "sonnet",
    "claude-haiku-4-5-20251001": "haiku",
}

# Backend: "cli" (Claude Code) or "api" (Anthropic API)
BACKEND = os.environ.get("SPARKS_BACKEND", "cli")


def llm_call(
    prompt: str,
    model: str,
    system: str = "",
    tool: str = "unknown",
    tracker: Optional[CostTracker] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> str:
    """Text LLM call via Claude Code CLI."""
    if BACKEND == "api":
        return _api_call(prompt, model, system, tool, tracker, temperature, max_tokens)

    full_prompt = prompt
    if system:
        full_prompt = f"[System: {system}]\n\n{prompt}"

    cli_model = MODEL_MAP.get(model, "sonnet")

    result = subprocess.run(
        ["claude", "-p", "--model", cli_model, "--output-format", "text"],
        input=full_prompt,
        capture_output=True,
        text=True,
        timeout=900,
    )

    if result.returncode != 0:
        err = result.stderr[:500] or result.stdout[:500]
        raise RuntimeError(f"Claude CLI failed (rc={result.returncode}): {err}")

    text = result.stdout.strip()

    if tracker:
        # Estimate tokens (CLI doesn't report exact usage)
        est_input = len(prompt) // 4
        est_output = len(text) // 4
        tracker.record(tool=tool, model=model, input_tokens=est_input, output_tokens=est_output)

    return text


def llm_structured(
    prompt: str,
    model: str,
    schema: Type[BaseModel],
    tool_name: str = "respond",
    tool: str = "unknown",
    tracker: Optional[CostTracker] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    system: str = "",
) -> BaseModel:
    """Structured output: ask for JSON, parse into Pydantic model."""
    if BACKEND == "api":
        return _api_structured(prompt, model, schema, tool_name, tool, tracker, temperature, max_tokens, system)

    # Add JSON instruction to prompt
    schema_hint = json.dumps(schema.model_json_schema(), indent=2)
    json_prompt = f"""{prompt}

IMPORTANT: Respond with ONLY valid JSON matching this schema. No markdown, no explanation, just JSON.

Schema:
{schema_hint}"""

    text = llm_call(json_prompt, model=model, system=system, tool=tool, tracker=tracker)

    # Extract JSON from response (handle markdown code blocks)
    json_str = _extract_json(text)

    try:
        data = json.loads(json_str)
        return schema.model_validate(data)
    except Exception:
        # Retry with stricter prompt
        retry_prompt = f"""The previous response was not valid JSON. Please respond with ONLY valid JSON, no other text.

{json_prompt}"""
        try:
            text2 = llm_call(retry_prompt, model=model, system=system, tool=f"{tool}_retry", tracker=tracker)
            json_str2 = _extract_json(text2)
            data2 = json.loads(json_str2)
            return schema.model_validate(data2)
        except Exception:
            # Return empty schema as last resort
            return schema.model_validate({})


def _extract_json(text: str) -> str:
    """Extract JSON from text that might have markdown code blocks."""
    # Try direct parse first
    text = text.strip()
    if text.startswith("{") or text.startswith("["):
        return text

    # Try extracting from ```json ... ``` blocks
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try finding first { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return text[start:end + 1]

    # Try finding first [ to last ]
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        return text[start:end + 1]

    return text


# ── API Backend (fallback if API key is available) ──

def _api_call(prompt, model, system, tool, tracker, temperature, max_tokens):
    import anthropic
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": prompt}]
    kwargs: dict[str, Any] = {
        "model": model, "max_tokens": max_tokens,
        "messages": messages, "temperature": temperature,
    }
    if system:
        kwargs["system"] = system
    response = client.messages.create(**kwargs)
    text = response.content[0].text
    if tracker:
        tracker.record(tool=tool, model=model,
                       input_tokens=response.usage.input_tokens,
                       output_tokens=response.usage.output_tokens)
    return text


def _api_structured(prompt, model, schema, tool_name, tool, tracker, temperature, max_tokens, system):
    import anthropic
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": prompt}]
    tools = [{"name": tool_name, "description": f"Respond with {schema.__name__}.",
              "input_schema": schema.model_json_schema()}]
    kwargs: dict[str, Any] = {
        "model": model, "max_tokens": max_tokens, "messages": messages,
        "tools": tools, "tool_choice": {"type": "tool", "name": tool_name},
        "temperature": temperature,
    }
    if system:
        kwargs["system"] = system
    response = client.messages.create(**kwargs)
    if tracker:
        tracker.record(tool=tool, model=model,
                       input_tokens=response.usage.input_tokens,
                       output_tokens=response.usage.output_tokens)
    for block in response.content:
        if block.type == "tool_use":
            return schema.model_validate(block.input)
    for block in response.content:
        if block.type == "text":
            return schema.model_validate(json.loads(block.text))
    raise ValueError("No structured output from API")
