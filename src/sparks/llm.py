"""LLM client — multi-provider backend.

Backends (set via SPARKS_BACKEND env var):
  cli          — Claude Code CLI (default, free with subscription)
  anthropic    — Anthropic API (Claude)
  openai       — OpenAI API (GPT-4o, o1, etc.)
  google       — Google Gemini API
  openai-compat — Any OpenAI-compatible API (Groq, Together, Ollama)

Model names are provider-agnostic internally. Each backend maps them
to the correct provider-specific model ID.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Any, Optional, Type

from pydantic import BaseModel

from sparks.cost import CostTracker

# ── Backend selection ──

BACKEND = os.environ.get("SPARKS_BACKEND", "cli")

# ── Model name mapping per provider ──

# Internal names used in cost.py MODEL_ROUTING
# Each provider maps these to their own model IDs

PROVIDER_MODELS = {
    "cli": {
        "claude-opus-4-20250514": "opus",
        "claude-sonnet-4-20250514": "sonnet",
        "claude-haiku-4-5-20251001": "haiku",
    },
    "anthropic": {
        # Pass through as-is (already Anthropic model IDs)
    },
    "openai": {
        "claude-opus-4-20250514": "gpt-4o",
        "claude-sonnet-4-20250514": "gpt-4o-mini",
        "claude-haiku-4-5-20251001": "gpt-4o-mini",
        # Native OpenAI names also work
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-4.1": "gpt-4.1",
        "gpt-4.1-mini": "gpt-4.1-mini",
        "gpt-4.1-nano": "gpt-4.1-nano",
        "o1": "o1",
        "o3": "o3",
        "o3-mini": "o3-mini",
        "o4-mini": "o4-mini",
    },
    "google": {
        "claude-opus-4-20250514": "gemini-2.5-pro",
        "claude-sonnet-4-20250514": "gemini-2.5-flash",
        "claude-haiku-4-5-20251001": "gemini-2.0-flash-lite",
        # Native Gemini names
        "gemini-2.5-pro": "gemini-2.5-pro",
        "gemini-2.5-flash": "gemini-2.5-flash",
        "gemini-2.0-flash-lite": "gemini-2.0-flash-lite",
    },
    "openai-compat": {
        # User sets SPARKS_MODEL_MAP or passes native model names
        "claude-opus-4-20250514": os.environ.get("SPARKS_COMPAT_LARGE", "llama-3.3-70b"),
        "claude-sonnet-4-20250514": os.environ.get("SPARKS_COMPAT_MEDIUM", "llama-3.1-8b"),
        "claude-haiku-4-5-20251001": os.environ.get("SPARKS_COMPAT_SMALL", "llama-3.1-8b"),
    },
}


def _resolve_model(model: str) -> str:
    """Map internal model name to provider-specific model ID."""
    mapping = PROVIDER_MODELS.get(BACKEND, {})
    return mapping.get(model, model)  # Fall through if already native name


# ── Public API ──


def llm_call(
    prompt: str,
    model: str,
    system: str = "",
    tool: str = "unknown",
    tracker: Optional[CostTracker] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> str:
    """Text LLM call — dispatches to the configured backend."""
    resolved = _resolve_model(model)

    if BACKEND == "cli":
        return _cli_call(prompt, resolved, system, tool, tracker, model)
    elif BACKEND == "anthropic":
        return _anthropic_call(prompt, model, system, tool, tracker, temperature, max_tokens)
    elif BACKEND == "openai":
        return _openai_call(prompt, resolved, system, tool, tracker, temperature, max_tokens, model)
    elif BACKEND == "google":
        return _google_call(prompt, resolved, system, tool, tracker, temperature, max_tokens, model)
    elif BACKEND == "openai-compat":
        return _openai_compat_call(prompt, resolved, system, tool, tracker, temperature, max_tokens, model)
    else:
        raise ValueError(f"Unknown backend: {BACKEND}. Use: cli, anthropic, openai, google, openai-compat")


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
    """Structured output — ask for JSON, parse into Pydantic model."""
    # Anthropic API has native tool_use for structured output
    if BACKEND == "anthropic":
        return _anthropic_structured(prompt, model, schema, tool_name, tool, tracker, temperature, max_tokens, system)

    # OpenAI has response_format for structured output
    if BACKEND == "openai":
        resolved = _resolve_model(model)
        return _openai_structured(prompt, resolved, schema, tool, tracker, temperature, max_tokens, system, model)

    # All other backends: JSON-in-prompt approach
    schema_hint = json.dumps(schema.model_json_schema(), indent=2)
    json_prompt = f"""{prompt}

IMPORTANT: Respond with ONLY valid JSON matching this schema. No markdown, no explanation, just JSON.

Schema:
{schema_hint}"""

    text = llm_call(json_prompt, model=model, system=system, tool=tool, tracker=tracker,
                    temperature=temperature, max_tokens=max_tokens)
    json_str = _extract_json(text)

    try:
        data = json.loads(json_str)
        return schema.model_validate(data)
    except Exception:
        retry_prompt = f"The previous response was not valid JSON. Respond with ONLY valid JSON.\n\n{json_prompt}"
        try:
            text2 = llm_call(retry_prompt, model=model, system=system, tool=f"{tool}_retry", tracker=tracker)
            json_str2 = _extract_json(text2)
            return schema.model_validate(json.loads(json_str2))
        except Exception:
            return schema.model_validate({})


# ── Backend: Claude Code CLI ──


def _cli_call(prompt, cli_model, system, tool, tracker, original_model):
    full_prompt = f"[System: {system}]\n\n{prompt}" if system else prompt

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
        est_input = len(prompt) // 4
        est_output = len(text) // 4
        tracker.record(tool=tool, model=original_model, input_tokens=est_input, output_tokens=est_output)

    return text


# ── Backend: Anthropic API ──


def _anthropic_call(prompt, model, system, tool, tracker, temperature, max_tokens):
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


def _anthropic_structured(prompt, model, schema, tool_name, tool, tracker, temperature, max_tokens, system):
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
    raise ValueError("No structured output from Anthropic API")


# ── Backend: OpenAI API ──


def _openai_call(prompt, resolved_model, system, tool, tracker, temperature, max_tokens, original_model):
    from openai import OpenAI
    client = OpenAI()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=resolved_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = response.choices[0].message.content or ""
    if tracker and response.usage:
        tracker.record(tool=tool, model=original_model,
                       input_tokens=response.usage.prompt_tokens,
                       output_tokens=response.usage.completion_tokens)
    return text


def _openai_structured(prompt, resolved_model, schema, tool, tracker, temperature, max_tokens, system, original_model):
    from openai import OpenAI
    client = OpenAI()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # Use response_format for structured output
    response = client.chat.completions.create(
        model=resolved_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_schema", "json_schema": {
            "name": schema.__name__,
            "schema": schema.model_json_schema(),
            "strict": False,
        }},
    )
    text = response.choices[0].message.content or "{}"
    if tracker and response.usage:
        tracker.record(tool=tool, model=original_model,
                       input_tokens=response.usage.prompt_tokens,
                       output_tokens=response.usage.completion_tokens)
    return schema.model_validate(json.loads(text))


# ── Backend: Google Gemini API ──


def _google_call(prompt, resolved_model, system, tool, tracker, temperature, max_tokens, original_model):
    from google import genai

    client = genai.Client()
    config = genai.types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=system if system else None,
    )
    response = client.models.generate_content(
        model=resolved_model,
        contents=prompt,
        config=config,
    )
    text = response.text or ""
    if tracker:
        usage = response.usage_metadata
        tracker.record(tool=tool, model=original_model,
                       input_tokens=usage.prompt_token_count if usage else len(prompt) // 4,
                       output_tokens=usage.candidates_token_count if usage else len(text) // 4)
    return text


# ── Backend: OpenAI-Compatible API (Groq, Together, Ollama, etc.) ──


def _openai_compat_call(prompt, resolved_model, system, tool, tracker, temperature, max_tokens, original_model):
    from openai import OpenAI

    base_url = os.environ.get("SPARKS_COMPAT_BASE_URL", "http://localhost:11434/v1")
    api_key = os.environ.get("SPARKS_COMPAT_API_KEY", "not-needed")

    client = OpenAI(base_url=base_url, api_key=api_key)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=resolved_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = response.choices[0].message.content or ""
    if tracker and response.usage:
        tracker.record(tool=tool, model=original_model,
                       input_tokens=response.usage.prompt_tokens,
                       output_tokens=response.usage.completion_tokens)
    return text


# ── JSON extraction helper ──


def _extract_json(text: str) -> str:
    """Extract JSON from text that might have markdown code blocks."""
    text = text.strip()
    if text.startswith("{") or text.startswith("["):
        return text

    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return text[start:end + 1]

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        return text[start:end + 1]

    return text
