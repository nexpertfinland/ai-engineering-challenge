"""Thin LLM wrapper — edit freely.

Defaults to a local Ollama model (nothing leaves your laptop). Swap in a
cloud model any time by setting LLM_PROVIDER=openai or LLM_PROVIDER=anthropic
(plus the matching API key) in your .env — no other code changes needed.

Every call through call_llm() records token usage (see _usage below) so the
harness can report a token-frugality score alongside the accuracy score. If
you replace call_llm's internals wholesale, keep calling _record_usage(...)
or the frugality score will just read as 0 for you.
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = {
    "ollama": os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
    "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
}[PROVIDER]

_usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}


def _record_usage(input_tokens: int, output_tokens: int) -> None:
    _usage["input_tokens"] += input_tokens
    _usage["output_tokens"] += output_tokens
    _usage["calls"] += 1


def reset_usage() -> None:
    """Zero the running token counter. Called by the harness before scoring."""
    _usage["input_tokens"] = 0
    _usage["output_tokens"] = 0
    _usage["calls"] = 0


def get_usage() -> dict:
    """Return cumulative token usage since the last reset_usage() call."""
    return dict(_usage)


def ensure_model_ready() -> None:
    """Fail once, clearly, before an evaluation attempts dozens of requests."""
    if PROVIDER == "ollama":
        try:
            response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                f"Ollama is unavailable at {OLLAMA_HOST}. Start Ollama, then retry."
            ) from exc
        names = {m.get("name", "") for m in response.json().get("models", [])}
        expected = DEFAULT_MODEL if ":" in DEFAULT_MODEL else f"{DEFAULT_MODEL}:latest"
        if expected not in names:
            raise RuntimeError(f"Model is not installed. Run: ollama pull {DEFAULT_MODEL}")
    elif not os.getenv({"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}[PROVIDER]):
        raise RuntimeError(f"The API key for {PROVIDER} is missing from the environment/.env.")


def call_llm(
    prompt: str, system: str | None = None, model: str | None = None,
    *, temperature: float | None = None, max_tokens: int | None = None,
    json_mode: bool = False,
) -> str:
    """Send a single-turn prompt to the configured LLM and return the text response."""
    model = model or DEFAULT_MODEL

    if PROVIDER == "ollama":
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": model, "messages": messages, "stream": False}
        options = {}
        if temperature is not None:
            options.update(temperature=temperature, seed=42)
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if options:
            payload["options"] = options
        if json_mode:
            payload["format"] = "json"
        try:
            resp = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json=payload,
                timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "180")),
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                "Could not reach Ollama. Is it running? Start it with `ollama serve` "
                f"and make sure the model is pulled: `ollama pull {model}`."
            ) from exc
        data = resp.json()
        _record_usage(data.get("prompt_eval_count", 0), data.get("eval_count", 0))
        return data["message"]["content"]

    if PROVIDER == "openai":
        from openai import OpenAI

        client = OpenAI()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["max_completion_tokens"] = max_tokens
        if json_mode:
            options["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(model=model, messages=messages, **options)
        _record_usage(resp.usage.prompt_tokens, resp.usage.completion_tokens)
        return resp.choices[0].message.content

    if PROVIDER == "anthropic":
        import anthropic

        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens or 512,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
            **({"temperature": temperature} if temperature is not None else {}),
        )
        _record_usage(resp.usage.input_tokens, resp.usage.output_tokens)
        return resp.content[0].text

    raise ValueError(f"Unknown LLM_PROVIDER: {PROVIDER}")
