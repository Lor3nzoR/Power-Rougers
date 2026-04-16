"""
Wrapper unificato per chiamate LLM via OpenRouter + tracing Langfuse.

Tutte le chiamate passano da qui per:
- centralizzare session_id propagation (requisito challenge)
- tracing (span naming)
- JSON mode default
- retry minimale
"""
from __future__ import annotations
import os
import time
from typing import Optional

from src.config import MODELS, OPENROUTER_BASE_URL, LANGFUSE_HOST_DEFAULT
from src.utils.logging import log


# Setup Langfuse host PRIMA di importare langfuse.openai
os.environ.setdefault("LANGFUSE_HOST", LANGFUSE_HOST_DEFAULT)

# Import lazy: se mancano i pacchetti offriamo comunque una fallback explain
_client = None
_langfuse = None


def _init():
    """Lazy init del client OpenAI e Langfuse (una sola volta)."""
    global _client, _langfuse
    if _client is not None:
        return
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY non settata in env")

    try:
        from langfuse.openai import OpenAI
        from langfuse import get_client
        _client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
        _langfuse = get_client()
    except ImportError:
        log.warning("langfuse non installato, uso OpenAI vanilla (no tracing)")
        from openai import OpenAI
        _client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
        _langfuse = None


def flush():
    """Forza il flush delle tracce Langfuse (chiamare a fine run)."""
    global _langfuse
    if _langfuse:
        try:
            _langfuse.flush()
        except Exception as e:
            log.warning(f"Langfuse flush error: {e}")


def llm_call(model_key: str,
             system: str,
             user: str,
             session_id: str,
             name: Optional[str] = None,
             json_mode: bool = True,
             max_tokens: Optional[int] = None,
             temperature: float = 0.1,
             max_retries: int = 2) -> str:
    """
    Chiama un modello LLM via OpenRouter, propaga session_id a Langfuse.

    `model_key` è una chiave di config.MODELS, es. "big", "cheap", "ctx", "arb".
    Ritorna il testo del response (già .content della chat completion).
    """
    _init()
    model_id = MODELS.get(model_key, model_key)
    span_name = name or model_id.split("/")[-1]

    kwargs = dict(
        model=model_id,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        temperature=temperature,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            if _langfuse is not None:
                from langfuse import propagate_attributes
                _langfuse.update_current_span(name=span_name) if hasattr(_langfuse, "update_current_span") else None
                with propagate_attributes(session_id=session_id):
                    resp = _client.chat.completions.create(**kwargs)
            else:
                resp = _client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            log.warning(f"LLM call failed (attempt {attempt+1}/{max_retries+1}): {e}")
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"LLM call failed after {max_retries+1} attempts: {last_err}")
