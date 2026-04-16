"""
Trascrizione audio (Deus Ex dataset).

Strategia: chiamiamo Whisper via OpenAI SDK direttamente (OpenRouter non sempre
espone endpoint audio). In alternativa si può usare un modello locale
tipo faster-whisper se la rete OpenAI non è disponibile.

Per contenere i costi, trascriviamo una sola volta per file e cache-iamo
il risultato su disco accanto all'audio.
"""
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import List, Dict, Any

from src.utils.logging import log

_CACHE_NAME = ".transcripts.json"


def _load_cache(audio_dir: Path) -> Dict[str, str]:
    p = audio_dir / _CACHE_NAME
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(audio_dir: Path, cache: Dict[str, str]) -> None:
    p = audio_dir / _CACHE_NAME
    try:
        p.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        log.warning(f"Could not persist audio cache: {e}")


def transcribe_audio_files(audio_files: List[Path]) -> List[Dict[str, Any]]:
    """
    Ritorna una lista di dict: {file, user_hint, transcript}.
    `user_hint` è estratto dal filename (es: 20870117_010505-guido_doehn.mp3
    → "guido_doehn").
    """
    if not audio_files:
        return []

    try:
        from openai import OpenAI
    except ImportError:
        log.error("openai package not installed; cannot transcribe audio.")
        return []

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        log.warning("No API key for transcription; skipping.")
        return []

    # Se abbiamo OPENAI_API_KEY usiamo OpenAI diretto; altrimenti tentiamo OpenRouter
    if os.environ.get("OPENAI_API_KEY"):
        client = OpenAI(api_key=api_key)
    else:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    audio_dir = audio_files[0].parent
    cache = _load_cache(audio_dir)
    out: List[Dict[str, Any]] = []

    for f in audio_files:
        key = f.name
        user_hint = f.stem.split("-", 1)[-1] if "-" in f.stem else ""
        if key in cache:
            out.append({"file": key, "user_hint": user_hint, "transcript": cache[key]})
            continue

        try:
            with f.open("rb") as fh:
                resp = client.audio.transcriptions.create(
                    model="whisper-1", file=fh,
                )
            text = getattr(resp, "text", "") or ""
        except Exception as e:
            log.warning(f"Transcription failed for {f.name}: {e}")
            text = ""

        cache[key] = text
        out.append({"file": key, "user_hint": user_hint, "transcript": text})

    _save_cache(audio_dir, cache)
    log.info(f"Transcribed {len(audio_files)} audio files "
             f"(cache hits saved redundant API calls).")
    return out
