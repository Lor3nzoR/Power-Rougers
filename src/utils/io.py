"""
I/O helpers: salvataggio output finale, serializzazione dossier per LLM.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, Any, List, Dict
import pandas as pd


def write_flagged_ids(path: Path, ids: Iterable[str]) -> int:
    """Scrive un ID per riga (formato output challenge)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(i) for i in ids if i is not None and str(i) != "nan"]
    # Dedup mantenendo l'ordine
    seen = set()
    uniq = []
    for x in lines:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    path.write_text("\n".join(uniq), encoding="utf-8")
    return len(uniq)


def df_to_records_clean(df: pd.DataFrame, cols: List[str]) -> List[Dict[str, Any]]:
    """
    Converte DataFrame in lista di dict "puliti" per il payload LLM:
    - arrotonda float a 3 decimali
    - converte timestamp a stringa ISO
    - drop NaN → None
    """
    cols = [c for c in cols if c in df.columns]
    sub = df[cols].copy()
    for c in sub.select_dtypes(include="float").columns:
        sub[c] = sub[c].round(3)
    for c in sub.select_dtypes(include="datetime").columns:
        sub[c] = sub[c].astype(str)
    records = sub.to_dict(orient="records")
    # Replace NaN con None per JSON compliance
    cleaned = []
    for r in records:
        cleaned.append({k: (None if isinstance(v, float) and pd.isna(v) else v)
                        for k, v in r.items()})
    return cleaned


def strip_json_fences(raw: str) -> str:
    """Rimuove ```json ... ``` eventuali dai responses LLM."""
    s = raw.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        # rimuove la prima e l'ultima riga se sono fences
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def safe_json_loads(raw: str, default: Any = None) -> Any:
    """Parse JSON tollerante ai fences e agli errori."""
    try:
        return json.loads(strip_json_fences(raw))
    except json.JSONDecodeError:
        return default
