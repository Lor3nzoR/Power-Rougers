"""
NLP locale: RegEx phishing + Social Vulnerability Score.

Riferimento: PDF sez. 6 - "Rilevamento Deterministico dell'Ingegneria Sociale".
L'obiettivo è calcolare per ogni utente un punteggio 0-1 basato su quanto
le sue comunicazioni (SMS + mail + trascrizioni audio) contengono pattern
di ingegneria sociale (urgenza, impersonation, credential request, ecc.).

Questo score viene poi unito alle tx come feature `social_vulnerability_score`
e usato per innalzare il rischio di transazioni sennò "innocue" ma
effettuate da utenti che sono stati potenzialmente adescati.
"""
from __future__ import annotations
import re
from collections import defaultdict
from typing import Dict, List, Any, Iterable
import numpy as np
import pandas as pd

from src.config import SOCIAL_ENG_PATTERNS
from src.utils.logging import log


# -------------------------------------------------------------------
# Compila RegEx una volta sola all'import
# -------------------------------------------------------------------
_COMPILED: Dict[str, List[re.Pattern]] = {
    cat: [re.compile(p, re.IGNORECASE) for p in patterns]
    for cat, patterns in SOCIAL_ENG_PATTERNS.items()
}


def count_patterns(text: str) -> Dict[str, int]:
    """Ritorna per ogni categoria quante occorrenze di pattern."""
    out = {cat: 0 for cat in _COMPILED}
    if not isinstance(text, str) or not text:
        return out
    for cat, regs in _COMPILED.items():
        c = 0
        for r in regs:
            c += len(r.findall(text))
        out[cat] = c
    return out


def _aggregate_user_text(user_id: str,
                         sms: List[Dict[str, Any]],
                         mails: List[Dict[str, Any]],
                         audio_transcripts: List[Dict[str, Any]]) -> str:
    """Concatena tutto il testo associato a un utente."""
    chunks: List[str] = []
    for s in sms:
        if str(s.get("user_id", s.get("sender_id", s.get("biotag", "")))) == str(user_id):
            chunks.append(str(s.get("sms", "")))
    for m in mails:
        # I mail non sempre hanno user_id; sosteniamo sia 'user_id' che niente
        uid = m.get("user_id") or m.get("sender") or ""
        if str(uid) == str(user_id):
            chunks.append(str(m.get("mail", "")))
    for a in audio_transcripts:
        # L'euristica user_hint estratta dal filename
        hint = str(a.get("user_hint", "")).lower().replace("_", " ")
        if hint and hint in str(user_id).lower():
            chunks.append(str(a.get("transcript", "")))
    return "\n".join(chunks)


def compute_social_vulnerability(users: List[Dict[str, Any]],
                                 sms: List[Dict[str, Any]],
                                 mails: List[Dict[str, Any]],
                                 audio_transcripts: List[Dict[str, Any]] | None = None,
                                 ) -> Dict[str, Dict[str, Any]]:
    """
    Ritorna un dict user_id -> {score, pattern_counts, has_comms}.

    Strategia:
    1) Raccogliamo per ogni utente tutto il testo (sms + mail + audio)
    2) Contiamo i match RegEx per categoria
    3) Convertiamo i count in un score 0-1 via log-normalization
       (più segnali ⇒ score più alto, saturando)
    """
    audio_transcripts = audio_transcripts or []
    out: Dict[str, Dict[str, Any]] = {}

    if not (sms or mails or audio_transcripts):
        return out

    for u in users:
        uid = u.get("biotag") or u.get("user_id") or u.get("id")
        if not uid:
            continue
        text = _aggregate_user_text(uid, sms, mails, audio_transcripts)
        counts = count_patterns(text)
        total = sum(counts.values())
        has_comms = bool(text.strip())
        # Saturation: 0 hit → 0, 1 hit → ~0.4, 3+ hits → ~0.8, 6+ → ~0.95
        score = 0.0 if total == 0 else float(1 - np.exp(-0.5 * total))
        out[str(uid)] = {
            "score": round(score, 3),
            "pattern_counts": counts,
            "has_comms": has_comms,
            "total_hits": total,
        }

    log.info(f"Social vulnerability computed for {len(out)} users; "
             f"at-risk (score>0.5): {sum(1 for v in out.values() if v['score'] > 0.5)}")
    return out


def attach_social_scores(tx: pd.DataFrame,
                         svi: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Aggiunge la colonna social_vulnerability_score alle tx."""
    tx = tx.copy()
    if not svi:
        tx["social_vulnerability_score"] = 0.0
        return tx
    mapping = {k: v["score"] for k, v in svi.items()}
    tx["social_vulnerability_score"] = (
        tx["sender_id"].astype(str).map(mapping).fillna(0.0)
    )
    return tx
