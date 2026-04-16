"""
Costruzione del blocco "user_profiles" da passare agli LLM.
Restituiamo dicts strutturati, non stringhe, così l'LLM può ragionare sui campi.
"""
from __future__ import annotations
from typing import List, Dict, Any


def build_user_profiles(users: List[Dict[str, Any]],
                        social_vuln: Dict[str, Dict[str, Any]] | None = None,
                        desc_chars: int = 500) -> List[Dict[str, Any]]:
    social_vuln = social_vuln or {}
    out: List[Dict[str, Any]] = []
    for u in users:
        uid = u.get("biotag") or u.get("user_id") or u.get("id")
        svi = social_vuln.get(str(uid), {}) if uid else {}
        out.append({
            "user_id":    uid,
            "name":       f'{u.get("first_name", "")} {u.get("last_name", "")}'.strip(),
            "iban":       u.get("iban"),
            "birth_year": u.get("birth_year"),
            "job":        u.get("job"),
            "salary_eur": u.get("salary"),
            "residence":  u.get("residence", {}),
            "description": (u.get("description") or "")[:desc_chars],
            "social_vulnerability_score": svi.get("score", 0.0),
            "social_signals": svi.get("pattern_counts", {}),
        })
    return out
