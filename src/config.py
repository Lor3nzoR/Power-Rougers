"""
Configurazione centrale del Mirror Agent.
Tutti i numeri magici, modelli LLM e soglie vivono qui.
"""
from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "datasets"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

for p in (OUTPUTS_DIR, SUBMISSIONS_DIR):
    p.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# LLM Models (OpenRouter IDs)
# -------------------------------------------------------------------
MODELS: Dict[str, str] = {
    # "Brain" - Reviewer principale su dossier piccoli o singoli pass
    "big":   "anthropic/claude-sonnet-4.5",
    # Worker economico per batch su chunk grandi
    "cheap": "google/gemini-2.5-flash-lite",
    # Summarizer SMS/mail (veloce, finestra ampia)
    "ctx":   "google/gemini-2.5-flash",
    # Arbiter cost-sensitive che filtra falsi positivi del worker
    "arb":   "openai/gpt-5-mini",
    # Trascrizione audio (solo Deus Ex) - modello Whisper via OpenRouter
    "asr":   "openai/whisper-1",
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LANGFUSE_HOST_DEFAULT = "https://challenges.reply.com/langfuse"

# -------------------------------------------------------------------
# Screening thresholds (Layer Muscolare)
# Riferimenti: PDF "Metriche Deterministiche", sezioni 2-7
# -------------------------------------------------------------------

@dataclass
class ScreeningThresholds:
    # --- Geo / Impossible Travel (Haversine) ---
    impossible_travel_kmh: float = 900.0   # oltre la velocità aviazione commerciale
    geo_mismatch_hours_window: float = 6.0  # finestra per "last known location"

    # --- Velocity ---
    # NB: calibrate sui training dataset. I valori default sono conservativi.
    # Troppo bassi → hard-flag di attività legittime ricorrenti.
    # Troppo alti  → miss degli attacchi "fast cash" post account-takeover.
    card_velocity_window_min: int = 15
    card_velocity_max_tx: int = 8             # > 8 tx in 15 min dallo stesso sender
    amount_velocity_window_hours: int = 2
    amount_velocity_max_eur: float = 10_000.0

    # --- Economic anomaly ---
    amount_z_hard: float = 4.0               # z-score amount oltre 4σ = outlier severo
    amount_z_soft: float = 2.0
    salary_ratio_hard: float = 1.0           # >100% stipendio mensile in una tx
    salary_ratio_soft: float = 0.5

    # --- NLP / Social Vulnerability ---
    social_vuln_hard: float = 0.8            # score 0-1 dalla pipeline NLP locale
    social_vuln_soft: float = 0.5

    # --- Isolation Forest ---
    iforest_contamination: float = 0.05      # assume ~5% anomalie attese
    iforest_n_estimators: int = 200
    iforest_random_state: int = 42

    # --- Triage final ---
    # Frazione massima di tx che inviamo al Reviewer LLM.
    # Il PDF parla di 1% in produzione; qui siamo più larghi perché
    # i dataset di challenge hanno contamination molto più alta.
    max_llm_review_fraction: float = 0.30

    # Frazione massima di tx nel file di output finale.
    # Il problem statement esclude output che flaggano TUTTO o NULLA,
    # e con <15% di recall il risultato è "invalid".
    max_flag_fraction: float = 0.45
    min_flag_count: int = 1

# Instance globale (si può override nei test)
THRESHOLDS = ScreeningThresholds()

# -------------------------------------------------------------------
# Feature columns passate all'Isolation Forest
# -------------------------------------------------------------------
IFOREST_FEATURES: List[str] = [
    "amount",
    "amount_z",
    "hour",
    "is_night",
    "balance_after",
    "balance_negative",
    "iban_cross_border",
    "geo_mismatch",
    "impossible_travel_kmh_val",
    "amount_vs_monthly_salary",
    "velocity_tx_15min",
    "velocity_amt_2h",
    "social_vulnerability_score",
    "risk_score_heuristic",
]

# -------------------------------------------------------------------
# RegEx patterns per NLP locale (phishing / social engineering)
# Riferimento: PDF sez. 6.2 modello FUDGE
# -------------------------------------------------------------------
SOCIAL_ENG_PATTERNS: Dict[str, List[str]] = {
    "urgency": [
        r"\bact (immediately|now|today)\b",
        r"\burgent(ly)?\b",
        r"\bwithin \d+ (hours?|minutes?)\b",
        r"\baccount (suspended|blocked|frozen|locked)\b",
        r"\bimmediate action required\b",
    ],
    "auth_impersonation": [
        r"\b(fraud|security) (department|team|division)\b",
        r"\bIRS\b|\btax (office|authority)\b",
        r"\bIT (support|helpdesk|department)\b",
        r"\bbank (security|fraud|alert)\b",
    ],
    "credential_request": [
        r"\b(verify|confirm) (your )?(password|otp|pin|credentials)\b",
        r"\bsend (us |me )?your (iban|card|cvv|otp)\b",
        r"\bclick (here|the link) to (verify|confirm|reactivate)\b",
        r"\bgift card(s)? for\b",
    ],
    "financial_action": [
        r"\b(reverse|cancel) (the )?transaction\b",
        r"\bverification (payment|transfer|deposit)\b",
        r"\btransfer.{0,20}(immediately|right away|now)\b",
    ],
    "obfuscated_link": [
        r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",       # raw IP URL
        r"https?://bit\.ly|tinyurl\.com|t\.co|goo\.gl",        # shorteners
        r"https?://[^/\s]*[0-9lIO]{3,}[^/\s]*\.(com|net|org)", # typosquat-like
    ],
}
