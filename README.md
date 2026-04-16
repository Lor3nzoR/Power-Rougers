# Mirror Agent — Power Rougers

Pipeline multi-agente per la **Reply AI Agent Challenge 2026** (Reply Mirror).
Implementa il paradigma **Filter-then-Reason**: un Layer Muscolare deterministico
screma il 70-99% delle transazioni con regole + ML non supervisionato,
passando solo l'ambiguità residua a un Layer Cerebrale basato su LLM.

## 📁 Struttura

```
src/
├── config.py                    # Soglie, modelli LLM, regex patterns
├── main.py                      # CLI entry point
├── loaders/
│   ├── dataset.py               # Caricamento CSV/JSON
│   └── audio.py                 # Trascrizione MP3 con cache (Deus Ex)
├── features/
│   ├── temporal.py              # Velocity checks (card + amount)
│   ├── spatial.py               # Haversine + Impossible Travel
│   ├── economic.py              # amount_z, salary ratio, overdraft
│   ├── behavioral.py            # cross-border IBAN, citizen flags
│   └── nlp_local.py             # RegEx FUDGE + Social Vulnerability Score
├── screening/
│   ├── rule_engine.py           # Regole HARD vs SUSPECT
│   ├── isolation_forest.py      # Anomaly scoring non supervisionato
│   └── triage.py                # PASS / HARD_FLAG / REVIEW
├── agents/
│   ├── llm_client.py            # Wrapper OpenRouter + Langfuse
│   ├── comms_analyst.py         # Summary SMS/mail/audio (Gemini Flash)
│   ├── reviewer.py              # Single-pass (Sonnet) / worker batch (Gemini Lite)
│   ├── arbiter.py               # Filtra falsi positivi (GPT-5-mini)
│   └── user_profiles.py         # Helper profili utenti
├── pipelines/
│   └── filter_then_reason.py    # Orchestratore principale
└── utils/
    ├── logging.py
    └── io.py
```

## 🚀 Quick start

```bash
# 1. Setup ambiente
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
pip install -r requirements.txt

# 2. Configura .env
cp .env.example .env
# edita .env con le tue API key

# 3. Lancia su un dataset
python -m src.main \
    --dataset "datasets/The Truman Show - train" \
    --out outputs/truman_train.txt
```

## ⚙️ Modalità di esecuzione

```bash
# Auto (default): single-pass se <=300 tx REVIEW, batch altrimenti
python -m src.main --dataset "datasets/Brave New World - train" --out outputs/bnw.txt

# Forza single-pass (Claude Sonnet diretto, più costoso ma più accurato)
python -m src.main --dataset "datasets/..." --out outputs/... --mode single

# Forza batch (worker + arbiter, economico)
python -m src.main --dataset "datasets/..." --out outputs/... --mode batch

# Disabilita audio (anche se presenti MP3)
python -m src.main --dataset "datasets/Deus Ex - train" --out outputs/deus.txt --no-audio

# Override session_id
python -m src.main --dataset "..." --out "..." --session-id power-rougers-truman-eval1
```

## 🧠 Architettura Filter-then-Reason

```
     ┌──────────────────────────────────────────────────────────┐
     │  LAYER MUSCOLARE (deterministico, 0 token, ms di latency)│
     │                                                          │
     │  features ──► rule engine ──► iforest ──► triage         │
     │                                             │            │
     │         ┌───────────────┬───────────────────┤            │
     │         ▼               ▼                   ▼            │
     │       PASS          HARD_FLAG            REVIEW          │
     │     (scartato)    (output diretto)   (dossier LLM)       │
     └──────────────────────────────────────────┼───────────────┘
                                                │
     ┌──────────────────────────────────────────▼───────────────┐
     │  LAYER CEREBRALE (LLM, solo su dossier residuale)        │
     │                                                          │
     │  comms_analyst ──► reviewer (single) ──► output          │
     │                 └─► worker batch ──► arbiter ──► output  │
     └──────────────────────────────────────────────────────────┘
```

### Regole HARD (flag automatico senza LLM)
- `impossible_travel`: velocity > 900 km/h tra GPS fix
- `salary_catastrophic`: tx > 100% stipendio mensile
- `velocity_combo`: > 15 tx in 15min OR somma importi > 10k€ in 2h

### Regole SUSPECT (dossier verso LLM)
- `amount_z_extreme`: z-score > 4σ
- `overdraft_large`: saldo negativo + z > 2
- `social_high_plus`: social vulnerability > 0.8 + anomalia
- `geo_mismatch`: city location ≠ ultima GPS city

### Isolation Forest
Contaminazione default 5%. Usa 14 feature (amount_z, velocity, geo, social, ecc.).
Le tx con `iforest_flag=True` vanno a REVIEW se non già HARD.

## 🔒 Safety nets

Il problem statement richiede output validi:
- **mai 0 flag** → fallback a top-10% risk score
- **mai ~tutti flaggati** → cap al 45% (configurabile in `config.py`)
- **mai > 30% verso LLM** → evita di saturare budget token

## 🔑 Modelli LLM (OpenRouter)

| Ruolo | Modello | Quando |
|-------|---------|--------|
| Reviewer single-pass | `anthropic/claude-sonnet-4.5` | dossier ≤ 300 tx |
| Worker batch | `google/gemini-2.5-flash-lite` | chunk da 150 tx |
| Comms Analyst | `google/gemini-2.5-flash` | summary SMS/mail |
| Arbiter | `openai/gpt-5-mini` | filtra FP dei worker |
| ASR (Deus Ex) | `whisper-1` | trascrizione MP3 |

Modifica `src/config.py` → `MODELS` per cambiare.

## 📊 Langfuse tracking

- `LANGFUSE_HOST` punta a `https://challenges.reply.com/langfuse` (già default)
- `session_id` generato automaticamente come `<TEAM>-<dataset>` (senza spazi)
- Tutte le chiamate LLM sono tracciate con span nominati
- Il flush viene chiamato automaticamente a fine pipeline

## 🧪 Testing con training dataset

Il challenge permette submission illimitate su training; una sola per eval.
Usa i training per calibrare le soglie in `src/config.py`:

```python
THRESHOLDS = ScreeningThresholds(
    iforest_contamination=0.05,     # alza se sotto-flaggi
    amount_z_hard=4.0,              # abbassa se miss troppe frodi grandi
    max_llm_review_fraction=0.30,   # cruciale per budget token
    max_flag_fraction=0.45,         # limite output finale
)
```

## 🛠 Tips per la challenge

1. **Prima di eval**, riprodurre training su stesso dataset e verificare:
   - `final_flags / total_tx` tra 15-40%
   - `hard_flags` > 0 (vuol dire che il layer muscolare funziona)
   - Budget token speso < 1/3 del totale stage
2. **Non disabilitare audio** su Deus Ex senza motivo: contiene phishing vocale
3. **Session ID**: se il dataset name ha spazi, vengono auto-convertiti in `-`
4. **Re-run sicuro**: la trascrizione audio è cachata in `datasets/<ds>/audio/.transcripts.json`
