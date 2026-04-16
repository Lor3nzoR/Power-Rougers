"""
Mirror Agent v2 — pipeline multi-agente fraud detection
Adattata allo schema reale dei dataset (lowercase cols, JSON aux files, description field).

Uso:
    export OPENROUTER_API_KEY=...
    export LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=...
    python src/mirror_agent.py --dataset "datasets/The Truman Show - train" --out outputs/truman_train.txt

Requirements:
    pip install pandas numpy openai langfuse python-dotenv
"""

import os, json, argparse, math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# =============================================================
# SETUP LANGFUSE & OPENAI
# =============================================================
os.environ.setdefault("LANGFUSE_HOST", "https://challenges.reply.com/langfuse")

from langfuse.openai import OpenAI
from langfuse import observe, get_client, propagate_attributes # <--- Aggiunto propagate_attributes

langfuse = get_client()

MODELS = {
    "big":   "anthropic/claude-sonnet-4.5",       
    "cheap": "google/gemini-2.5-flash-lite",      
    "ctx":   "google/gemini-2.5-flash",           
    "arb":   "openai/gpt-5-mini",                 
}

client = OpenAI(base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"])

@observe(as_type="span") # <--- Trasformato in span, perché OpenAI genera già la "generation"
def llm(model, system, user, session_id, name=None, json_mode=True, max_tokens=None):
    lf = get_client()
    
    # Rinominiamo lo span corrente usando il nuovo metodo della v4
    custom_name = name or model.split("/")[-1]
    lf.update_current_span(name=custom_name)
    
    kwargs = dict(
        model=model,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.1,
    )
    
    if json_mode: 
        kwargs["response_format"] = {"type":"json_object"}
    if max_tokens: 
        kwargs["max_tokens"] = max_tokens
        
    # In Langfuse v4+ il session_id si propaga tramite context manager
    with propagate_attributes(session_id=session_id):
        return client.chat.completions.create(**kwargs).choices[0].message.content

# =============================================================
# 1. LOADING (schema reale)
# =============================================================

def load_dataset(ds_path: Path) -> Dict[str, Any]:
    """Carica tutto. Gestisce file mancanti gracefully (non tutti i livelli hanno tutto)."""
    def _maybe_json(name):
        p = ds_path / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []

    tx = pd.read_csv(ds_path / "transactions.csv")
    tx["timestamp"] = pd.to_datetime(tx["timestamp"], errors="coerce")
    tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce").fillna(0)
    tx["balance_after"] = pd.to_numeric(tx["balance_after"], errors="coerce")

    users = _maybe_json("users.json")
    locations = _maybe_json("locations.json")
    sms = _maybe_json("sms.json")
    mails = _maybe_json("mails.json")

    # Normalizza locations in DataFrame per lookup veloce
    if locations:
        loc_df = pd.DataFrame(locations)
        loc_df["timestamp"] = pd.to_datetime(loc_df["timestamp"], errors="coerce")
        loc_df["lat"] = pd.to_numeric(loc_df["lat"], errors="coerce")
        loc_df["lng"] = pd.to_numeric(loc_df["lng"], errors="coerce")
    else:
        loc_df = pd.DataFrame()

    # Index users by IBAN (l'unico identifier stabile) e derive biotag mapping
    iban_to_user = {}
    for u in users:
        if u.get("iban"): iban_to_user[u["iban"]] = u

    # sender_id ↔ biotag quando coincidono (il citizen è sia nelle tx che nelle locations)
    biotags = set(loc_df["biotag"].unique()) if len(loc_df) else set()

    return dict(tx=tx, users=users, locations=loc_df, sms=sms, mails=mails,
                iban_to_user=iban_to_user, biotags=biotags)


# =============================================================
# 2. FEATURE ENGINEERING (sempre utile anche a dataset piccolo)
# =============================================================

def compute_features(data: Dict[str, Any]) -> pd.DataFrame:
    tx = data["tx"].copy()
    loc_df = data["locations"]
    biotags = data["biotags"]
    iban_to_user = data["iban_to_user"]

    # Time features
    tx["hour"] = tx["timestamp"].dt.hour.fillna(-1).astype(int)
    tx["is_night"] = tx["hour"].between(0, 5) | (tx["hour"] == 23)
    tx["dow"] = tx["timestamp"].dt.dayofweek.fillna(-1).astype(int)

    # Amount z-score per sender
    sstats = tx.groupby("sender_id")["amount"].agg(["mean","std","count"])
    sstats.columns = ["sender_mean","sender_std","sender_count"]
    tx = tx.merge(sstats, left_on="sender_id", right_index=True, how="left")
    std_fb = tx["amount"].std() or 1.0
    tx["amount_z"] = ((tx["amount"] - tx["sender_mean"]) /
                      tx["sender_std"].replace(0, np.nan).fillna(std_fb)).fillna(0)

    # Balance anomaly
    tx["balance_negative"] = tx["balance_after"].fillna(0) < 0

    # IBAN cross-border (primi 2 char)
    def _cb(r):
        s,rc = r.get("sender_iban"), r.get("recipient_iban")
        if isinstance(s,str) and isinstance(rc,str) and len(s)>=2 and len(rc)>=2:
            return s[:2] != rc[:2]
        return False
    tx["iban_cross_border"] = tx.apply(_cb, axis=1)

    # Sender is a tracked citizen?
    tx["sender_is_citizen"] = tx["sender_id"].isin(biotags)
    tx["recipient_is_citizen"] = tx["recipient_id"].isin(biotags)

    # Amount vs sender's salary (se citizen)
    def _salary(iban):
        u = iban_to_user.get(iban)
        return u.get("salary") if u else None
    tx["sender_salary"] = tx["sender_iban"].apply(_salary)
    tx["amount_vs_monthly_salary"] = tx.apply(
        lambda r: r["amount"] / (r["sender_salary"]/12) if r["sender_salary"] else np.nan, axis=1
    )

    # Geo mismatch: in-person payment con location != GPS del citizen a quel timestamp
    def _geo_mismatch(row):
        if not isinstance(row["location"], str) or pd.isna(row["location"]): return False
        if not row["sender_is_citizen"]: return False
        if not len(loc_df): return False
        # trova il punto GPS più vicino in tempo per quel biotag
        same = loc_df[loc_df["biotag"] == row["sender_id"]]
        if not len(same): return False
        delta = (same["timestamp"] - row["timestamp"]).abs()
        closest = same.loc[delta.idxmin()]
        if pd.isna(closest.get("city")): return False
        return closest["city"].lower() not in row["location"].lower() and \
               row["location"].lower() not in closest["city"].lower()
    tx["geo_mismatch"] = tx.apply(_geo_mismatch, axis=1)

    # Composite risk score (pesi iniziali, da tunare)
    tx["risk_score"] = (
          (tx["amount_z"].abs() > 2).astype(int) * 2
        + (tx["amount_z"].abs() > 4).astype(int) * 2
        + tx["is_night"].astype(int)
        + tx["balance_negative"].astype(int) * 3
        + tx["iban_cross_border"].astype(int)
        + tx["geo_mismatch"].astype(int) * 4
        + (tx["amount_vs_monthly_salary"] > 0.5).astype(int) * 2
    )

    return tx


# =============================================================
# 3. CONTEXT BUNDLE (user profiles + SMS/mail summary)
# =============================================================

def build_user_profiles(data) -> str:
    users = data["users"]
    if not users: return "No user profiles available."
    out = []
    for u in users:
        out.append({
            "name": f'{u.get("first_name","")} {u.get("last_name","")}'.strip(),
            "iban": u.get("iban"),
            "birth_year": u.get("birth_year"),
            "job": u.get("job"),
            "salary_eur": u.get("salary"),
            "residence": u.get("residence", {}),
            "profile_snippet": (u.get("description") or "")[:600],
        })
    return json.dumps(out, ensure_ascii=False, indent=2)


def summarize_comms(data, session_id) -> str:
    """Se i mail/sms sono pochi → li passiamo raw sintetizzati;
       se sono tanti → mandiamo un LLM cheap a estrarre segnali."""
    sms = data["sms"]
    mails = data["mails"]
    if not sms and not mails: return "No communications available."

    # Su dataset 1 (162 sms, 8 mail) stiamo sotto i 40k token tranquilli
    sample_sms = [s["sms"][:300] for s in sms[:40]]
    sample_mails = [m["mail"][:800] for m in mails[:10]]
    payload = json.dumps({"sms_samples": sample_sms, "mail_samples": sample_mails}, ensure_ascii=False)

    system = """You are a communications analyst. Given SMS and email samples from citizens,
extract signals relevant to fraud detection. Return ONLY JSON:
{
  "phishing_signals": ["short desc", ...],
  "legitimate_counterparties": ["company/person names mentioned in normal context"],
  "suspicious_keywords": ["word/phrase", ...],
  "notes": "2-3 sentences"
}"""
    raw = llm(MODELS["ctx"], system, payload, session_id, name="comms-summary")
    return raw


# =============================================================
# 4. SINGLE-PASS REVIEWER (dataset piccoli) / BATCH MODE (grandi)
# =============================================================

REVIEWER_SYS = """You are the lead fraud-detection agent. You receive:
- user_profiles: 1-3 citizens with salary, job, residence, behavioral description
- comms_summary: signals extracted from their SMS/emails (phishing indicators, known legitimate counterparties)
- transactions: list of ALL transactions (each with features + free-text description)

Output ONLY JSON:
{
  "flagged": [
    {"tx_id": "...", "confidence": 0.0-1.0, "reason": "<15 words>"}
  ]
}

Judge each transaction on:
1. Coherence with user profile (e.g. 40yo office clerk on 34k€ suddenly wires 2500€ to a foreign IBAN at 3am with vague description → suspicious)
2. The `description` field — normal ones say "Salary", "Rent", "Groceries". Fraud often has generic/missing/off-topic descriptions
3. Counterparty legitimacy — recipients mentioned positively in comms_summary are likely legitimate; new foreign IBANs are suspicious
4. geo_mismatch = true is a STRONG signal (GPS places citizen elsewhere)
5. iban_cross_border alone is not enough; combined with other flags yes
6. Economic asymmetry: be readier to flag high-amount borderline cases

Constraints:
- Never flag >40% of transactions
- Never flag 0 transactions (if truly nothing is off, flag the single most anomalous)
- Aim for precision; false positives cost too
"""


def run_single_pass(feats: pd.DataFrame, profiles: str, comms: str, session_id: str) -> List[dict]:
    cols = ["transaction_id","sender_id","recipient_id","transaction_type","amount","location",
            "sender_iban","recipient_iban","balance_after","description","timestamp",
            "hour","is_night","amount_z","balance_negative","iban_cross_border",
            "sender_is_citizen","geo_mismatch","amount_vs_monthly_salary","risk_score"]
    cols = [c for c in cols if c in feats.columns]
    tx_records = feats[cols].copy()
    for c in tx_records.select_dtypes("float").columns:
        tx_records[c] = tx_records[c].round(3)
    tx_records["timestamp"] = tx_records["timestamp"].astype(str)
    tx_list = tx_records.to_dict(orient="records")

    # --- INIZIO NUOVO BLOCCO ANTI-CRASH PER I DATI IN INGRESSO ---
    
    # 1. Parsing sicuro per i profili
    try:
        parsed_profiles = json.loads(profiles)
    except Exception:
        parsed_profiles = profiles

    # 2. Parsing sicuro per il riassunto comunicazioni (comms)
    comms_clean = comms.strip()
    if comms_clean.startswith("```"):
        comms_clean = "\n".join(comms_clean.split("\n")[1:-1]).strip()
        
    try:
        parsed_comms = json.loads(comms_clean)
    except json.JSONDecodeError:
        print("\n[WARN] Il summary di Gemini non è un JSON perfetto. Lo passo a Claude come testo grezzo.")
        parsed_comms = comms_clean

    # 3. Creazione del payload finale a prova di crash
    payload = json.dumps({
        "user_profiles": parsed_profiles,
        "comms_summary": parsed_comms,
        "transactions": tx_list,
    }, ensure_ascii=False)
    
    # --- FINE NUOVO BLOCCO ---

    raw = llm(MODELS["big"], REVIEWER_SYS, payload, session_id, name="single-pass-reviewer")
    
    # --- BLOCCO ANTI-CRASH PER L'OUTPUT DI CLAUDE (che avevi già messo) ---
    raw_clean = raw.strip()
    if raw_clean.startswith("```"):
        raw_clean = "\n".join(raw_clean.split("\n")[1:-1]).strip()
        
    try:
        data_json = json.loads(raw_clean)
        return data_json.get("flagged", [])
    except json.JSONDecodeError:
        print(f"\n[ERRORE CRITICO] Il modello non ha restituito un JSON valido.")
        print(f"Output grezzo ricevuto:\n{raw}\n")
        return []


# =============================================================
# 5. MAIN
# =============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--session-id", default=None, help="Langfuse session ID. Default: <TEAM_NAME>-<dataset-folder-name>")
    ap.add_argument("--mode", choices=["auto","single","batch"], default="auto")
    args = ap.parse_args()
    
    if not args.session_id:
        team = os.environ.get("TEAM_NAME", "team").replace(" ", "-")
        ds_name = Path(args.dataset).name.replace(" ", "-")
        args.session_id = f"{team}-{ds_name}"
    assert " " not in args.session_id, "session-id non deve contenere spazi"

    data = load_dataset(Path(args.dataset))
    n = len(data["tx"])
    print(f"Loaded: {n} tx, {len(data['users'])} users, "
          f"{len(data['locations'])} locs, {len(data['sms'])} sms, {len(data['mails'])} mails")

    feats = compute_features(data)

    mode = args.mode
    if mode == "auto":
        mode = "single" if n <= 300 else "batch"
    print(f"Mode: {mode}")

    if mode == "single":
        profiles = build_user_profiles(data)
        comms = summarize_comms(data, args.session_id) if (data["sms"] or data["mails"]) else "{}"
        flagged = run_single_pass(feats, profiles, comms, args.session_id)

    else:
        # batch mode — riusa la pipeline v1 (screening + worker + arbiter)
        # placeholder: fall back to a simple top-risk selection
        raise NotImplementedError("Batch mode: reuse the v1 skeleton (orchestrator+worker+arbiter)")

    # Safety net
    ids_all = set(data["tx"]["transaction_id"])
    final = [f["tx_id"] for f in flagged if f.get("tx_id") in ids_all]
    final = list(dict.fromkeys(final))  # dedupe

    if not final:
        print("[WARN] Empty flag list. Fallback to top-risk 10%.")
        final = feats.nlargest(max(3, n//10), "risk_score")["transaction_id"].tolist()
    if len(final) >= n * 0.9:
        print(f"[WARN] {len(final)}/{n} flagged. Truncating to 30%.")
        final = final[:max(1, int(n*0.3))]

    Path(args.out).write_text("\n".join(map(str, final)))
    print(f"\n✓ {len(final)}/{n} flagged → {args.out}")
    print(f"  session_id: {args.session_id}")
    print("\nTop 5 reasons:")
    for f in flagged[:5]:
        print(f"  {f.get('tx_id','?')[:12]}… conf={f.get('confidence','?'):.2f} {f.get('reason','')}")

    langfuse.flush()


if __name__ == "__main__":
    main()