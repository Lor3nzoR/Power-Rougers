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
from langfuse import observe, get_client, propagate_attributes

# Inizializza il client di Langfuse
langfuse = get_client()

MODELS = {
    "big":   "anthropic/claude-sonnet-4.5",       # orchestrator / single-shot su dataset piccoli
    "cheap": "google/gemini-2.5-flash-lite",      # worker batch su dataset grandi
    "ctx":   "google/gemini-2.5-flash",           # context fusion (lettura SMS/mail)
    "arb":   "openai/gpt-5-mini",                 # arbiter cost-sensitive
}

client = OpenAI(base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"])

@observe(as_type="span")
def llm(model, system, user, session_id, name=None, json_mode=True, max_tokens=None):
    lf = get_client()
    
    # Rinominiamo lo span corrente per leggerlo bene in dashboard
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
        
    # Propaga il session_id alla chiamata di OpenAI (Generation)
    with propagate_attributes(session_id=session_id):
        return client.chat.completions.create(**kwargs).choices[0].message.content


# =============================================================
# 1. LOADING (schema reale)
# =============================================================

def load_dataset(ds_path: Path) -> Dict[str, Any]:
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

    if locations:
        loc_df = pd.DataFrame(locations)
        loc_df["timestamp"] = pd.to_datetime(loc_df["timestamp"], errors="coerce")
        loc_df["lat"] = pd.to_numeric(loc_df["lat"], errors="coerce")
        loc_df["lng"] = pd.to_numeric(loc_df["lng"], errors="coerce")
    else:
        loc_df = pd.DataFrame()

    iban_to_user = {}
    for u in users:
        if u.get("iban"): iban_to_user[u["iban"]] = u

    biotags = set(loc_df["biotag"].unique()) if len(loc_df) else set()

    return dict(tx=tx, users=users, locations=loc_df, sms=sms, mails=mails,
                iban_to_user=iban_to_user, biotags=biotags)


# =============================================================
# 2. FEATURE ENGINEERING
# =============================================================

def compute_features(data: Dict[str, Any]) -> pd.DataFrame:
    tx = data["tx"].copy()
    loc_df = data["locations"]
    biotags = data["biotags"]
    iban_to_user = data["iban_to_user"]

    tx["hour"] = tx["timestamp"].dt.hour.fillna(-1).astype(int)
    tx["is_night"] = tx["hour"].between(0, 5) | (tx["hour"] == 23)
    tx["dow"] = tx["timestamp"].dt.dayofweek.fillna(-1).astype(int)

    sstats = tx.groupby("sender_id")["amount"].agg(["mean","std","count"])
    sstats.columns = ["sender_mean","sender_std","sender_count"]
    tx = tx.merge(sstats, left_on="sender_id", right_index=True, how="left")
    std_fb = tx["amount"].std() or 1.0
    tx["amount_z"] = ((tx["amount"] - tx["sender_mean"]) /
                      tx["sender_std"].replace(0, np.nan).fillna(std_fb)).fillna(0)

    tx["balance_negative"] = tx["balance_after"].fillna(0) < 0

    def _cb(r):
        s,rc = r.get("sender_iban"), r.get("recipient_iban")
        if isinstance(s,str) and isinstance(rc,str) and len(s)>=2 and len(rc)>=2:
            return s[:2] != rc[:2]
        return False
    tx["iban_cross_border"] = tx.apply(_cb, axis=1)

    tx["sender_is_citizen"] = tx["sender_id"].isin(biotags)
    tx["recipient_is_citizen"] = tx["recipient_id"].isin(biotags)

    def _salary(iban):
        u = iban_to_user.get(iban)
        return u.get("salary") if u else None
    tx["sender_salary"] = tx["sender_iban"].apply(_salary)
    tx["amount_vs_monthly_salary"] = tx.apply(
        lambda r: r["amount"] / (r["sender_salary"]/12) if r["sender_salary"] else np.nan, axis=1
    )

    def _geo_mismatch(row):
        if not isinstance(row["location"], str) or pd.isna(row["location"]): return False
        if not row["sender_is_citizen"]: return False
        if not len(loc_df): return False
        same = loc_df[loc_df["biotag"] == row["sender_id"]]
        if not len(same): return False
        delta = (same["timestamp"] - row["timestamp"]).abs()
        closest = same.loc[delta.idxmin()]
        if pd.isna(closest.get("city")): return False
        return closest["city"].lower() not in row["location"].lower() and \
               row["location"].lower() not in closest["city"].lower()
    tx["geo_mismatch"] = tx.apply(_geo_mismatch, axis=1)

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
# 3. CONTEXT BUNDLE
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
    sms = data["sms"]
    mails = data["mails"]
    if not sms and not mails: return "No communications available."

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
# 4. SYSTEM PROMPTS
# =============================================================

REVIEWER_SYS = """You are the lead fraud-detection agent. You receive:
- user_profiles: citizens with salary, job, residence
- comms_summary: signals extracted from their SMS/emails
- transactions: list of ALL transactions

Output ONLY JSON:
{
  "flagged": [
    {"tx_id": "...", "confidence": 0.0-1.0, "reason": "<15 words>"}
  ]
}

Judge each transaction on:
1. Coherence with user profile
2. The `description` field
3. Counterparty legitimacy (recipients mentioned positively in comms are likely legitimate)
4. geo_mismatch = true is a STRONG signal
5. Economic asymmetry

Constraints:
- Never flag >40% of transactions
- Never flag 0 transactions
- Aim for precision
"""

WORKER_SYS = """You are a junior fraud-detection worker. You receive user profiles, a comms summary, and a batch of transactions.
Your job is to FLAG ANY potentially suspicious transaction. Be slightly aggressive in your flagging; a senior arbiter will review your choices.
Pay attention to: geo_mismatch, negative balances, and high amount_z.
Output ONLY JSON:
{
  "flagged": [
    {"tx_id": "...", "confidence": 0.0-1.0, "reason": "<10 words>"}
  ]
}"""

ARBITER_SYS = """You are the Lead Fraud Arbiter. A junior agent has flagged the following transactions as potentially fraudulent.
Review these specific transactions against the user profiles and communications summary.
Your job is to FILTER OUT false positives. Keep ONLY the genuinely suspicious ones based on economic asymmetry, context, and behavioral anomalies. 
Aim for high precision. Output ONLY JSON:
{
  "flagged": [
    {"tx_id": "...", "confidence": 0.0-1.0, "reason": "<15 words>"}
  ]
}"""


# =============================================================
# 5. EXECUTION MODES (SINGLE & BATCH)
# =============================================================

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

    try:
        parsed_profiles = json.loads(profiles)
    except Exception:
        parsed_profiles = profiles

    comms_clean = comms.strip()
    if comms_clean.startswith("```"):
        comms_clean = "\n".join(comms_clean.split("\n")[1:-1]).strip()
        
    try:
        parsed_comms = json.loads(comms_clean)
    except json.JSONDecodeError:
        print("\n[WARN] Il summary di Gemini non è un JSON perfetto. Lo passo a Claude come testo grezzo.")
        parsed_comms = comms_clean

    payload = json.dumps({
        "user_profiles": parsed_profiles,
        "comms_summary": parsed_comms,
        "transactions": tx_list,
    }, ensure_ascii=False)

    raw = llm(MODELS["big"], REVIEWER_SYS, payload, session_id, name="single-pass-reviewer")
    
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


def run_batch_mode(feats: pd.DataFrame, profiles: str, comms: str, session_id: str) -> List[dict]:
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

    try:
        parsed_profiles = json.loads(profiles)
    except Exception:
        parsed_profiles = profiles

    comms_clean = comms.strip()
    if comms_clean.startswith("```"):
        comms_clean = "\n".join(comms_clean.split("\n")[1:-1]).strip()
    try:
        parsed_comms = json.loads(comms_clean)
    except json.JSONDecodeError:
        parsed_comms = comms_clean

    # 1. FASE WORKER: Chunking
    CHUNK_SIZE = 150
    worker_flags = []
    
    print(f"  [Batch] Avvio fase Worker ({math.ceil(len(tx_list)/CHUNK_SIZE)} chunk)...")
    for i in range(0, len(tx_list), CHUNK_SIZE):
        chunk = tx_list[i:i+CHUNK_SIZE]
        payload = json.dumps({
            "user_profiles": parsed_profiles,
            "comms_summary": parsed_comms,
            "transactions": chunk,
        }, ensure_ascii=False)
        
        raw = llm(MODELS["cheap"], WORKER_SYS, payload, session_id, name=f"worker-chunk-{i//CHUNK_SIZE}")
        
        raw_clean = raw.strip()
        if raw_clean.startswith("```"):
            raw_clean = "\n".join(raw_clean.split("\n")[1:-1]).strip()
            
        try:
            data = json.loads(raw_clean)
            worker_flags.extend(data.get("flagged", []))
        except json.JSONDecodeError:
            print(f"  [WARN] Il Worker (chunk {i//CHUNK_SIZE}) ha fallito la generazione JSON.")

    print(f"  [Batch] I Worker hanno sollevato {len(worker_flags)} possibili frodi. Passo all'Arbitro...")
    if not worker_flags:
        return []

    # 2. FASE ARBITER: Sintesi finale
    flagged_ids = {f.get("tx_id") for f in worker_flags if f.get("tx_id")}
    arbiter_tx = [t for t in tx_list if t["transaction_id"] in flagged_ids]
    
    payload_arbiter = json.dumps({
        "user_profiles": parsed_profiles,
        "comms_summary": parsed_comms,
        "flagged_transactions_to_review": arbiter_tx,
    }, ensure_ascii=False)
    
    raw_arb = llm(MODELS["arb"], ARBITER_SYS, payload_arbiter, session_id, name="arbiter-review")
    
    raw_arb_clean = raw_arb.strip()
    if raw_arb_clean.startswith("```"):
        raw_arb_clean = "\n".join(raw_arb_clean.split("\n")[1:-1]).strip()
        
    try:
        final_data = json.loads(raw_arb_clean)
        return final_data.get("flagged", [])
    except json.JSONDecodeError:
        print("\n[ERRORE CRITICO] L'Arbiter non ha restituito un JSON valido. Ricado sui flag dei worker.")
        return worker_flags


# =============================================================
# 6. MAIN
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
        profiles = build_user_profiles(data)
        comms = summarize_comms(data, args.session_id) if (data["sms"] or data["mails"]) else "{}"
        flagged = run_batch_mode(feats, profiles, comms, args.session_id)

    # Safety net e deduplicazione
    ids_all = set(data["tx"]["transaction_id"])
    final = [f["tx_id"] for f in flagged if f.get("tx_id") in ids_all]
    final = list(dict.fromkeys(final))  

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