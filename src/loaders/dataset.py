"""
Caricamento dataset challenge.
Schema atteso (lowercase cols come da dataset reali):
- transactions.csv         (obbligatorio)
- users.json               (opzionale)
- locations.json           (opzionale)
- sms.json                 (opzionale)
- mails.json               (opzionale)
- audio/*.mp3              (opzionale, solo Deus Ex)
"""
from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List
import pandas as pd

from src.utils.logging import log


@dataclass
class Dataset:
    path: Path
    tx: pd.DataFrame
    users: List[Dict[str, Any]] = field(default_factory=list)
    locations: pd.DataFrame = field(default_factory=pd.DataFrame)
    sms: List[Dict[str, Any]] = field(default_factory=list)
    mails: List[Dict[str, Any]] = field(default_factory=list)
    audio_files: List[Path] = field(default_factory=list)

    # Lookups derivati
    iban_to_user: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    user_biotags: set = field(default_factory=set)

    @property
    def n_tx(self) -> int:
        return len(self.tx)

    def summary(self) -> str:
        return (f"{self.n_tx} tx · {len(self.users)} users · "
                f"{len(self.locations)} locs · {len(self.sms)} sms · "
                f"{len(self.mails)} mails · {len(self.audio_files)} audio")


def _maybe_json(p: Path) -> Any:
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning(f"Failed to parse {p.name}: {e}")
    return []


def load_dataset(ds_path: str | Path) -> Dataset:
    ds_path = Path(ds_path)
    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {ds_path}")

    tx_path = ds_path / "transactions.csv"
    if not tx_path.exists():
        raise FileNotFoundError(f"Missing transactions.csv in {ds_path}")

    tx = pd.read_csv(tx_path)
    tx["timestamp"]     = pd.to_datetime(tx["timestamp"], errors="coerce")
    tx["amount"]        = pd.to_numeric(tx["amount"], errors="coerce").fillna(0.0)
    tx["balance_after"] = pd.to_numeric(tx["balance_after"], errors="coerce")

    users = _maybe_json(ds_path / "users.json") or []
    sms   = _maybe_json(ds_path / "sms.json") or []
    mails = _maybe_json(ds_path / "mails.json") or []

    locs_raw = _maybe_json(ds_path / "locations.json") or []
    if locs_raw:
        loc_df = pd.DataFrame(locs_raw)
        if "timestamp" in loc_df:
            loc_df["timestamp"] = pd.to_datetime(loc_df["timestamp"], errors="coerce")
        for c in ("lat", "lng"):
            if c in loc_df:
                loc_df[c] = pd.to_numeric(loc_df[c], errors="coerce")
    else:
        loc_df = pd.DataFrame()

    iban_to_user = {u["iban"]: u for u in users if u.get("iban")}
    user_biotags = set(loc_df["biotag"].dropna().unique()) if "biotag" in loc_df else set()

    audio_dir = ds_path / "audio"
    audio_files = sorted(audio_dir.glob("*.mp3")) if audio_dir.exists() else []

    ds = Dataset(
        path=ds_path, tx=tx, users=users, locations=loc_df,
        sms=sms, mails=mails, audio_files=audio_files,
        iban_to_user=iban_to_user, user_biotags=user_biotags,
    )
    log.info(f"Loaded [{ds_path.name}]: {ds.summary()}")
    return ds
