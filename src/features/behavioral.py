"""
Feature comportamentali e attributi.

- iban_cross_border: country code IBAN sender ≠ recipient
- sender_is_citizen: sender presente nelle locations (ha BioTag)
- tx_type one-hot
"""
from __future__ import annotations
import pandas as pd
from typing import Set


def _country_code(iban: object) -> str | None:
    if isinstance(iban, str) and len(iban) >= 2 and iban[:2].isalpha():
        return iban[:2].upper()
    return None


def add_behavioral_features(tx: pd.DataFrame, user_biotags: Set[str]) -> pd.DataFrame:
    tx = tx.copy()

    tx["sender_country"]    = tx["sender_iban"].apply(_country_code)    if "sender_iban"    in tx else None
    tx["recipient_country"] = tx["recipient_iban"].apply(_country_code) if "recipient_iban" in tx else None

    tx["iban_cross_border"] = False
    mask = tx["sender_country"].notna() & tx["recipient_country"].notna()
    tx.loc[mask, "iban_cross_border"] = (
        tx.loc[mask, "sender_country"] != tx.loc[mask, "recipient_country"]
    )

    tx["sender_is_citizen"]    = tx["sender_id"].isin(user_biotags)
    tx["recipient_is_citizen"] = tx["recipient_id"].isin(user_biotags) if "recipient_id" in tx else False

    return tx
