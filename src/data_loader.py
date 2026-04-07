from __future__ import annotations

import io
import json
from typing import BinaryIO, TextIO

import pandas as pd


EXPECTED_COLUMNS = [
    "day",
    "timestamp",
    "product",
    "bid_price_1",
    "bid_volume_1",
    "bid_price_2",
    "bid_volume_2",
    "bid_price_3",
    "bid_volume_3",
    "ask_price_1",
    "ask_volume_1",
    "ask_price_2",
    "ask_volume_2",
    "ask_price_3",
    "ask_volume_3",
    "mid_price",
    "profit_and_loss",
]


def _normalize_column_name(name: str) -> str:
    return name.strip().replace("\\", "").lower()


def _read_text(file_obj: BinaryIO | io.BytesIO | TextIO) -> str:
    if hasattr(file_obj, "getvalue"):
        raw = file_obj.getvalue()
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return str(raw)
    if hasattr(file_obj, "read"):
        raw = file_obj.read()
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return str(raw)
    raise TypeError("Unsupported file object type.")


def _extract_table_text(raw_text: str) -> str:
    stripped = raw_text.lstrip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
            activities_log = payload.get("activitiesLog")
            if isinstance(activities_log, str) and activities_log.strip():
                return activities_log
        except json.JSONDecodeError:
            pass
    return raw_text


def load_csv(file_obj: BinaryIO | io.BytesIO | TextIO) -> pd.DataFrame:
    raw_text = _read_text(file_obj)
    table_text = _extract_table_text(raw_text)

    # Logs are semicolon-separated, but we keep comma fallback for compatibility.
    df = pd.read_csv(io.StringIO(table_text), sep=";|,", engine="python", on_bad_lines="skip")
    df.columns = [_normalize_column_name(c) for c in df.columns]
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[EXPECTED_COLUMNS].copy()
    numeric_cols = [c for c in EXPECTED_COLUMNS if c != "product"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Core fields are mandatory; deeper book levels may be sparse.
    df = df.dropna(subset=["day", "timestamp", "product", "mid_price"])
    df["day"] = df["day"].astype(int)
    df["timestamp"] = df["timestamp"].astype(int)
    df["product"] = df["product"].astype(str).str.strip()

    # Keep integer semantics for book volumes while allowing missing sparse levels.
    volume_cols = [
        "bid_volume_1",
        "bid_volume_2",
        "bid_volume_3",
        "ask_volume_1",
        "ask_volume_2",
        "ask_volume_3",
    ]
    for col in volume_cols:
        df[col] = df[col].round().astype("Int64")

    df = df.sort_values(["product", "day", "timestamp"]).reset_index(drop=True)
    return df
