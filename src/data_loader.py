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


def _parse_snapshots(table_text: str) -> pd.DataFrame:
    if not table_text.strip():
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

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


def _normalize_trades(trades_raw: list[dict]) -> pd.DataFrame:
    if not trades_raw:
        return pd.DataFrame(columns=["timestamp", "symbol", "price", "quantity", "buyer", "seller", "side", "currency"])
    trades = pd.DataFrame(trades_raw).copy()
    trades.columns = [_normalize_column_name(c) for c in trades.columns]
    for col in ["timestamp", "price", "quantity"]:
        if col in trades.columns:
            trades[col] = pd.to_numeric(trades[col], errors="coerce")
    if "symbol" not in trades.columns:
        trades["symbol"] = ""
    if "buyer" not in trades.columns:
        trades["buyer"] = ""
    if "seller" not in trades.columns:
        trades["seller"] = ""
    if "currency" not in trades.columns:
        trades["currency"] = ""
    trades["side"] = "OTHER"
    trades.loc[trades["buyer"] == "SUBMISSION", "side"] = "BUY"
    trades.loc[trades["seller"] == "SUBMISSION", "side"] = "SELL"
    return trades.sort_values("timestamp").reset_index(drop=True)


def _normalize_logs(logs_raw: list[dict]) -> pd.DataFrame:
    if not logs_raw:
        return pd.DataFrame(columns=["timestamp", "sandboxlog", "lambdalog"])
    logs = pd.DataFrame(logs_raw).copy()
    logs.columns = [_normalize_column_name(c) for c in logs.columns]
    if "timestamp" in logs.columns:
        logs["timestamp"] = pd.to_numeric(logs["timestamp"], errors="coerce")
    return logs.sort_values("timestamp").reset_index(drop=True)


def load_simulation_data(file_obj: BinaryIO | io.BytesIO | TextIO) -> dict[str, pd.DataFrame | dict]:
    raw_text = _read_text(file_obj)
    payload: dict = {}
    table_text = raw_text
    stripped = raw_text.lstrip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
            activities_log = payload.get("activitiesLog")
            if isinstance(activities_log, str):
                table_text = activities_log
        except json.JSONDecodeError:
            payload = {}
            table_text = raw_text

    snapshots = _parse_snapshots(table_text)
    trades = _normalize_trades(payload.get("tradeHistory", [])) if payload else _normalize_trades([])
    logs = _normalize_logs(payload.get("logs", [])) if payload else _normalize_logs([])
    meta = {"submissionId": payload.get("submissionId")} if payload else {}
    return {"snapshots": snapshots, "trades": trades, "logs": logs, "meta": meta}


def load_csv(file_obj: BinaryIO | io.BytesIO | TextIO) -> pd.DataFrame:
    return load_simulation_data(file_obj)["snapshots"]
