"""Raw data loading and canonical panel construction utilities.

This module standardizes dates and instrument names, loads OHLCV and primary
signal files, and merges them into the panel used by later pipeline steps.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ASSET_CLASS, CourseworkConfig


def standardize_date_instrument_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    lower_cols = {col: col.strip().lower() for col in out.columns}
    out = out.rename(columns=lower_cols)
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
    if "instrument" in out.columns:
        out["instrument"] = out["instrument"].astype(str).str.lower()
    return out


def load_ohlcv(path: str | Path) -> pd.DataFrame:
    ohlcv = pd.read_csv(path)
    ohlcv = standardize_date_instrument_columns(ohlcv)
    required = {"date", "instrument", "open", "high", "low", "close", "volume", "open_interest"}
    missing = required - set(ohlcv.columns)
    if missing:
        raise ValueError(f"OHLCV file is missing required columns: {sorted(missing)}")
    return ohlcv.sort_values(["instrument", "date"]).reset_index(drop=True)


def load_primary_signals(path: str | Path) -> pd.DataFrame:
    signals = pd.read_csv(path)
    signals = standardize_date_instrument_columns(signals)
    if "date" not in signals.columns:
        raise ValueError("Primary signal file must contain a date column.")
    return signals.sort_values("date").reset_index(drop=True)


def melt_signals(signals: pd.DataFrame) -> pd.DataFrame:
    signals = signals.copy()
    signals["date"] = pd.to_datetime(signals["date"])
    signal_cols = [c for c in signals.columns if c != "date"]
    long = signals.melt(id_vars="date", value_vars=signal_cols, var_name="instrument", value_name="primary_signal")
    long["instrument"] = long["instrument"].str.lower()
    long["primary_signal"] = long["primary_signal"].astype(int)
    return long.sort_values(["instrument", "date"]).reset_index(drop=True)


def merge_price_and_signal_data(features: pd.DataFrame, signals_long: pd.DataFrame) -> pd.DataFrame:
    full = signals_long.merge(features, on=["date", "instrument"], how="left")
    if "asset_class" not in full.columns:
        full["asset_class"] = full["instrument"].map(ASSET_CLASS)
    return full.sort_values(["date", "instrument"]).reset_index(drop=True)


def load_and_merge_data(config: CourseworkConfig) -> dict[str, pd.DataFrame]:
    """Load the raw coursework inputs and return the canonical raw frames.

    The expensive feature engineering step is deliberately left to
    `features.build_feature_matrix` so the main runner reads as the official
    coursework pipeline: raw data first, then features, then labels.
    """

    ohlcv = load_ohlcv(config.ohlcv_path)
    signals = load_primary_signals(config.primary_signals_path)
    signals_long = melt_signals(signals)
    return {
        "ohlcv": ohlcv,
        "primary_signals": signals,
        "signals_long": signals_long,
    }
