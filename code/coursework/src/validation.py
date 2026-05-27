"""Input, panel, leakage, and prediction-file validation.

This module contains schema checks, duplicate checks, no-lookahead safeguards,
and final CSV validation for the coursework deliverable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CourseworkConfig, INSTRUMENTS


def validate_input_files(*paths: str | Path) -> None:
    if len(paths) == 1 and isinstance(paths[0], CourseworkConfig):
        config = paths[0]
        paths = (config.ohlcv_path, config.primary_signals_path)
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing input file(s): {missing}")


def validate_no_duplicates(frame: pd.DataFrame, keys: list[str] | tuple[str, ...] = ("date", "instrument")) -> None:
    duplicate_count = int(frame.duplicated(list(keys)).sum())
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate rows for keys {list(keys)}")


def validate_date_instrument_panel(frame: pd.DataFrame, required_instruments: list[str] | None = None) -> dict[str, object]:
    required = {"date", "instrument"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Panel is missing required columns: {sorted(missing)}")
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"])
    validate_no_duplicates(out)
    instruments = sorted(out["instrument"].dropna().astype(str).str.lower().unique().tolist())
    if required_instruments:
        missing_inst = sorted(set(required_instruments) - set(instruments))
        if missing_inst:
            raise ValueError(f"Panel is missing instruments: {missing_inst}")
    return {
        "rows": len(out),
        "start_date": out["date"].min(),
        "end_date": out["date"].max(),
        "n_dates": out["date"].nunique(),
        "n_instruments": len(instruments),
        "instruments": instruments,
    }


def validate_prediction_file(predictions: pd.DataFrame | str | Path, require_probability_bounds: bool = True) -> dict[str, object]:
    if isinstance(predictions, (str, Path)):
        predictions = pd.read_csv(predictions)
    expected = ["date", "instrument", "prediction"]
    if list(predictions.columns) != expected:
        raise ValueError(f"Prediction file must have exact columns {expected}; got {list(predictions.columns)}")
    out = predictions.copy()
    out["date"] = pd.to_datetime(out["date"])
    validate_no_duplicates(out)
    if out["prediction"].isna().any():
        raise ValueError("Prediction file contains missing probabilities.")
    if require_probability_bounds and not out["prediction"].between(0, 1).all():
        raise ValueError("Prediction probabilities must lie in [0, 1].")
    return validate_date_instrument_panel(out)


def validate_panel(raw_data: dict[str, pd.DataFrame] | pd.DataFrame, config: CourseworkConfig) -> dict[str, object]:
    """Validate the coursework date-instrument panel used by the pipeline."""

    if isinstance(raw_data, dict):
        panel = raw_data["signals_long"]
    else:
        panel = raw_data
    return validate_date_instrument_panel(panel, list(INSTRUMENTS))


def validate_no_lookahead_features(frame: pd.DataFrame, feature_cols: list[str]) -> dict[str, object]:
    bad_names = [c for c in feature_cols if c.startswith("future_") or c.endswith("_fwd") or "lead" in c]
    if bad_names:
        raise ValueError(f"Feature names suggest lookahead leakage: {bad_names}")
    same_day_ohlcv = [c for c in feature_cols if c in {"open", "high", "low", "close", "volume", "open_interest"}]
    if same_day_ohlcv:
        raise ValueError(f"Same-day raw OHLCV columns are not allowed as features: {same_day_ohlcv}")
    return {"n_features_checked": len(feature_cols), "status": "passed"}


def validate_barrier_end_dates(labels: pd.DataFrame, prediction_start: str | pd.Timestamp | None = None) -> dict[str, object]:
    if labels.empty:
        return {"rows": 0, "invalid_end_before_start": 0, "crossing_prediction_start": 0}
    out = labels.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["label_end_date"] = pd.to_datetime(out["label_end_date"])
    invalid = int((out["label_end_date"] <= out["date"]).sum())
    if invalid:
        raise ValueError(f"Found {invalid} label end dates on or before event date.")
    crossing = 0
    if prediction_start is not None:
        start = pd.Timestamp(prediction_start)
        crossing = int(((out["date"] < start) & (out["label_end_date"] >= start)).sum())
    return {
        "rows": len(out),
        "invalid_end_before_start": invalid,
        "crossing_prediction_start": crossing,
        "max_holding_days": float(out.get("holding_days", pd.Series(dtype=float)).max()) if "holding_days" in out else np.nan,
    }
