"""Triple-barrier meta-label construction.

This module creates binary meta-labels for non-zero primary-signal trade
opportunities and records diagnostics about barrier outcomes and sample counts.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .config import CourseworkConfig
from .features import add_controlled_features


def triple_barrier_labels(
    ohlcv: pd.DataFrame,
    signals_long: pd.DataFrame,
    feature_frame: pd.DataFrame,
    horizon: int,
    pt_mult: float,
    sl_mult: float,
) -> pd.DataFrame:
    price = ohlcv[["date", "instrument", "close"]].copy()
    price["date"] = pd.to_datetime(price["date"])
    price["instrument"] = price["instrument"].str.lower()
    vol_cols = ["date", "instrument", "ewma_vol_60", "rv_20"]
    vol_frame = feature_frame[vol_cols].copy()
    signal_price = signals_long.merge(price, on=["date", "instrument"], how="left").merge(
        vol_frame,
        on=["date", "instrument"],
        how="left",
    )

    labelled_rows: list[dict[str, Any]] = []
    for inst, inst_prices in price.groupby("instrument", sort=False):
        inst_prices = inst_prices.sort_values("date").reset_index(drop=True)
        date_to_pos = {d: i for i, d in enumerate(inst_prices["date"])}
        closes = inst_prices["close"].to_numpy()
        dates = inst_prices["date"].to_numpy()
        median_vol = (
            signal_price.loc[signal_price["instrument"] == inst, ["ewma_vol_60", "rv_20"]]
            .stack()
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .median()
        )
        if not np.isfinite(median_vol) or median_vol <= 0:
            median_vol = 0.01

        events = signal_price[(signal_price["instrument"] == inst) & (signal_price["primary_signal"] != 0)]
        for row in events.itertuples(index=False):
            date = row.date
            pos = date_to_pos.get(date)
            if pos is None or pos >= len(closes) - 1 or not np.isfinite(row.close):
                continue

            vol = row.ewma_vol_60
            if not np.isfinite(vol) or vol <= 1e-8:
                vol = row.rv_20
            if not np.isfinite(vol) or vol <= 1e-8:
                vol = median_vol
            vol = float(np.clip(vol, 0.0025, 0.12))
            pt = pt_mult * vol
            sl = sl_mult * vol
            side = int(row.primary_signal)
            entry = closes[pos]
            end_pos = min(pos + horizon, len(closes) - 1)
            if end_pos <= pos:
                continue

            future_positions = np.arange(pos + 1, end_pos + 1)
            side_returns = side * (closes[future_positions] / entry - 1)
            label = np.nan
            reason = "incomplete"
            exit_pos = end_pos
            exit_ret = np.nan

            pt_hits = np.where(side_returns >= pt)[0]
            sl_hits = np.where(side_returns <= -sl)[0]
            first_pt = pt_hits[0] if len(pt_hits) else math.inf
            first_sl = sl_hits[0] if len(sl_hits) else math.inf

            if first_pt < first_sl:
                exit_pos = future_positions[int(first_pt)]
                label = 1
                reason = "profit_taking"
                exit_ret = side_returns[int(first_pt)]
            elif first_sl < first_pt:
                exit_pos = future_positions[int(first_sl)]
                label = 0
                reason = "stop_loss"
                exit_ret = side_returns[int(first_sl)]
            elif len(future_positions) == horizon:
                exit_ret = side_returns[-1]
                label = int(exit_ret > 0)
                reason = "vertical_barrier"

            labelled_rows.append(
                {
                    "date": date,
                    "instrument": inst,
                    "label": label,
                    "label_end_date": pd.Timestamp(dates[exit_pos]),
                    "label_reason": reason,
                    "side_return": exit_ret,
                    "holding_days": int(exit_pos - pos),
                }
            )

    return pd.DataFrame(labelled_rows)

def compute_lagged_daily_volatility(ohlcv: pd.DataFrame, vol_lookback_days: int) -> pd.DataFrame:
    prices = ohlcv[["date", "instrument", "close"]].copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices["instrument"] = prices["instrument"].astype(str).str.lower()
    prices = prices.sort_values(["instrument", "date"])
    prices["daily_return"] = prices.groupby("instrument")["close"].pct_change()
    prices["rolling_vol"] = (
        prices.groupby("instrument")["daily_return"]
        .transform(lambda s: s.rolling(vol_lookback_days, min_periods=max(5, vol_lookback_days // 2)).std().shift(1))
    )
    prices["vol_lookback_days"] = vol_lookback_days
    return prices[["date", "instrument", "daily_return", "rolling_vol", "vol_lookback_days"]]


def _barrier_label_events(
    inst_prices: pd.DataFrame,
    inst_events: pd.DataFrame,
    vertical_barrier_days: int,
    profit_taking_multiplier: float,
    stop_loss_multiplier: float,
) -> list[dict[str, Any]]:
    inst_prices = inst_prices.sort_values("date").reset_index(drop=True)
    date_to_pos = {d: i for i, d in enumerate(inst_prices["date"])}
    closes = inst_prices["close"].to_numpy(dtype=float)
    dates = inst_prices["date"].to_numpy()
    rows: list[dict[str, Any]] = []
    for row in inst_events.sort_values("date").itertuples(index=False):
        pos = date_to_pos.get(row.date)
        if pos is None or pos >= len(closes) - 1 or not np.isfinite(getattr(row, "close", np.nan)):
            continue
        vol = float(getattr(row, "rolling_vol", np.nan))
        end_pos = min(pos + vertical_barrier_days, len(closes) - 1)
        crosses_end = int(pos + vertical_barrier_days > len(closes) - 1)
        base = {
            "date": row.date,
            "instrument": row.instrument,
            "missing_volatility": int(not np.isfinite(vol) or vol <= 0),
            "barrier_end_crosses_data": crosses_end,
        }
        if not np.isfinite(vol) or vol <= 0 or end_pos <= pos:
            rows.append({**base, "label": np.nan, "label_end_date": pd.NaT, "label_reason": "missing_volatility", "side_return": np.nan, "holding_days": np.nan})
            continue
        pt = profit_taking_multiplier * vol
        sl = stop_loss_multiplier * vol
        side = int(row.primary_signal)
        entry = closes[pos]
        future_positions = np.arange(pos + 1, end_pos + 1)
        side_returns = side * (closes[future_positions] / entry - 1)
        pt_hits = np.where(side_returns >= pt)[0]
        sl_hits = np.where(side_returns <= -sl)[0]
        first_pt = pt_hits[0] if len(pt_hits) else math.inf
        first_sl = sl_hits[0] if len(sl_hits) else math.inf
        label = np.nan
        reason = "incomplete"
        exit_pos = end_pos
        exit_ret = np.nan
        if first_pt < first_sl:
            exit_pos = future_positions[int(first_pt)]
            label = 1
            reason = "profit_taking"
            exit_ret = side_returns[int(first_pt)]
        elif first_sl < first_pt:
            exit_pos = future_positions[int(first_sl)]
            label = 0
            reason = "stop_loss"
            exit_ret = side_returns[int(first_sl)]
        elif len(future_positions) == vertical_barrier_days:
            exit_ret = side_returns[-1]
            label = int(exit_ret > 0)
            reason = "vertical_barrier"
        rows.append(
            {
                **base,
                "label": label,
                "label_end_date": pd.Timestamp(dates[exit_pos]),
                "label_reason": reason,
                "side_return": exit_ret,
                "holding_days": int(exit_pos - pos),
            }
        )
    return rows


def apply_triple_barrier_for_instrument(
    ohlcv: pd.DataFrame,
    signals_long: pd.DataFrame,
    instrument: str,
    vertical_barrier_days: int,
    profit_taking_multiplier: float,
    stop_loss_multiplier: float,
    vol_lookback_days: int,
) -> pd.DataFrame:
    price = ohlcv[["date", "instrument", "close"]].copy()
    price["date"] = pd.to_datetime(price["date"])
    price["instrument"] = price["instrument"].astype(str).str.lower()
    vol = compute_lagged_daily_volatility(price, vol_lookback_days)
    events = signals_long.copy()
    events["date"] = pd.to_datetime(events["date"])
    events["instrument"] = events["instrument"].astype(str).str.lower()
    events = events[(events["instrument"] == instrument) & (events["primary_signal"] != 0)]
    events = events.merge(price, on=["date", "instrument"], how="left").merge(
        vol[["date", "instrument", "rolling_vol"]], on=["date", "instrument"], how="left"
    )
    rows = _barrier_label_events(
        price[price["instrument"] == instrument],
        events,
        vertical_barrier_days,
        profit_taking_multiplier,
        stop_loss_multiplier,
    )
    return pd.DataFrame(rows)


def apply_triple_barrier_all_instruments(
    ohlcv: pd.DataFrame,
    signals_long: pd.DataFrame,
    vertical_barrier_days: int,
    profit_taking_multiplier: float,
    stop_loss_multiplier: float,
    vol_lookback_days: int,
) -> pd.DataFrame:
    frames = []
    for instrument in sorted(signals_long["instrument"].dropna().unique()):
        frames.append(
            apply_triple_barrier_for_instrument(
                ohlcv,
                signals_long,
                instrument,
                vertical_barrier_days,
                profit_taking_multiplier,
                stop_loss_multiplier,
                vol_lookback_days,
            )
        )
    if not frames:
        return pd.DataFrame()
    labels = pd.concat(frames, ignore_index=True)
    labels["vertical_barrier_days"] = vertical_barrier_days
    labels["profit_taking_multiplier"] = profit_taking_multiplier
    labels["stop_loss_multiplier"] = stop_loss_multiplier
    labels["vol_lookback_days"] = vol_lookback_days
    return labels.sort_values(["date", "instrument"]).reset_index(drop=True)


def create_meta_labels(
    ohlcv: pd.DataFrame,
    signals_long: pd.DataFrame,
    vertical_barrier_days: int = 10,
    profit_taking_multiplier: float = 1.5,
    stop_loss_multiplier: float = 1.5,
    vol_lookback_days: int = 60,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    labels = apply_triple_barrier_all_instruments(
        ohlcv,
        signals_long,
        vertical_barrier_days,
        profit_taking_multiplier,
        stop_loss_multiplier,
        vol_lookback_days,
    )
    return labels, label_diagnostics(labels, signals_long)


def handle_zero_primary_signals(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    zero = out["primary_signal"].fillna(0).eq(0)
    if "label" in out.columns:
        out.loc[zero, "label"] = np.nan
    return out


def label_diagnostics(labels: pd.DataFrame, signals_long: pd.DataFrame | None = None) -> dict[str, Any]:
    if labels.empty:
        return {
            "label_count": 0,
            "valid_label_count": 0,
            "positive_rate": np.nan,
            "stop_loss_hit_rate": np.nan,
            "profit_taking_hit_rate": np.nan,
            "timeout_rate": np.nan,
            "average_days_to_first_barrier": np.nan,
            "number_of_valid_samples": 0,
        }
    valid = labels[labels["label"].notna()].copy()
    nonzero_count = int(len(labels))
    if signals_long is not None and "primary_signal" in signals_long:
        nonzero_count = int((signals_long["primary_signal"] != 0).sum())
    counts = valid["label"].astype(int).value_counts().to_dict() if len(valid) else {}
    return {
        "non_zero_signal_samples": nonzero_count,
        "label_count": len(labels),
        "valid_label_count": len(valid),
        "label_counts": counts,
        "positive_rate": float(valid["label"].mean()) if len(valid) else np.nan,
        "stop_loss_hit_rate": float((valid["label_reason"] == "stop_loss").mean()) if len(valid) else np.nan,
        "profit_taking_hit_rate": float((valid["label_reason"] == "profit_taking").mean()) if len(valid) else np.nan,
        "timeout_rate": float((valid["label_reason"] == "vertical_barrier").mean()) if len(valid) else np.nan,
        "average_days_to_first_barrier": float(valid["holding_days"].mean()) if len(valid) else np.nan,
        "number_of_valid_samples": len(valid),
        "missing_volatility_count": int(labels.get("missing_volatility", pd.Series(0, index=labels.index)).sum()),
        "barrier_end_date_crossing_count": int(labels.get("barrier_end_crosses_data", pd.Series(0, index=labels.index)).sum()),
        "class_imbalance": float(abs(valid["label"].mean() - 0.5)) if len(valid) else np.nan,
    }


def create_triple_barrier_labels(
    feature_data: pd.DataFrame,
    config: CourseworkConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create triple-barrier meta-labels and attach label-dependent features.

    These labels define the meta-labeling task; the barrier specification is a
    modelling choice.
    """

    ohlcv = feature_data.attrs.get("ohlcv")
    signals_long = feature_data.attrs.get("signals_long")
    engineered_features = feature_data.attrs.get("engineered_features", feature_data)
    if ohlcv is None or signals_long is None:
        raise ValueError("feature_data must come from build_feature_matrix so raw OHLCV and signal metadata are available.")

    labels = triple_barrier_labels(
        ohlcv=ohlcv,
        signals_long=signals_long,
        feature_frame=engineered_features,
        horizon=config.vertical_barrier_days,
        pt_mult=config.profit_taking_multiplier,
        sl_mult=config.stop_loss_multiplier,
    )
    labeled_data = feature_data.merge(labels, on=["date", "instrument"], how="left")
    labeled_data = add_controlled_features(labeled_data)
    labeled_data.attrs.update(feature_data.attrs)
    return labeled_data, label_diagnostics(labels, signals_long)
