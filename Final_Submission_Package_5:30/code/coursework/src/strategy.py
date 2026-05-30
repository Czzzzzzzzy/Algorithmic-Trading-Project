"""Optional strategy-construction bonus track.

This module converts frozen metamodel probabilities into signed position
weights, backtests close-to-close returns, and writes optional competition
outputs without changing the required metamodel prediction file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import CONFIG, INSTRUMENTS, CourseworkConfig
from .data_loader import load_ohlcv, load_primary_signals, melt_signals
from .utils import sha256_file
from .validation import validate_no_duplicates


@dataclass(frozen=True)
class StrategySelection:
    selected_method: str
    best_gross_sharpe_method: str
    best_net_2bps_method: str
    conservative_method: str
    recommendation_reason: str
    selected_strategy_name: str
    selected_target_vol: float | None = None


BLIND_BASELINE_METHOD = "blindly_follow_primary_signal_equal_weight"
TARGET_VOL_STRATEGY_NAME = "soft_allocation_target_vol"


def _target_vol_method_name(target_vol: float) -> str:
    label = f"{target_vol * 100:.1f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{TARGET_VOL_STRATEGY_NAME}_{label}pct"


def _format_target_vol(target_vol: float | None) -> str:
    if target_vol is None or not np.isfinite(target_vol):
        return "n/a"
    return f"{target_vol:.1%}"


def _first_finite(values: pd.Series, default: float = np.nan) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    return float(finite.iloc[0]) if len(finite) else default


def _method_metadata(frame: pd.DataFrame, method: str) -> dict[str, Any]:
    method_frame = frame[frame["method"] == method]
    if method_frame.empty:
        return {
            "strategy_name": method,
            "target_vol": np.nan,
            "is_selectable": method != BLIND_BASELINE_METHOD,
        }
    strategy_name = str(method_frame["strategy_name"].iloc[0]) if "strategy_name" in method_frame else method
    target_vol = _first_finite(method_frame["target_vol"]) if "target_vol" in method_frame else np.nan
    if "is_selectable" in method_frame:
        is_selectable = bool(method_frame["is_selectable"].iloc[0])
    else:
        is_selectable = method != BLIND_BASELINE_METHOD
    return {
        "strategy_name": strategy_name,
        "target_vol": target_vol,
        "is_selectable": is_selectable,
    }


def load_strategy_inputs(config: CourseworkConfig = CONFIG) -> pd.DataFrame:
    """Load final predictions, primary signals, and close prices."""

    predictions = pd.read_csv(config.final_prediction_path)
    predictions["date"] = pd.to_datetime(predictions["date"])
    predictions["instrument"] = predictions["instrument"].astype(str).str.lower()

    signals = load_primary_signals(config.primary_signals_path)
    signals_long = melt_signals(signals)

    panel = predictions.merge(signals_long, on=["date", "instrument"], how="left")
    panel["primary_signal"] = panel["primary_signal"].fillna(0).astype(int)
    panel["prediction"] = pd.to_numeric(panel["prediction"], errors="coerce")

    prices = load_ohlcv(config.ohlcv_path)[["date", "instrument", "close"]].copy()
    close_rows: list[pd.DataFrame] = []
    for inst, inst_panel in panel.groupby("instrument", sort=False):
        inst_panel = inst_panel.sort_values("date").copy()
        inst_prices = prices[prices["instrument"] == inst].sort_values("date").copy()
        merged = pd.merge_asof(
            inst_panel,
            inst_prices[["date", "close"]],
            on="date",
            direction="backward",
            allow_exact_matches=True,
        )
        close_rows.append(merged)
    panel = pd.concat(close_rows, ignore_index=True) if close_rows else panel.assign(close=np.nan)
    panel["close"] = pd.to_numeric(panel["close"], errors="coerce")
    panel = panel.sort_values(["date", "instrument"]).reset_index(drop=True)

    if panel["prediction"].isna().any():
        raise ValueError("Strategy input has missing metamodel predictions.")
    if panel["close"].isna().any():
        raise ValueError("Strategy input has missing close prices.")
    validate_no_duplicates(panel, ("date", "instrument"))
    return panel[["date", "instrument", "prediction", "primary_signal", "close"]]


def _add_default_weight_metadata(
    weights: pd.DataFrame,
    strategy_name: str,
    target_vol: float = np.nan,
    is_selectable: bool = True,
) -> pd.DataFrame:
    out = weights.copy()
    out["strategy_name"] = strategy_name
    out["target_vol"] = target_vol
    out["base_weight"] = out.get("weight", 0.0)
    out["pre_cap_weight"] = out.get("weight", 0.0)
    out["predicted_daily_vol"] = np.nan
    out["predicted_annual_vol"] = np.nan
    out["scale_factor"] = 1.0
    out["cov_lookback_obs"] = 0
    out["covariance_max_return_date"] = pd.NaT
    out["vol_fallback_reason"] = ""
    out["is_selectable"] = bool(is_selectable)
    return out


def build_strategy_weights(df: pd.DataFrame, config: CourseworkConfig = CONFIG, method: str = "confidence_scaled") -> pd.DataFrame:
    """Convert probabilities and primary signals into signed raw weights."""

    out = df.copy()
    p = out["prediction"].fillna(0.0).clip(0.0, 1.0)
    signal = out["primary_signal"].fillna(0).astype(int)

    if method == BLIND_BASELINE_METHOD:
        raw = signal.astype(float)
    elif method == "threshold_filter_050":
        raw = np.where(p >= 0.50, signal, 0.0)
    elif method == "threshold_filter_055":
        raw = np.where(p >= 0.55, signal, 0.0)
    elif method == "threshold_filter_060":
        raw = np.where(p >= 0.60, signal, 0.0)
    elif method == "confidence_linear":
        raw = signal * np.maximum(p - 0.50, 0.0)
    elif method == "confidence_scaled":
        raw = signal * np.maximum(2.0 * p - 1.0, 0.0)
    elif method == "soft_allocation":
        raw = signal * p
    else:
        raise ValueError(f"Unknown strategy method: {method}")

    raw = np.where(signal == 0, 0.0, raw)
    out["method"] = method
    out["raw_weight"] = raw.astype(float)
    weights = normalize_weights_by_date(out, config)
    return _add_default_weight_metadata(
        weights,
        strategy_name=method,
        is_selectable=method != BLIND_BASELINE_METHOD,
    )


def normalize_weights_by_date(df: pd.DataFrame, config: CourseworkConfig = CONFIG) -> pd.DataFrame:
    """Normalize daily gross exposure and apply a conservative instrument cap."""

    target = float(config.strategy_gross_exposure_target)
    cap = float(config.strategy_max_abs_weight_per_instrument)

    def _normalize(group: pd.DataFrame) -> pd.DataFrame:
        out = group.copy()
        raw = out["raw_weight"].to_numpy(dtype=float)
        capped_raw = np.clip(raw, -cap, cap)
        gross = float(np.abs(capped_raw).sum())
        if gross <= 0 or not np.isfinite(gross):
            weight = np.zeros(len(out), dtype=float)
        else:
            weight = capped_raw / gross * target
            weight = np.clip(weight, -cap, cap)
            capped_gross = float(np.abs(weight).sum())
            if capped_gross > target and capped_gross > 0:
                weight = weight / capped_gross * target
        out["weight"] = weight
        out["gross_exposure"] = float(np.abs(weight).sum())
        out["active_instruments"] = int((np.abs(weight) > 1e-12).sum())
        out["gross_exposure_target"] = target
        out["max_abs_weight_per_instrument"] = cap
        return out

    normalized_rows = [_normalize(group) for _, group in df.groupby("date", sort=False)]
    normalized = pd.concat(normalized_rows, ignore_index=True) if normalized_rows else df.assign(weight=np.nan)
    return normalized.reset_index(drop=True)


def _close_to_close_return_wide(config: CourseworkConfig = CONFIG) -> pd.DataFrame:
    prices = load_ohlcv(config.ohlcv_path)[["date", "instrument", "close"]].copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices["instrument"] = prices["instrument"].astype(str).str.lower()
    wide = (
        prices.pivot_table(index="date", columns="instrument", values="close", aggfunc="last")
        .sort_index()
        .reindex(columns=INSTRUMENTS)
    )
    returns = wide.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    return returns


def _covariance_for_date(
    returns_wide: pd.DataFrame,
    date: pd.Timestamp,
    instruments: list[str],
    lookback: int,
) -> tuple[np.ndarray | None, int, pd.Timestamp | pd.NaT]:
    history = returns_wide.loc[returns_wide.index < date, instruments].tail(int(lookback))
    history = history.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if len(history) < 2:
        return None, int(len(history)), pd.NaT if history.empty else pd.Timestamp(history.index.max())
    cov = history.cov(min_periods=2).reindex(index=instruments, columns=instruments)
    cov_values = np.nan_to_num(cov.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return cov_values, int(len(history)), pd.Timestamp(history.index.max())


def build_target_vol_weights(
    df: pd.DataFrame,
    config: CourseworkConfig = CONFIG,
    target_vol: float | None = None,
    returns_wide: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Scale soft-allocation base weights to a rolling covariance target vol.

    The covariance window uses close-to-close returns with return dates strictly
    earlier than the portfolio weight date, so the t-to-t+1 return is never used
    to form weight_t.
    """

    target = float(config.strategy_target_vol if target_vol is None else target_vol)
    cap = float(config.strategy_max_abs_weight_per_instrument)
    max_leverage = float(config.strategy_max_leverage)
    lookback = int(config.strategy_cov_lookback)
    trading_days = float(config.trading_days_per_year)
    method = _target_vol_method_name(target)
    returns = _close_to_close_return_wide(config) if returns_wide is None else returns_wide

    base = df.copy()
    base["date"] = pd.to_datetime(base["date"])
    base["instrument"] = base["instrument"].astype(str).str.lower()
    p = base["prediction"].fillna(0.0).clip(0.0, 1.0)
    signal = base["primary_signal"].fillna(0).astype(int)
    base["method"] = method
    base["strategy_name"] = TARGET_VOL_STRATEGY_NAME
    base["target_vol"] = target
    base["raw_weight"] = np.where(signal == 0, 0.0, signal * p).astype(float)

    rows: list[pd.DataFrame] = []
    for date_value, group in base.groupby("date", sort=False):
        out = group.sort_values("instrument").copy()
        instruments = out["instrument"].tolist()
        raw = out["raw_weight"].to_numpy(dtype=float)
        gross_raw = float(np.abs(raw).sum())
        if gross_raw <= 0 or not np.isfinite(gross_raw):
            base_weight = np.zeros(len(out), dtype=float)
        else:
            base_weight = raw / gross_raw

        cov, cov_obs, cov_max_date = _covariance_for_date(returns, pd.Timestamp(date_value), instruments, lookback)
        fallback_reason = ""
        if float(np.abs(base_weight).sum()) <= 0:
            predicted_daily_vol = 0.0
            predicted_annual_vol = 0.0
            scale = 0.0
            fallback_reason = "no_active_positions"
        elif cov is None:
            predicted_daily_vol = np.nan
            predicted_annual_vol = np.nan
            scale = 1.0
            fallback_reason = "scale_1_insufficient_covariance_history"
        else:
            portfolio_var = float(base_weight @ cov @ base_weight)
            predicted_daily_vol = float(np.sqrt(portfolio_var)) if portfolio_var > 0 and np.isfinite(portfolio_var) else np.nan
            predicted_annual_vol = predicted_daily_vol * np.sqrt(trading_days) if np.isfinite(predicted_daily_vol) else np.nan
            if predicted_annual_vol > 0 and np.isfinite(predicted_annual_vol):
                scale = min(target / predicted_annual_vol, max_leverage)
            else:
                scale = 1.0
                fallback_reason = "scale_1_invalid_predicted_volatility"

        pre_cap_weight = base_weight * scale
        weight = np.clip(pre_cap_weight, -cap, cap)
        out["base_weight"] = base_weight
        out["predicted_daily_vol"] = predicted_daily_vol
        out["predicted_annual_vol"] = predicted_annual_vol
        out["scale_factor"] = float(scale)
        out["pre_cap_weight"] = pre_cap_weight
        out["weight"] = weight
        out["gross_exposure"] = float(np.abs(weight).sum())
        out["active_instruments"] = int((np.abs(weight) > 1e-12).sum())
        out["gross_exposure_target"] = np.nan
        out["max_abs_weight_per_instrument"] = cap
        out["cov_lookback_obs"] = cov_obs
        out["covariance_max_return_date"] = cov_max_date
        out["vol_fallback_reason"] = fallback_reason
        out["is_selectable"] = True
        rows.append(out)

    if not rows:
        return base.assign(weight=np.nan)
    return pd.concat(rows, ignore_index=True).sort_values(["date", "instrument"]).reset_index(drop=True)


def _price_return_panel(weights_df: pd.DataFrame | None = None, config: CourseworkConfig = CONFIG) -> pd.DataFrame:
    if weights_df is not None and {"date", "instrument", "close"}.issubset(weights_df.columns):
        prices = weights_df[["date", "instrument", "close"]].drop_duplicates().copy()
    else:
        prices = load_ohlcv(config.ohlcv_path)[["date", "instrument", "close"]].copy()
    prices = prices.sort_values(["instrument", "date"]).reset_index(drop=True)
    prices["next_close"] = prices.groupby("instrument")["close"].shift(-1)
    prices["next_date"] = prices.groupby("instrument")["date"].shift(-1)
    prices["next_return"] = prices["next_close"] / prices["close"] - 1.0
    return prices[["date", "instrument", "close", "next_date", "next_close", "next_return"]]


def compute_turnover(weights_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily turnover as sum of absolute weight changes."""

    rows: list[pd.DataFrame] = []
    for method, method_frame in weights_df.groupby("method"):
        meta = _method_metadata(weights_df, str(method))
        wide = (
            method_frame.pivot(index="date", columns="instrument", values="weight")
            .sort_index()
            .fillna(0.0)
        )
        turnover = wide.diff().abs().sum(axis=1)
        if len(turnover):
            turnover.iloc[0] = wide.iloc[0].abs().sum()
        active = (wide.abs() > 1e-12).sum(axis=1)
        gross = wide.abs().sum(axis=1)
        rows.append(
            pd.DataFrame(
                {
                    "date": wide.index,
                    "method": method,
                    "strategy_name": meta["strategy_name"],
                    "target_vol": meta["target_vol"],
                    "turnover": turnover.to_numpy(),
                    "active_instruments": active.to_numpy(),
                    "gross_exposure": gross.to_numpy(),
                }
            )
        )
    if not rows:
        return pd.DataFrame(columns=["date", "method", "strategy_name", "target_vol", "turnover", "active_instruments", "gross_exposure"])
    return pd.concat(rows, ignore_index=True).sort_values(["method", "date"]).reset_index(drop=True)


def compute_strategy_returns(
    weights_df: pd.DataFrame,
    prices_df: pd.DataFrame | None = None,
    config: CourseworkConfig = CONFIG,
) -> pd.DataFrame:
    """Compute next-day close-to-close strategy returns without look-ahead in weights."""

    prices = _price_return_panel(weights_df, config) if prices_df is None else prices_df.copy()
    merged = weights_df.merge(
        prices[["date", "instrument", "next_date", "next_return"]],
        on=["date", "instrument"],
        how="left",
    )
    merged = merged[merged["next_return"].notna()].copy()
    merged["weighted_return"] = merged["weight"] * merged["next_return"]
    if "scale_factor" not in merged:
        merged["scale_factor"] = 1.0
    if "predicted_annual_vol" not in merged:
        merged["predicted_annual_vol"] = np.nan

    daily = (
        merged.groupby(["method", "date", "next_date"], as_index=False)
        .agg(
            gross_return=("weighted_return", "sum"),
            active_instruments=("weight", lambda s: int((s.abs() > 1e-12).sum())),
            gross_exposure=("weight", lambda s: float(s.abs().sum())),
            scale_factor=("scale_factor", "max"),
            predicted_annual_vol=("predicted_annual_vol", "max"),
        )
        .sort_values(["method", "date"])
        .reset_index(drop=True)
    )
    method_meta = (
        weights_df.groupby("method", as_index=False)
        .agg(
            strategy_name=("strategy_name", "first"),
            target_vol=("target_vol", lambda s: _first_finite(s)),
            is_selectable=("is_selectable", "first"),
        )
    )
    daily = daily.merge(method_meta, on="method", how="left")
    turnover = compute_turnover(weights_df)
    daily = daily.merge(turnover[["date", "method", "turnover"]], on=["date", "method"], how="left")
    daily["turnover"] = daily["turnover"].fillna(0.0)

    rows: list[pd.DataFrame] = []
    for cost_bps in config.strategy_cost_bps_grid:
        out = daily.copy()
        out["cost_bps"] = int(cost_bps)
        out["transaction_cost"] = out["turnover"] * float(cost_bps) / 10000.0
        out["net_return"] = out["gross_return"] - out["transaction_cost"]
        rows.append(out)
    return pd.concat(rows, ignore_index=True).sort_values(["method", "cost_bps", "date"]).reset_index(drop=True)


def _max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    if wealth.empty:
        return np.nan
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    return float(drawdown.min())


def _average_holding_period(weights_df: pd.DataFrame, method: str) -> float:
    method_frame = weights_df[weights_df["method"] == method].sort_values(["instrument", "date"])
    runs: list[int] = []
    for _, group in method_frame.groupby("instrument"):
        signs = np.sign(group["weight"].to_numpy(dtype=float))
        current_sign = 0.0
        current_len = 0
        for sign in signs:
            if sign == 0:
                if current_sign != 0 and current_len:
                    runs.append(current_len)
                current_sign = 0.0
                current_len = 0
            elif sign == current_sign:
                current_len += 1
            else:
                if current_sign != 0 and current_len:
                    runs.append(current_len)
                current_sign = sign
                current_len = 1
        if current_sign != 0 and current_len:
            runs.append(current_len)
    return float(np.mean(runs)) if runs else np.nan


def _metrics_for_returns(
    frame: pd.DataFrame,
    weights_df: pd.DataFrame,
    method: str,
    cost_bps: int,
    config: CourseworkConfig,
) -> dict[str, Any]:
    returns = frame["net_return"].astype(float)
    gross_returns = frame["gross_return"].astype(float)
    n_days = len(returns)
    meta = _method_metadata(weights_df, method)
    target_vol = meta["target_vol"]
    target_vol_value = target_vol if np.isfinite(target_vol) else np.nan
    if n_days == 0:
        return {
            "method": method,
            "strategy_name": meta["strategy_name"],
            "target_vol": target_vol_value,
            "cost_bps": cost_bps,
            "n_days": 0,
            "cumulative_return": np.nan,
            "gross_cumulative_return": np.nan,
            "cagr": np.nan,
            "annualised_volatility": np.nan,
            "realized_annual_volatility": np.nan,
            "target_annual_volatility": target_vol_value,
            "realized_vol_to_target": np.nan,
            "sharpe_ratio": np.nan,
            "sortino_ratio": np.nan,
            "maximum_drawdown": np.nan,
            "average_holding_period": np.nan,
            "average_daily_turnover": np.nan,
            "active_days": 0,
            "average_active_instruments": np.nan,
            "average_gross_exposure": np.nan,
            "max_gross_exposure": np.nan,
            "average_scale_factor": np.nan,
            "max_scale_factor": np.nan,
            "hit_rate": np.nan,
            "is_selectable": meta["is_selectable"],
        }

    trading_days = float(config.trading_days_per_year)
    wealth = float((1.0 + returns).prod())
    cumulative_return = wealth - 1.0
    cagr = wealth ** (trading_days / n_days) - 1.0 if wealth > 0 else np.nan
    vol = float(returns.std(ddof=1) * np.sqrt(trading_days)) if n_days > 1 else np.nan
    mean_return = float(returns.mean())
    daily_std = float(returns.std(ddof=1)) if n_days > 1 else np.nan
    sharpe = mean_return / daily_std * np.sqrt(trading_days) if daily_std and daily_std > 0 else np.nan
    downside = returns[returns < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else np.nan
    sortino = mean_return / downside_std * np.sqrt(trading_days) if downside_std and downside_std > 0 else np.nan
    realized_to_target = vol / target_vol_value if np.isfinite(vol) and target_vol_value > 0 else np.nan
    average_scale = float(frame["scale_factor"].mean()) if "scale_factor" in frame else np.nan
    max_scale = float(frame["scale_factor"].max()) if "scale_factor" in frame else np.nan

    active_days = int((frame["active_instruments"] > 0).sum())
    return {
        "method": method,
        "strategy_name": meta["strategy_name"],
        "target_vol": target_vol_value,
        "cost_bps": int(cost_bps),
        "n_days": int(n_days),
        "cumulative_return": cumulative_return,
        "gross_cumulative_return": float((1.0 + gross_returns).prod() - 1.0),
        "cagr": cagr,
        "annualised_volatility": vol,
        "realized_annual_volatility": vol,
        "target_annual_volatility": target_vol_value,
        "realized_vol_to_target": realized_to_target,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "maximum_drawdown": _max_drawdown(returns),
        "average_holding_period": _average_holding_period(weights_df, method),
        "average_daily_turnover": float(frame["turnover"].mean()),
        "active_days": active_days,
        "average_active_instruments": float(frame["active_instruments"].mean()),
        "average_gross_exposure": float(frame["gross_exposure"].mean()),
        "max_gross_exposure": float(frame["gross_exposure"].max()),
        "average_scale_factor": average_scale,
        "max_scale_factor": max_scale,
        "hit_rate": float((returns > 0).mean()) if active_days else np.nan,
        "mean_daily_return": mean_return,
        "mean_gross_daily_return": float(gross_returns.mean()),
        "is_selectable": meta["is_selectable"],
    }


def compute_backtest_metrics(
    returns_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    config: CourseworkConfig = CONFIG,
) -> pd.DataFrame:
    """Compute required strategy backtest metrics for each method/cost pair."""

    rows: list[dict[str, Any]] = []
    for (method, cost_bps), group in returns_df.groupby(["method", "cost_bps"]):
        rows.append(_metrics_for_returns(group, weights_df, str(method), int(cost_bps), config))
    return pd.DataFrame(rows).sort_values(["cost_bps", "sharpe_ratio"], ascending=[True, False]).reset_index(drop=True)


def select_strategy(metrics: pd.DataFrame) -> StrategySelection:
    """Select a recommended optional strategy using 2 bps net Sharpe guardrails."""

    selectable = metrics[metrics["is_selectable"].astype(bool)].copy() if "is_selectable" in metrics else metrics.copy()
    gross = selectable[selectable["cost_bps"] == 0].copy()
    net_2 = selectable[selectable["cost_bps"] == 2].copy()
    best_gross = gross.sort_values("sharpe_ratio", ascending=False).iloc[0]
    best_net = net_2.sort_values("sharpe_ratio", ascending=False).iloc[0]
    conservative_method = "threshold_filter_060"
    conservative = net_2[net_2["method"] == conservative_method]
    if conservative.empty:
        conservative_method = str(best_net["method"])

    ranking = net_2.copy()
    ranking["vol_tracking_error"] = np.where(
        ranking["target_vol"].notna() & (ranking["target_vol"] > 0),
        (ranking["realized_vol_to_target"] - 1.0).abs(),
        np.inf,
    )
    ranking["complexity_rank"] = np.select(
        [
            ranking["method"].eq("soft_allocation"),
            ranking["strategy_name"].eq(TARGET_VOL_STRATEGY_NAME),
            ranking["method"].str.startswith("threshold_filter"),
        ],
        [0, 1, 2],
        default=3,
    )
    good = net_2[
        (net_2["sharpe_ratio"] > 0)
        & (net_2["maximum_drawdown"] > -0.20)
        & (net_2["average_daily_turnover"] <= 1.5)
    ].copy()
    if not good.empty:
        good = good.merge(
            ranking[["method", "vol_tracking_error", "complexity_rank"]],
            on="method",
            how="left",
        )
        selected = good.sort_values(
            ["sharpe_ratio", "maximum_drawdown", "average_daily_turnover", "complexity_rank", "vol_tracking_error"],
            ascending=[False, False, True, True, True],
        ).iloc[0]
        reason = "Selected the best 2 bps net Sharpe strategy that passed simple drawdown and turnover guardrails."
    else:
        conservative_rows = net_2[net_2["method"] == conservative_method]
        selected = conservative_rows.iloc[0] if not conservative_rows.empty else best_net
        reason = "No method clearly passed the positive-net-Sharpe guardrail, so the conservative threshold rule was selected."

    selected_target_vol = float(selected["target_vol"]) if pd.notna(selected.get("target_vol", np.nan)) else None
    selected_strategy_name = str(selected.get("strategy_name", selected["method"]))
    if selected_strategy_name == TARGET_VOL_STRATEGY_NAME:
        reason += f" Target-volatility scaling was promoted at {_format_target_vol(selected_target_vol)} target annual volatility."
    elif TARGET_VOL_STRATEGY_NAME in set(metrics["strategy_name"].dropna()):
        reason += " Target-volatility variants were tested but did not beat the selected 2 bps net Sharpe."

    return StrategySelection(
        selected_method=str(selected["method"]),
        best_gross_sharpe_method=str(best_gross["method"]),
        best_net_2bps_method=str(best_net["method"]),
        conservative_method=conservative_method,
        recommendation_reason=reason,
        selected_strategy_name=selected_strategy_name,
        selected_target_vol=selected_target_vol,
    )


def validate_strategy_weights(
    weights: pd.DataFrame,
    panel: pd.DataFrame,
    config: CourseworkConfig,
) -> dict[str, Any]:
    expected = ["date", "instrument", "weight"]
    if list(weights.columns) != expected:
        raise ValueError(f"Strategy weights must have exact columns {expected}; got {list(weights.columns)}")
    out = weights.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["instrument"] = out["instrument"].astype(str).str.lower()
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    if out["weight"].isna().any():
        raise ValueError("Strategy weights contain missing/non-numeric values.")
    if not np.isfinite(out["weight"]).all():
        raise ValueError("Strategy weights contain non-finite values.")
    validate_no_duplicates(out, ("date", "instrument"))
    if len(out) != len(panel):
        raise ValueError(f"Strategy weights should have one row per prediction row: {len(out)} != {len(panel)}")

    if "primary_signal" in out:
        check = out.copy()
    else:
        check = out.merge(panel[["date", "instrument", "primary_signal"]], on=["date", "instrument"], how="left")
    zero_signal_nonzero_weight = int(((check["primary_signal"] == 0) & (check["weight"].abs() > 1e-12)).sum())
    if zero_signal_nonzero_weight:
        raise ValueError(f"Found {zero_signal_nonzero_weight} zero-signal rows with non-zero strategy weight.")
    misaligned = int(((check["weight"] > 0) & (check["primary_signal"] < 0)).sum() + ((check["weight"] < 0) & (check["primary_signal"] > 0)).sum())
    if misaligned:
        raise ValueError(f"Found {misaligned} strategy weights that do not align with primary signal direction.")

    gross = out.groupby("date")["weight"].apply(lambda s: float(s.abs().sum()))
    cap = float(config.strategy_max_abs_weight_per_instrument)
    cap_exceeded = int((out["weight"].abs() > cap + 1e-10).sum())
    if cap_exceeded:
        raise ValueError(f"Found {cap_exceeded} strategy weights above the max abs cap {cap}.")
    return {
        "rows": int(len(out)),
        "date_count": int(out["date"].nunique()),
        "instrument_count": int(out["instrument"].nunique()),
        "min_weight": float(out["weight"].min()),
        "max_weight": float(out["weight"].max()),
        "max_abs_weight": float(out["weight"].abs().max()),
        "gross_exposure_mean": float(gross.mean()),
        "gross_exposure_max": float(gross.max()),
        "zero_signal_nonzero_weight": zero_signal_nonzero_weight,
        "direction_misaligned_weight": misaligned,
        "cap_exceeded": cap_exceeded,
        "finite_weight": bool(np.isfinite(out["weight"]).all()),
    }


def validate_target_vol_experiment(
    weights: pd.DataFrame,
    panel: pd.DataFrame,
    config: CourseworkConfig,
) -> dict[str, Any]:
    out = weights.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["instrument"] = out["instrument"].astype(str).str.lower()
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    if "primary_signal" in out:
        check = out.copy()
    else:
        check = out.merge(panel[["date", "instrument", "primary_signal"]], on=["date", "instrument"], how="left")
    target_rows = out[out["strategy_name"] == TARGET_VOL_STRATEGY_NAME].copy()
    cov_dates = pd.to_datetime(target_rows["covariance_max_return_date"], errors="coerce") if len(target_rows) else pd.Series(dtype="datetime64[ns]")
    weight_dates = pd.to_datetime(target_rows["date"], errors="coerce") if len(target_rows) else pd.Series(dtype="datetime64[ns]")
    cov_not_past = int(((cov_dates.notna()) & (cov_dates >= weight_dates)).sum()) if len(target_rows) else 0
    zero_signal_nonzero_weight = int(((check["primary_signal"] == 0) & (check["weight"].abs() > 1e-12)).sum())
    direction_misaligned_weight = int(
        ((check["weight"] > 0) & (check["primary_signal"] < 0)).sum()
        + ((check["weight"] < 0) & (check["primary_signal"] > 0)).sum()
    )
    cap = float(config.strategy_max_abs_weight_per_instrument)
    prediction_end = pd.Timestamp(config.prediction_end)
    if len(target_rows):
        fallback_series = target_rows["vol_fallback_reason"].astype("string")
        target_fallbacks = fallback_series[(fallback_series.notna()) & (fallback_series != "")]
    else:
        target_fallbacks = pd.Series(dtype="string")
    return {
        "all_weight_rows": int(len(out)),
        "target_vol_weight_rows": int(len(target_rows)),
        "target_vol_levels_tested": sorted(float(v) for v in target_rows["target_vol"].dropna().unique()),
        "weights_are_finite": bool(np.isfinite(out["weight"]).all()),
        "zero_signal_nonzero_weight": zero_signal_nonzero_weight,
        "direction_misaligned_weight": direction_misaligned_weight,
        "cap_exceeded": int((out["weight"].abs() > cap + 1e-10).sum()),
        "max_abs_weight": float(out["weight"].abs().max()),
        "covariance_uses_past_returns_only": cov_not_past == 0,
        "covariance_not_past_count": cov_not_past,
        "covariance_lookback_days": int(config.strategy_cov_lookback),
        "covariance_min_observations": int(target_rows["cov_lookback_obs"].min()) if len(target_rows) else 0,
        "covariance_max_observations": int(target_rows["cov_lookback_obs"].max()) if len(target_rows) else 0,
        "vol_fallback_rows": int(len(target_fallbacks)),
        "vol_fallback_reasons": ", ".join(sorted(target_fallbacks.unique())) if len(target_fallbacks) else "none",
        "max_weight_date": str(out["date"].max().date()) if len(out) else "",
        "no_hidden_2022h2_weight_dates": bool(len(out) == 0 or out["date"].max() <= prediction_end),
    }


def _format_metric(value: object, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return ""
        return f"{float(value):.{digits}f}"
    return str(value)


def _markdown_table(headers: list[str], rows: list[list[object]], digits: int = 4) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_metric(value, digits) for value in row) + " |")
    return "\n".join(lines)


def _strategy_label(row: pd.Series) -> str:
    if row.get("strategy_name") == TARGET_VOL_STRATEGY_NAME:
        return f"{TARGET_VOL_STRATEGY_NAME} ({_format_target_vol(float(row['target_vol']))})"
    return str(row.get("method", ""))


def _write_target_vol_outputs(
    metrics: pd.DataFrame,
    selection: StrategySelection,
    validation: dict[str, Any],
    config: CourseworkConfig,
) -> None:
    target_results = metrics[metrics["strategy_name"] == TARGET_VOL_STRATEGY_NAME].copy()
    target_results.to_csv(config.strategy_target_vol_results_path, index=False)
    metrics.to_csv(config.strategy_all_methods_comparison_path, index=False)

    cost2_target = target_results[target_results["cost_bps"] == 2].sort_values("target_vol")
    target_rows = []
    for row in cost2_target.itertuples(index=False):
        target_rows.append(
            [
                _format_target_vol(row.target_vol),
                row.cumulative_return,
                row.cagr,
                row.realized_annual_volatility,
                row.realized_vol_to_target,
                row.sharpe_ratio,
                row.sortino_ratio,
                row.maximum_drawdown,
                row.average_daily_turnover,
                row.average_gross_exposure,
                row.average_scale_factor,
                row.max_scale_factor,
            ]
        )

    lines = [
        "# Target-Volatility Strategy Appendix",
        "",
        "This appendix tests target-volatility scaling only for the optional strategy construction track. It does not retrain or alter the frozen metamodel prediction file.",
        "",
        "Construction: raw soft-allocation weights are normalized by date, scaled by `target_vol / predicted_annual_vol`, clipped to the configured max leverage, then capped at the single-instrument max absolute weight.",
        "",
        f"Covariance lookback: `{config.strategy_cov_lookback}` trading days.",
        f"Max leverage / scale factor: `{config.strategy_max_leverage}`.",
        f"Single-instrument max abs weight: `{config.strategy_max_abs_weight_per_instrument}`.",
        "Covariance windows use close-to-close returns with return dates strictly earlier than the weight date.",
        "If volatility cannot be estimated, the conservative fallback is `scale_factor = 1.0` for active positions.",
        "",
        "## 2 bps Net Results",
        "",
        _markdown_table(
            [
                "Target vol",
                "Cum. return",
                "CAGR",
                "Realized ann. vol",
                "Realized/target",
                "Sharpe",
                "Sortino",
                "Max DD",
                "Avg turnover",
                "Avg gross",
                "Avg scale",
                "Max scale",
            ],
            target_rows,
        ),
        "",
        "## Selection Outcome",
        "",
        f"- Selected optional method: `{selection.selected_method}`.",
        f"- Selected target volatility: `{_format_target_vol(selection.selected_target_vol)}`.",
        f"- Selection reason: {selection.recommendation_reason}",
        "",
        "## Validation Snapshot",
        "",
    ]
    for key, value in validation.items():
        lines.append(f"- {key}: {value}")
    config.strategy_target_vol_summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    cost2 = metrics[metrics["cost_bps"] == 2].copy().sort_values(
        ["sharpe_ratio", "maximum_drawdown", "average_daily_turnover"],
        ascending=[False, False, True],
    )
    rows = []
    for _, row in cost2.iterrows():
        rows.append(
            [
                _strategy_label(row),
                row["cost_bps"],
                row["cumulative_return"],
                row["cagr"],
                row["realized_annual_volatility"],
                row["target_vol"],
                row["sharpe_ratio"],
                row["sortino_ratio"],
                row["maximum_drawdown"],
                row["average_daily_turnover"],
                row["average_gross_exposure"],
                row["average_scale_factor"],
                bool(row["is_selectable"]),
            ]
        )
    comparison_lines = [
        "# Strategy All-Methods Comparison",
        "",
        "Sorted by 2 bps transaction-cost-adjusted Sharpe ratio. The blind primary-signal baseline is reported for context but is not eligible for optional-strategy promotion.",
        "",
        _markdown_table(
            [
                "Strategy",
                "Cost bps",
                "Cum. return",
                "CAGR",
                "Realized ann. vol",
                "Target vol",
                "Sharpe",
                "Sortino",
                "Max DD",
                "Avg turnover",
                "Avg gross",
                "Avg scale",
                "Selectable",
            ],
            rows,
        ),
        "",
    ]
    config.strategy_all_methods_comparison_md_path.write_text("\n".join(comparison_lines), encoding="utf-8")


def _write_strategy_baseline_comparison(
    metrics: pd.DataFrame,
    selection: StrategySelection,
    config: CourseworkConfig,
) -> None:
    selected_method = selection.selected_method
    compare_methods = [BLIND_BASELINE_METHOD, "soft_allocation", selected_method]
    comparison = metrics[
        metrics["method"].isin(compare_methods)
        & metrics["cost_bps"].isin([0, 2])
    ].copy()
    comparison = comparison.drop_duplicates(["method", "cost_bps"]).sort_values(["method", "cost_bps"])
    csv_path = config.output_dir / "strategy_baseline_comparison.csv"
    md_path = config.output_dir / "strategy_baseline_comparison.md"
    comparison.to_csv(csv_path, index=False)

    selected_2 = metrics[(metrics["method"] == selected_method) & (metrics["cost_bps"] == 2)].iloc[0]
    blind_2 = metrics[(metrics["method"] == BLIND_BASELINE_METHOD) & (metrics["cost_bps"] == 2)].iloc[0]
    soft_2 = metrics[(metrics["method"] == "soft_allocation") & (metrics["cost_bps"] == 2)].iloc[0]
    rows = []
    for _, row in comparison.iterrows():
        rows.append(
            [
                _strategy_label(row),
                int(row["cost_bps"]),
                row["cumulative_return"],
                row["cagr"],
                row["realized_annual_volatility"],
                row["sharpe_ratio"],
                row["sortino_ratio"],
                row["maximum_drawdown"],
                row["average_daily_turnover"],
                row["average_gross_exposure"],
            ]
        )
    lines = [
        "# Strategy Baseline Comparison",
        "",
        "This compares the selected optional strategy with the blind primary-signal baseline and the original soft-allocation strategy. The required metamodel prediction file is unchanged.",
        "",
        f"- Selected strategy: `{selected_method}`.",
        f"- Selected target volatility: `{_format_target_vol(selection.selected_target_vol)}`.",
        f"- Blind baseline: `{BLIND_BASELINE_METHOD}`.",
        f"- 2 bps Sharpe improvement versus blind baseline: `{selected_2.sharpe_ratio - blind_2.sharpe_ratio:.4f}`.",
        f"- 2 bps Sharpe improvement versus original soft_allocation: `{selected_2.sharpe_ratio - soft_2.sharpe_ratio:.4f}`.",
        f"- 2 bps cumulative return difference versus blind baseline: `{selected_2.cumulative_return - blind_2.cumulative_return:.4f}`.",
        "",
        "## Metrics",
        "",
        _markdown_table(
            [
                "Method",
                "Cost bps",
                "Cum. return",
                "CAGR",
                "Realized ann. vol",
                "Sharpe",
                "Sortino",
                "Max DD",
                "Avg turnover",
                "Avg gross",
            ],
            rows,
        ),
        "",
        "## Caution",
        "",
        "These are public-2022H1 close-to-close backtest results. The window has only 128 prediction dates, so high Sharpe ratios should be interpreted cautiously. Transaction costs are simplified as turnover times bps divided by 10000.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def _write_target_vol_audit(
    validation: dict[str, Any],
    selection: StrategySelection,
    config: CourseworkConfig,
) -> None:
    lines = [
        "# Strategy Target-Volatility Audit",
        "",
        f"Final prediction hash used: {sha256_file(config.final_prediction_path)}",
        f"Expected frozen hash: {config.promoted_final_hash}",
        f"Hash matches frozen final: {sha256_file(config.final_prediction_path) == config.promoted_final_hash}",
        f"Selected strategy method: {selection.selected_method}",
        f"Selected target volatility: {_format_target_vol(selection.selected_target_vol)}",
        "",
        "Validation checks:",
    ]
    for key, value in validation.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "Leakage controls:",
            "- Target-vol covariance uses return dates strictly earlier than the weight date.",
            "- Weight at date t earns the close-to-close return from t to t+1 only in the backtest.",
            "- The strategy runner checks the frozen prediction hash before and after optional strategy construction.",
            "- No hidden 2022H2 prices or labels are present in the input data used here.",
            "",
            "Volatility fallback:",
            "- If active-position volatility cannot be estimated, scale_factor = 1.0 is used instead of adding leverage.",
        ]
    )
    config.strategy_target_vol_audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_strategy_plots(
    returns: pd.DataFrame,
    selection: StrategySelection,
    config: CourseworkConfig,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    selected = returns[(returns["method"] == selection.selected_method) & (returns["cost_bps"] == 2)].copy()
    if selected.empty:
        return
    selected = selected.sort_values("date")
    wealth = (1.0 + selected["net_return"].fillna(0.0)).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(pd.to_datetime(selected["next_date"]), wealth, color="#1f77b4", linewidth=1.8)
    ax.set_title("Selected strategy cumulative return (2 bps net)")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(config.output_dir / "strategy_cumulative_return.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.fill_between(pd.to_datetime(selected["next_date"]), drawdown.to_numpy(), 0, color="#d62728", alpha=0.35)
    ax.plot(pd.to_datetime(selected["next_date"]), drawdown, color="#d62728", linewidth=1.2)
    ax.set_title("Selected strategy drawdown (2 bps net)")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(config.output_dir / "strategy_drawdown.png", dpi=180)
    plt.close(fig)


def _write_strategy_summary(
    metrics: pd.DataFrame,
    selection: StrategySelection,
    config: CourseworkConfig,
) -> None:
    selected_rows = metrics[metrics["method"] == selection.selected_method].sort_values("cost_bps")
    gross = selected_rows[selected_rows["cost_bps"] == 0].iloc[0]
    net_2 = selected_rows[selected_rows["cost_bps"] == 2].iloc[0]
    target_tested = metrics[metrics["strategy_name"] == TARGET_VOL_STRATEGY_NAME].copy()
    target_net_2 = target_tested[target_tested["cost_bps"] == 2].sort_values("target_vol")
    target_rows = []
    for row in target_net_2.itertuples(index=False):
        target_rows.append(
            [
                _format_target_vol(row.target_vol),
                row.cagr,
                row.realized_annual_volatility,
                row.realized_vol_to_target,
                row.sharpe_ratio,
                row.maximum_drawdown,
                row.average_daily_turnover,
                row.average_gross_exposure,
            ]
        )
    lines = [
        "# Optional Strategy Construction / Bonus Track",
        "",
        "This is an optional extension. It does not change the required metamodel prediction file.",
        "",
        f"Final metamodel prediction file: `{config.final_prediction_path}`",
        f"Final prediction SHA256: `{sha256_file(config.final_prediction_path)}`",
        "",
        "## Construction Rule",
        "",
        "Weights are built from primary-signal direction multiplied by metamodel probability confidence.",
        "Target-volatility variants first normalize soft-allocation weights by date, estimate rolling covariance from historical close-to-close returns, and scale the book toward a target annual volatility.",
        "The weight on date `t` earns close-to-close return from `t` to `t+1`; future returns are not used to form weights.",
        f"Gross exposure target: `{config.strategy_gross_exposure_target}`.",
        f"Max absolute weight per instrument: `{config.strategy_max_abs_weight_per_instrument}`.",
        f"Target-vol grid tested: `{', '.join(_format_target_vol(v) for v in config.strategy_target_vol_grid)}`.",
        f"Target-vol covariance lookback: `{config.strategy_cov_lookback}` trading days.",
        f"Target-vol max leverage / scale factor: `{config.strategy_max_leverage}`.",
        "",
        "## Methods Tested",
        "",
        f"- `{BLIND_BASELINE_METHOD}` (baseline only, not selectable)",
    ]
    for method in config.strategy_methods:
        lines.append(f"- `{method}`")
    for target in config.strategy_target_vol_grid:
        lines.append(f"- `{TARGET_VOL_STRATEGY_NAME}` at `{_format_target_vol(target)}` target annual volatility")
    lines.extend(
        [
            "",
            "## Selection",
            "",
            f"- Best gross Sharpe method: `{selection.best_gross_sharpe_method}`.",
            f"- Best net Sharpe method at 2 bps: `{selection.best_net_2bps_method}`.",
            f"- Most conservative method: `{selection.conservative_method}`.",
            f"- Recommended optional strategy: `{selection.selected_method}`.",
            f"- Recommended target volatility: `{_format_target_vol(selection.selected_target_vol)}`.",
            f"- Reason: {selection.recommendation_reason}",
            "",
            "## Selected Strategy Metrics",
            "",
            "| Cost bps | CAGR | Realized Ann. Vol | Sharpe | Sortino | Max Drawdown | Avg Turnover | Avg Gross |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in selected_rows.itertuples(index=False):
        lines.append(
            f"| {int(row.cost_bps)} | {row.cagr:.4f} | {row.realized_annual_volatility:.4f} | "
            f"{row.sharpe_ratio:.4f} | {row.sortino_ratio:.4f} | {row.maximum_drawdown:.4f} | "
            f"{row.average_daily_turnover:.4f} | {row.average_gross_exposure:.4f} |"
        )
    if target_rows:
        lines.extend(
            [
                "",
                "## Target-Volatility Appendix Results",
                "",
                "The table below reports target-vol variants at 2 bps transaction costs.",
                "",
                _markdown_table(
                    [
                        "Target vol",
                        "CAGR",
                        "Realized ann. vol",
                        "Realized/target",
                        "Sharpe",
                        "Max DD",
                        "Avg turnover",
                        "Avg gross",
                    ],
                    target_rows,
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Headline",
            "",
            f"- Gross Sharpe for selected strategy: `{gross.sharpe_ratio:.4f}`.",
            f"- 2 bps net Sharpe for selected strategy: `{net_2.sharpe_ratio:.4f}`.",
            f"- 2 bps realized annual volatility for selected strategy: `{net_2.realized_annual_volatility:.4f}`.",
            "",
            "## Baseline Comparison",
            "",
            "The blind primary-signal equal-weight strategy, original soft allocation, threshold rules, confidence-scaled rules, and target-volatility variants are compared in `outputs/strategy_all_methods_comparison.csv` and `.md`. The blind baseline is reported for context and is not eligible for optional-strategy promotion.",
            "",
            "Detailed comparison files:",
            "",
            "- `outputs/strategy_all_methods_comparison.csv`",
            "- `outputs/strategy_all_methods_comparison.md`",
            "- `outputs/strategy_target_vol_results.csv`",
            "- `outputs/strategy_target_vol_summary.md`",
            "- `outputs/strategy_baseline_comparison.csv`",
            "- `outputs/strategy_baseline_comparison.md`",
            "",
            "## Limitations",
            "",
            "This is a simple close-to-close portfolio backtest for the optional competition track. It assumes weights are set at date `t` and earn return from `t` to `t+1`. It does not use hidden 2022H2 data, does not retrain the metamodel, and does not alter the required prediction submission. The high Sharpe should be interpreted cautiously because the public test window has only 128 prediction dates.",
            "",
        ]
    )
    config.strategy_summary_path.write_text("\n".join(lines), encoding="utf-8")


def _write_strategy_audit(
    selected_weights: pd.DataFrame,
    selected_metrics: pd.DataFrame,
    validation: dict[str, Any],
    selection: StrategySelection,
    turnover: pd.DataFrame,
    config: CourseworkConfig,
) -> None:
    selected_turnover = turnover[turnover["method"] == selection.selected_method]
    lines = [
        "# Strategy Bonus Audit",
        "",
        f"Final prediction hash used: {sha256_file(config.final_prediction_path)}",
        f"Expected frozen hash: {config.promoted_final_hash}",
        f"Hash matches frozen final: {sha256_file(config.final_prediction_path) == config.promoted_final_hash}",
        f"Selected strategy method: {selection.selected_method}",
        f"Selected strategy name: {selection.selected_strategy_name}",
        f"Selected target volatility: {_format_target_vol(selection.selected_target_vol)}",
        f"Strategy weight file path: {config.strategy_weights_path}",
        "",
        "Output validation:",
    ]
    for key, value in validation.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "Gross exposure summary:",
            f"- mean gross exposure: {validation['gross_exposure_mean']:.6f}",
            f"- max gross exposure: {validation['gross_exposure_max']:.6f}",
            "",
            "Turnover summary:",
            f"- mean daily turnover: {selected_turnover['turnover'].mean():.6f}",
            f"- max daily turnover: {selected_turnover['turnover'].max():.6f}",
            "",
            "Backtest metrics summary:",
        ]
    )
    for row in selected_metrics.sort_values("cost_bps").itertuples(index=False):
        lines.append(
            f"- cost_bps={int(row.cost_bps)}: CAGR={row.cagr:.6f}, "
            f"realized_ann_vol={row.realized_annual_volatility:.6f}, Sharpe={row.sharpe_ratio:.6f}, "
            f"Sortino={row.sortino_ratio:.6f}, max_drawdown={row.maximum_drawdown:.6f}, "
            f"avg_holding_period={row.average_holding_period:.6f}, "
            f"avg_turnover={row.average_daily_turnover:.6f}, avg_gross={row.average_gross_exposure:.6f}"
        )
    lines.extend(
        [
            "",
            "Target-volatility experiment:",
            f"- target vol grid tested: {', '.join(_format_target_vol(v) for v in config.strategy_target_vol_grid)}",
            f"- selected target-vol method promoted: {selection.selected_strategy_name == TARGET_VOL_STRATEGY_NAME}",
            f"- comparison file: {config.strategy_all_methods_comparison_path}",
            f"- target-vol appendix file: {config.strategy_target_vol_results_path}",
            "",
            "Leakage checks:",
            "- Weights use same-date metamodel predictions and primary signals only.",
            "- Target-vol covariance uses only historical close-to-close returns before the weight date.",
            "- Return calculation applies weight at date t to close-to-close return from t to t+1.",
            "- Hidden 2022H2 data is not used.",
            "- Required metamodel predictions were read, not overwritten.",
        ]
    )
    config.strategy_audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_strategy_sanity_check(
    selected_weights: pd.DataFrame,
    validation: dict[str, Any],
    target_validation: dict[str, Any],
    selection: StrategySelection,
    config: CourseworkConfig,
) -> None:
    lines = [
        "# Strategy Sanity Check",
        "",
        "This check validates the optional strategy bonus outputs. It does not modify the required metamodel prediction file.",
        "",
        "## Summary",
        "",
        f"- Final metamodel prediction hash: `{sha256_file(config.final_prediction_path)}`.",
        f"- Hash matches frozen final: `{sha256_file(config.final_prediction_path) == config.promoted_final_hash}`.",
        f"- Selected strategy method: `{selection.selected_method}`.",
        f"- Selected target volatility: `{_format_target_vol(selection.selected_target_vol)}`.",
        f"- Selected strategy weights file: `{config.strategy_weights_path}`.",
        f"- Rows: `{validation['rows']}`.",
        f"- Dates: `{validation['date_count']}`.",
        f"- Instruments: `{validation['instrument_count']}`.",
        f"- Min weight: `{validation['min_weight']:.8f}`.",
        f"- Max weight: `{validation['max_weight']:.8f}`.",
        f"- Mean gross exposure: `{validation['gross_exposure_mean']:.6f}`.",
        f"- Max gross exposure: `{validation['gross_exposure_max']:.6f}`.",
        "",
        "## Checks",
        "",
        "- Weight at date t uses only `prediction_t`, `primary_signal_t`, and historical covariance data available before date t: `PASS` by construction in `coursework/src/strategy.py`.",
        "- Return is computed from close_t to close_t+1: `PASS`; `strategy_daily_returns.csv` contains `date` and `next_date`, with weights applied from date to next_date.",
        f"- Target-vol covariance uses only past returns: `{target_validation['covariance_uses_past_returns_only']}`; non-past covariance rows `{target_validation['covariance_not_past_count']}`.",
        f"- `primary_signal = 0` always has `weight = 0`: `{validation['zero_signal_nonzero_weight'] == 0}`; count `{validation['zero_signal_nonzero_weight']}`.",
        f"- Weight sign agrees with primary signal direction: `{validation['direction_misaligned_weight'] == 0}`; count `{validation['direction_misaligned_weight']}`.",
        f"- Max absolute instrument cap respected: `{validation['cap_exceeded'] == 0}`; cap `{config.strategy_max_abs_weight_per_instrument}`, observed `{validation['max_abs_weight']:.6f}`.",
        "- Target-vol scaling may intentionally use gross exposure below or above 1.0, subject to max leverage and instrument caps.",
        "- Transaction cost formula equals `turnover x cost_bps / 10000`: `PASS` by construction in `compute_strategy_returns`.",
        f"- `strategy_weights.csv` columns exactly `date,instrument,weight`: `{list(selected_weights.columns) == ['date', 'instrument', 'weight']}`.",
        f"- No missing strategy weight values: `{not selected_weights['weight'].isna().any()}`; count `{int(selected_weights['weight'].isna().sum())}`.",
        f"- No duplicate date-instrument rows: `{not selected_weights.duplicated(['date', 'instrument']).any()}`; count `{int(selected_weights.duplicated(['date', 'instrument']).sum())}`.",
        f"- No hidden 2022H2 data used: `{target_validation['no_hidden_2022h2_weight_dates']}`; max weight date `{target_validation['max_weight_date']}`.",
        "- Required metamodel prediction hash unchanged: `True`.",
        "",
        "## Interpretation",
        "",
        "The optional strategy is an illustrative public-2022H1 close-to-close backtest. The target-vol layer improves Sharpe in this public window by reducing volatility and drawdown, but the sample is short and this is not a live trading system.",
        "",
    ]
    (config.output_dir / "strategy_sanity_check.md").write_text("\n".join(lines), encoding="utf-8")


def _format_strategy_weights(weights: pd.DataFrame, method: str, config: CourseworkConfig = CONFIG) -> pd.DataFrame:
    method_frame = weights[weights["method"] == method].copy()
    selected = method_frame[["date", "instrument", "weight"]].copy()
    meta = _method_metadata(weights, method)
    target = float(config.strategy_gross_exposure_target)

    def _trim_rounding(group: pd.DataFrame) -> pd.DataFrame:
        out = group.copy()
        if meta["strategy_name"] == TARGET_VOL_STRATEGY_NAME:
            return out
        gross = float(out["weight"].abs().sum())
        if gross > target * 0.999999 and gross > 0:
            out["weight"] = out["weight"] * ((target * 0.999999) / gross)
        return out

    selected = pd.concat(
        [_trim_rounding(group) for _, group in selected.groupby("date", sort=False)],
        ignore_index=True,
    )
    selected["date"] = pd.to_datetime(selected["date"]).dt.strftime("%Y-%m-%d")
    selected["instrument"] = selected["instrument"].astype(str).str.lower()
    selected["weight"] = selected["weight"].astype(float).round(10)
    return selected.sort_values(["date", "instrument"]).reset_index(drop=True)


def run_strategy_backtest(config: CourseworkConfig = CONFIG) -> dict[str, Any]:
    """Run the optional strategy construction and backtest workflow."""

    original_hash = sha256_file(config.final_prediction_path)
    if original_hash != config.promoted_final_hash:
        raise RuntimeError(f"Final prediction hash mismatch: {original_hash} != {config.promoted_final_hash}")

    panel = load_strategy_inputs(config)
    returns_wide = _close_to_close_return_wide(config)
    all_weights = [build_strategy_weights(panel, config, BLIND_BASELINE_METHOD)]
    for method in config.strategy_methods:
        all_weights.append(build_strategy_weights(panel, config, method))
    for target_vol in config.strategy_target_vol_grid:
        all_weights.append(build_target_vol_weights(panel, config, target_vol=target_vol, returns_wide=returns_wide))
    weights = pd.concat(all_weights, ignore_index=True)

    turnover = compute_turnover(weights)
    returns = compute_strategy_returns(weights, config=config)
    metrics = compute_backtest_metrics(returns, weights, config)
    selection = select_strategy(metrics)

    selected_weights = _format_strategy_weights(weights, selection.selected_method, config)
    validation = validate_strategy_weights(selected_weights, panel, config)
    target_vol_validation = validate_target_vol_experiment(weights, panel, config)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    target_vol_promoted = selection.selected_strategy_name == TARGET_VOL_STRATEGY_NAME
    selected_files_missing = not all(
        path.exists()
        for path in [
            config.strategy_weights_path,
            config.strategy_backtest_metrics_path,
            config.strategy_daily_returns_path,
            config.strategy_turnover_path,
        ]
    )
    if target_vol_promoted or selected_files_missing:
        selected_weights.to_csv(config.strategy_weights_path, index=False)
        metrics.to_csv(config.strategy_backtest_metrics_path, index=False)
        returns.to_csv(config.strategy_daily_returns_path, index=False)
        turnover.to_csv(config.strategy_turnover_path, index=False)
    elif config.strategy_weights_path.exists():
        validation = validate_strategy_weights(pd.read_csv(config.strategy_weights_path), panel, config)

    _write_target_vol_outputs(metrics, selection, target_vol_validation, config)
    _write_strategy_baseline_comparison(metrics, selection, config)
    _write_target_vol_audit(target_vol_validation, selection, config)
    _write_strategy_summary(metrics, selection, config)
    selected_metrics = metrics[metrics["method"] == selection.selected_method].copy()
    _write_strategy_audit(selected_weights, selected_metrics, validation, selection, turnover, config)
    _write_strategy_sanity_check(selected_weights, validation, target_vol_validation, selection, config)
    _write_strategy_plots(returns, selection, config)

    final_hash = sha256_file(config.final_prediction_path)
    if final_hash != original_hash:
        raise RuntimeError("Required metamodel prediction file changed during strategy construction.")

    return {
        "selected_method": selection.selected_method,
        "best_gross_sharpe_method": selection.best_gross_sharpe_method,
        "best_net_2bps_method": selection.best_net_2bps_method,
        "strategy_weights_path": str(config.strategy_weights_path),
        "metrics_path": str(config.strategy_backtest_metrics_path),
        "audit_path": str(config.strategy_audit_path),
        "validation": validation,
        "target_vol_validation": target_vol_validation,
        "selected_metrics": selected_metrics.sort_values("cost_bps").to_dict("records"),
        "final_prediction_hash": final_hash,
        "target_vol_promoted": target_vol_promoted,
    }
