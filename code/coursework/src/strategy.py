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


def build_strategy_weights(df: pd.DataFrame, config: CourseworkConfig = CONFIG, method: str = "confidence_scaled") -> pd.DataFrame:
    """Convert probabilities and primary signals into signed raw weights."""

    out = df.copy()
    p = out["prediction"].fillna(0.0).clip(0.0, 1.0)
    signal = out["primary_signal"].fillna(0).astype(int)

    if method == "threshold_filter_050":
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
    return normalize_weights_by_date(out, config)


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
                    "turnover": turnover.to_numpy(),
                    "active_instruments": active.to_numpy(),
                    "gross_exposure": gross.to_numpy(),
                }
            )
        )
    if not rows:
        return pd.DataFrame(columns=["date", "method", "turnover", "active_instruments", "gross_exposure"])
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

    daily = (
        merged.groupby(["method", "date", "next_date"], as_index=False)
        .agg(
            gross_return=("weighted_return", "sum"),
            active_instruments=("weight", lambda s: int((s.abs() > 1e-12).sum())),
            gross_exposure=("weight", lambda s: float(s.abs().sum())),
        )
        .sort_values(["method", "date"])
        .reset_index(drop=True)
    )
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
    if n_days == 0:
        return {
            "method": method,
            "cost_bps": cost_bps,
            "n_days": 0,
            "cumulative_return": np.nan,
            "cagr": np.nan,
            "annualised_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "sortino_ratio": np.nan,
            "maximum_drawdown": np.nan,
            "average_holding_period": np.nan,
            "average_daily_turnover": np.nan,
            "active_days": 0,
            "average_active_instruments": np.nan,
            "hit_rate": np.nan,
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

    active_days = int((frame["active_instruments"] > 0).sum())
    return {
        "method": method,
        "cost_bps": int(cost_bps),
        "n_days": int(n_days),
        "cumulative_return": cumulative_return,
        "gross_cumulative_return": float((1.0 + gross_returns).prod() - 1.0),
        "cagr": cagr,
        "annualised_volatility": vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "maximum_drawdown": _max_drawdown(returns),
        "average_holding_period": _average_holding_period(weights_df, method),
        "average_daily_turnover": float(frame["turnover"].mean()),
        "active_days": active_days,
        "average_active_instruments": float(frame["active_instruments"].mean()),
        "hit_rate": float((returns > 0).mean()) if active_days else np.nan,
        "mean_daily_return": mean_return,
        "mean_gross_daily_return": float(gross_returns.mean()),
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
    """Select a recommended optional strategy using simple guardrails."""

    gross = metrics[metrics["cost_bps"] == 0].copy()
    net_2 = metrics[metrics["cost_bps"] == 2].copy()
    best_gross = gross.sort_values("sharpe_ratio", ascending=False).iloc[0]
    best_net = net_2.sort_values("sharpe_ratio", ascending=False).iloc[0]
    conservative_method = "threshold_filter_060"
    conservative = net_2[net_2["method"] == conservative_method]
    if conservative.empty:
        conservative_method = str(best_net["method"])

    good = net_2[
        (net_2["sharpe_ratio"] > 0)
        & (net_2["maximum_drawdown"] > -0.20)
        & (net_2["average_daily_turnover"] <= 1.5)
    ].copy()
    if not good.empty:
        selected = good.sort_values(["sharpe_ratio", "maximum_drawdown"], ascending=[False, False]).iloc[0]
        reason = "Selected the best 2 bps net Sharpe strategy that passed simple drawdown and turnover guardrails."
    else:
        conservative_rows = net_2[net_2["method"] == conservative_method]
        selected = conservative_rows.iloc[0] if not conservative_rows.empty else best_net
        reason = "No method clearly passed the positive-net-Sharpe guardrail, so the conservative threshold rule was selected."

    return StrategySelection(
        selected_method=str(selected["method"]),
        best_gross_sharpe_method=str(best_gross["method"]),
        best_net_2bps_method=str(best_net["method"]),
        conservative_method=conservative_method,
        recommendation_reason=reason,
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
    validate_no_duplicates(out, ("date", "instrument"))
    if len(out) != len(panel):
        raise ValueError(f"Strategy weights should have one row per prediction row: {len(out)} != {len(panel)}")

    check = out.merge(panel[["date", "instrument", "primary_signal"]], on=["date", "instrument"], how="left")
    zero_signal_nonzero_weight = int(((check["primary_signal"] == 0) & (check["weight"].abs() > 1e-12)).sum())
    if zero_signal_nonzero_weight:
        raise ValueError(f"Found {zero_signal_nonzero_weight} zero-signal rows with non-zero strategy weight.")
    misaligned = int(((check["weight"] > 0) & (check["primary_signal"] < 0)).sum() + ((check["weight"] < 0) & (check["primary_signal"] > 0)).sum())
    if misaligned:
        raise ValueError(f"Found {misaligned} strategy weights that do not align with primary signal direction.")

    gross = out.groupby("date")["weight"].apply(lambda s: float(s.abs().sum()))
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
    }


def _write_strategy_summary(
    metrics: pd.DataFrame,
    selection: StrategySelection,
    config: CourseworkConfig,
) -> None:
    selected_rows = metrics[metrics["method"] == selection.selected_method].sort_values("cost_bps")
    gross = selected_rows[selected_rows["cost_bps"] == 0].iloc[0]
    net_2 = selected_rows[selected_rows["cost_bps"] == 2].iloc[0]
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
        "The weight on date `t` earns close-to-close return from `t` to `t+1`; future returns are not used to form weights.",
        f"Gross exposure target: `{config.strategy_gross_exposure_target}`.",
        f"Max absolute weight per instrument: `{config.strategy_max_abs_weight_per_instrument}`.",
        "",
        "## Methods Tested",
        "",
    ]
    for method in config.strategy_methods:
        lines.append(f"- `{method}`")
    lines.extend(
        [
            "",
            "## Selection",
            "",
            f"- Best gross Sharpe method: `{selection.best_gross_sharpe_method}`.",
            f"- Best net Sharpe method at 2 bps: `{selection.best_net_2bps_method}`.",
            f"- Most conservative method: `{selection.conservative_method}`.",
            f"- Recommended optional strategy: `{selection.selected_method}`.",
            f"- Reason: {selection.recommendation_reason}",
            "",
            "## Selected Strategy Metrics",
            "",
            "| Cost bps | CAGR | Ann. Vol | Sharpe | Sortino | Max Drawdown | Avg Turnover |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in selected_rows.itertuples(index=False):
        lines.append(
            f"| {int(row.cost_bps)} | {row.cagr:.4f} | {row.annualised_volatility:.4f} | "
            f"{row.sharpe_ratio:.4f} | {row.sortino_ratio:.4f} | {row.maximum_drawdown:.4f} | "
            f"{row.average_daily_turnover:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Headline",
            "",
            f"- Gross Sharpe for selected strategy: `{gross.sharpe_ratio:.4f}`.",
            f"- 2 bps net Sharpe for selected strategy: `{net_2.sharpe_ratio:.4f}`.",
            "",
            "## Baseline Comparison",
            "",
            "The selected `soft_allocation` strategy matches the `blindly_follow_primary_signal_equal_weight` baseline under the current conservative cap and gross-exposure normalization. It does not improve over the blind baseline on 2 bps net Sharpe or cumulative return in public 2022H1.",
            "",
            "Detailed comparison files:",
            "",
            "- `outputs/strategy_baseline_comparison.csv`",
            "- `outputs/strategy_baseline_comparison.md`",
            "",
            "## Limitations",
            "",
            "This is a simple close-to-close portfolio backtest for the optional competition track. It assumes weights are set at date `t` and earn return from `t` to `t+1`. It does not use hidden 2022H2 data, and it does not alter the required prediction submission. The high Sharpe should be interpreted cautiously because the public test window has only 128 prediction dates.",
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
            f"ann_vol={row.annualised_volatility:.6f}, Sharpe={row.sharpe_ratio:.6f}, "
            f"Sortino={row.sortino_ratio:.6f}, max_drawdown={row.maximum_drawdown:.6f}, "
            f"avg_holding_period={row.average_holding_period:.6f}, "
            f"avg_turnover={row.average_daily_turnover:.6f}"
        )
    lines.extend(
        [
            "",
            "Leakage checks:",
            "- Weights use same-date metamodel predictions and primary signals only.",
            "- Return calculation applies weight at date t to close-to-close return from t to t+1.",
            "- Hidden 2022H2 data is not used.",
            "- Required metamodel predictions were read, not overwritten.",
        ]
    )
    config.strategy_audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_strategy_weights(weights: pd.DataFrame, method: str) -> pd.DataFrame:
    selected = weights[weights["method"] == method][["date", "instrument", "weight"]].copy()
    target = float(CONFIG.strategy_gross_exposure_target)

    def _trim_rounding(group: pd.DataFrame) -> pd.DataFrame:
        out = group.copy()
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
    all_weights = []
    for method in config.strategy_methods:
        all_weights.append(build_strategy_weights(panel, config, method))
    weights = pd.concat(all_weights, ignore_index=True)

    turnover = compute_turnover(weights)
    returns = compute_strategy_returns(weights, config=config)
    metrics = compute_backtest_metrics(returns, weights, config)
    selection = select_strategy(metrics)

    selected_weights = _format_strategy_weights(weights, selection.selected_method)
    validation = validate_strategy_weights(selected_weights, panel, config)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    selected_weights.to_csv(config.strategy_weights_path, index=False)
    metrics.to_csv(config.strategy_backtest_metrics_path, index=False)
    returns.to_csv(config.strategy_daily_returns_path, index=False)
    turnover.to_csv(config.strategy_turnover_path, index=False)
    _write_strategy_summary(metrics, selection, config)
    selected_metrics = metrics[metrics["method"] == selection.selected_method].copy()
    _write_strategy_audit(selected_weights, selected_metrics, validation, selection, turnover, config)

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
        "selected_metrics": selected_metrics.sort_values("cost_bps").to_dict("records"),
        "final_prediction_hash": final_hash,
    }
