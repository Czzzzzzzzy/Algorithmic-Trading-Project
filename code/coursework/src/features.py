"""Feature engineering for the metamodel.

This module builds lagged OHLCV-derived, technical, regime, cross-sectional,
and signal-history features while avoiding same-day or future information.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .config import (
    ASSET_CLASS,
    CONTROLLED_FEATURE_GROUPS,
    CourseworkConfig,
    HMM_EXTENSION_FEATURES,
    RANDOM_STATE,
    SIGNAL_HISTORY_FEATURES,
    USE_HMM_EXTENSION,
)
from .data_loader import merge_price_and_signal_data
from .utils import rolling_percentile_rank, rolling_trend_tstat, rsi, safe_divide


def add_group_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("date").copy()
    close = group["close"]
    high = group["high"]
    low = group["low"]
    open_ = group["open"]
    volume = group["volume"]
    oi = group["open_interest"]
    prev_close = close.shift(1)

    ret = close.pct_change()
    log_ret = np.log(close).diff()
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    group["ret_1"] = ret
    group["log_ret_1"] = log_ret
    group["range_pct"] = safe_divide(high - low, close)
    group["body_pct"] = safe_divide(close - open_, open_)
    group["gap_pct"] = safe_divide(open_, prev_close) - 1
    group["gap_return"] = group["gap_pct"]
    group["overnight_return"] = safe_divide(open_, prev_close) - 1
    intraday_return = safe_divide(close, open_) - 1
    group["intraday_reversal"] = -np.sign(group["overnight_return"]) * intraday_return
    group["atr_14"] = safe_divide(true_range.rolling(14, min_periods=10).mean(), close)
    group["parkinson_vol_20"] = np.sqrt(
        (np.log(high / low) ** 2).rolling(20, min_periods=15).mean() / (4 * np.log(2))
    )

    for window in (5, 10, 20, 63):
        group[f"mom_{window}"] = close.pct_change(window)
        group[f"rv_{window}"] = ret.rolling(window, min_periods=max(3, window // 2)).std()
        sma = close.rolling(window, min_periods=max(3, window // 2)).mean()
        sd = close.rolling(window, min_periods=max(3, window // 2)).std()
        group[f"ma_gap_{window}"] = safe_divide(close, sma) - 1
        group[f"price_z_{window}"] = safe_divide(close - sma, sd)

    group["ewma_vol_20"] = ret.ewm(span=20, adjust=False, min_periods=10).std()
    group["ewma_vol_60"] = ret.ewm(span=60, adjust=False, min_periods=20).std()
    group["downside_vol_20"] = ret.clip(upper=0).rolling(20, min_periods=10).std()
    group["vol_of_vol_20d"] = group["rv_20"].rolling(20, min_periods=10).std()
    group["vol_percentile_63d"] = rolling_percentile_rank(group["rv_20"], 63)
    group["vol_percentile_252d"] = rolling_percentile_rank(group["rv_20"], 252)
    group["atr_percentile_252d"] = rolling_percentile_rank(group["atr_14"], 252)
    group["range_percentile_63d"] = rolling_percentile_rank(group["range_pct"], 63)
    for window in (20, 60, 252):
        rolling_peak = close.rolling(window, min_periods=max(10, window // 3)).max()
        group[f"drawdown_{window}d"] = safe_divide(close, rolling_peak) - 1

    sma_20 = close.rolling(20, min_periods=15).mean()
    sd_20 = close.rolling(20, min_periods=15).std()
    group["bollinger_pos_20"] = safe_divide(close - sma_20, 2 * sd_20)
    group["bollinger_bandwidth_20"] = safe_divide(4 * sd_20, sma_20)
    group["rsi_14"] = rsi(close, 14)

    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    group["macd_pct"] = safe_divide(macd, close)
    group["macd_signal_pct"] = safe_divide(macd_signal, close)
    group["macd_hist_pct"] = safe_divide(macd - macd_signal, close)

    low_14 = low.rolling(14, min_periods=10).min()
    high_14 = high.rolling(14, min_periods=10).max()
    group["stoch_14"] = safe_divide(close - low_14, high_14 - low_14)

    dollar_volume = close * volume
    group["volume_chg_1"] = volume.pct_change()
    group["volume_z_20"] = safe_divide(
        volume - volume.rolling(20, min_periods=10).mean(),
        volume.rolling(20, min_periods=10).std(),
    )
    group["volume_trend_20_60"] = safe_divide(
        volume.rolling(20, min_periods=10).mean(),
        volume.rolling(60, min_periods=30).mean(),
    ) - 1
    group["log_dollar_volume"] = np.log(dollar_volume.clip(lower=1))
    group["amihud_20"] = safe_divide(ret.abs(), dollar_volume).rolling(20, min_periods=10).mean()

    group["open_interest_chg_1"] = oi.pct_change()
    group["open_interest_z_20"] = safe_divide(
        oi - oi.rolling(20, min_periods=10).mean(),
        oi.rolling(20, min_periods=10).std(),
    )
    group["open_interest_trend_20_60"] = safe_divide(
        oi.rolling(20, min_periods=10).mean(),
        oi.rolling(60, min_periods=30).mean(),
    ) - 1
    group["volume_oi_ratio"] = safe_divide(volume, oi)

    log_price = np.log(close)
    group["trend_scan_tstat_20"] = rolling_trend_tstat(log_price, 20)
    group["trend_scan_tstat_60"] = rolling_trend_tstat(log_price, 60)
    scan_windows = [10, 20, 40, 60]
    scan_frame = pd.concat({window: rolling_trend_tstat(log_price, window) for window in scan_windows}, axis=1)
    scan_abs = scan_frame.abs().to_numpy(dtype=float)
    all_nan = np.isnan(scan_abs).all(axis=1)
    best_pos = np.argmax(np.where(np.isnan(scan_abs), -np.inf, scan_abs), axis=1)
    best_window = np.asarray(scan_windows, dtype=float)[best_pos]
    best_tstat = scan_frame.to_numpy(dtype=float)[np.arange(len(scan_frame)), best_pos]
    best_window[all_nan] = np.nan
    best_tstat[all_nan] = np.nan
    group["trend_scan_best_window"] = best_window
    group["trend_scan_sign"] = np.sign(best_tstat)
    group["trend_scan_abs_tstat"] = np.abs(best_tstat)

    feature_cols = [c for c in group.columns if c not in {"date", "instrument", "open", "high", "low", "close", "volume", "open_interest"}]
    group[feature_cols] = group[feature_cols].shift(1)
    return group


@dataclass
class DiagonalGaussianHMM:
    startprob: np.ndarray
    transmat: np.ndarray
    means: np.ndarray
    variances: np.ndarray


def logsumexp(a: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray:
    max_a = np.max(a, axis=axis, keepdims=True)
    max_a = np.where(np.isfinite(max_a), max_a, 0.0)
    summed = np.sum(np.exp(a - max_a), axis=axis, keepdims=True)
    out = max_a + np.log(summed.clip(1e-300))
    if not keepdims and axis is not None:
        out = np.squeeze(out, axis=axis)
    return out


def diagonal_gaussian_log_prob(x: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    variances = np.clip(variances, 1e-5, None)
    diff = x[:, None, :] - means[None, :, :]
    return -0.5 * (
        np.sum(np.log(2 * np.pi * variances), axis=1)[None, :]
        + np.sum((diff**2) / variances[None, :, :], axis=2)
    )


def forward_backward(
    log_emissions: np.ndarray,
    startprob: np.ndarray,
    transmat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    n_obs, n_states = log_emissions.shape
    log_start = np.log(startprob.clip(1e-12))
    log_trans = np.log(transmat.clip(1e-12))

    log_alpha = np.empty((n_obs, n_states))
    log_alpha[0] = log_start + log_emissions[0]
    for t in range(1, n_obs):
        log_alpha[t] = log_emissions[t] + logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0)

    log_beta = np.zeros((n_obs, n_states))
    for t in range(n_obs - 2, -1, -1):
        log_beta[t] = logsumexp(log_trans + log_emissions[t + 1][None, :] + log_beta[t + 1][None, :], axis=1)

    log_likelihood = float(logsumexp(log_alpha[-1], axis=0))
    gamma = np.exp(log_alpha + log_beta - log_likelihood)
    gamma = gamma / gamma.sum(axis=1, keepdims=True).clip(1e-12)

    xi_sum = np.zeros((n_states, n_states))
    for t in range(n_obs - 1):
        log_xi = (
            log_alpha[t][:, None]
            + log_trans
            + log_emissions[t + 1][None, :]
            + log_beta[t + 1][None, :]
            - log_likelihood
        )
        xi_sum += np.exp(log_xi)
    return gamma, xi_sum, log_likelihood


def fit_diagonal_gaussian_hmm(
    x: np.ndarray,
    n_states: int = 3,
    seed: int = RANDOM_STATE,
    max_iter: int = 35,
) -> DiagonalGaussianHMM:
    if len(x) < n_states * 20:
        raise ValueError("Not enough observations to fit HMM.")

    init = GaussianMixture(
        n_components=n_states,
        covariance_type="diag",
        reg_covar=1e-4,
        n_init=3,
        random_state=seed,
    )
    init.fit(x)
    labels = init.predict(x)

    means = init.means_.copy()
    variances = np.clip(init.covariances_.copy(), 1e-5, None)
    state_counts = np.bincount(labels, minlength=n_states).astype(float) + 1.0
    startprob = state_counts / state_counts.sum()
    trans_counts = np.ones((n_states, n_states))
    for prev, current in zip(labels[:-1], labels[1:]):
        trans_counts[prev, current] += 1.0
    transmat = trans_counts / trans_counts.sum(axis=1, keepdims=True)

    previous_log_likelihood = -np.inf
    for _ in range(max_iter):
        log_emissions = diagonal_gaussian_log_prob(x, means, variances)
        gamma, xi_sum, log_likelihood = forward_backward(log_emissions, startprob, transmat)

        weights = gamma.sum(axis=0).clip(1e-8)
        startprob = (gamma[0] + 1e-3)
        startprob = startprob / startprob.sum()
        transmat = xi_sum + 1e-3
        transmat = transmat / transmat.sum(axis=1, keepdims=True)
        means = (gamma.T @ x) / weights[:, None]
        variances = np.sum(gamma[:, :, None] * ((x[:, None, :] - means[None, :, :]) ** 2), axis=0) / weights[:, None]
        variances = np.clip(variances, 1e-5, None)

        if abs(log_likelihood - previous_log_likelihood) < 1e-4:
            break
        previous_log_likelihood = log_likelihood

    return DiagonalGaussianHMM(startprob=startprob, transmat=transmat, means=means, variances=variances)


def hmm_filtered_features(model: DiagonalGaussianHMM, x: np.ndarray) -> dict[str, np.ndarray]:
    log_emissions = diagonal_gaussian_log_prob(x, model.means, model.variances)
    n_obs, n_states = log_emissions.shape
    filtered = np.zeros((n_obs, n_states))
    change_prob = np.zeros(n_obs)
    days_in_state = np.ones(n_obs)
    state_id = np.zeros(n_obs, dtype=int)

    alpha = model.startprob.copy()
    for t in range(n_obs):
        if t > 0:
            change_prob[t] = 1.0 - float(np.dot(alpha, np.diag(model.transmat)))
            alpha = alpha @ model.transmat
        likelihood = np.exp(log_emissions[t] - np.max(log_emissions[t]))
        alpha = alpha * likelihood
        alpha = alpha / alpha.sum().clip(1e-12)
        filtered[t] = alpha
        state_id[t] = int(np.argmax(alpha))
        if t > 0 and state_id[t] == state_id[t - 1]:
            days_in_state[t] = days_in_state[t - 1] + 1

    entropy = -(filtered * np.log(filtered.clip(1e-12))).sum(axis=1) / np.log(n_states)
    out = {
        "hmm_entropy": entropy,
        "hmm_state_id": state_id.astype(float),
        "hmm_days_in_state": days_in_state,
        "hmm_state_change_prob": change_prob,
    }
    for state in range(n_states):
        out[f"hmm_state_{state}_prob"] = filtered[:, state]
    return out


def add_unsupervised_features(features: pd.DataFrame, seed: int, use_hmm_extension: bool = USE_HMM_EXTENSION) -> pd.DataFrame:
    """Fit GMM/HMM/PCA on pre-2020 lagged history per instrument and transform all rows."""
    latent_input_cols = [
        "ret_1",
        "rv_20",
        "ewma_vol_60",
        "mom_20",
        "range_pct",
        "volume_z_20",
        "open_interest_z_20",
    ]
    out = features.copy()
    latent_cols = [
        "gmm_regime_0",
        "gmm_regime_1",
        "gmm_regime_2",
        "gmm_entropy",
        "gmm_regime_id",
        "pca_ohlcv_1",
        "pca_ohlcv_2",
        "pca_ohlcv_3",
    ]
    if use_hmm_extension:
        latent_cols += HMM_EXTENSION_FEATURES
    for col in latent_cols:
        out[col] = np.nan

    for inst, idx in out.groupby("instrument").groups.items():
        inst_frame = out.loc[idx].sort_values("date")
        available_cols = [c for c in latent_input_cols if c in inst_frame.columns]
        train = inst_frame.loc[inst_frame["date"] < pd.Timestamp("2020-01-01"), available_cols].replace([np.inf, -np.inf], np.nan).dropna()
        transform = inst_frame[available_cols].replace([np.inf, -np.inf], np.nan)
        good_rows = transform.dropna().index
        if len(train) < 250 or len(good_rows) == 0:
            continue

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train)
        transform_scaled = scaler.transform(transform.loc[good_rows])

        try:
            gmm = GaussianMixture(n_components=3, covariance_type="full", reg_covar=1e-4, random_state=seed)
            gmm.fit(train_scaled)
            proba = gmm.predict_proba(transform_scaled)
            entropy = -(proba * np.log(proba.clip(1e-12))).sum(axis=1) / np.log(proba.shape[1])
            out.loc[good_rows, ["gmm_regime_0", "gmm_regime_1", "gmm_regime_2"]] = proba
            out.loc[good_rows, "gmm_entropy"] = entropy
            out.loc[good_rows, "gmm_regime_id"] = proba.argmax(axis=1)
        except Exception:
            pass

        if use_hmm_extension:
            try:
                hmm = fit_diagonal_gaussian_hmm(train_scaled, n_states=3, seed=seed)
                hmm_features = hmm_filtered_features(hmm, transform_scaled)
                for col, values in hmm_features.items():
                    out.loc[good_rows, col] = values
            except Exception:
                pass

        try:
            n_components = min(3, train_scaled.shape[1])
            pca = PCA(n_components=n_components, random_state=seed)
            pca.fit(train_scaled)
            comps = pca.transform(transform_scaled)
            for j in range(n_components):
                out.loc[good_rows, f"pca_ohlcv_{j + 1}"] = comps[:, j]
        except Exception:
            pass

    return out


def add_cross_sectional_features(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    rank_specs = {
        "cs_mom_20_rank": "mom_20",
        "cs_mom_63_rank": "mom_63",
        "cs_vol_20_rank": "rv_20",
        "cs_volume_rank": "log_dollar_volume",
        "cs_rsi_rank": "rsi_14",
    }
    for new_col, source_col in rank_specs.items():
        out[new_col] = out.groupby("date")[source_col].rank(pct=True)

    class_mom = (
        out.groupby(["date", "asset_class"])["mom_20"]
        .mean()
        .rename("asset_class_mom_20")
        .reset_index()
    )
    class_vol = (
        out.groupby(["date", "asset_class"])["rv_20"]
        .mean()
        .rename("asset_class_vol_20")
        .reset_index()
    )
    out = out.merge(class_mom, on=["date", "asset_class"], how="left")
    out = out.merge(class_vol, on=["date", "asset_class"], how="left")
    return out


def build_features(ohlcv: pd.DataFrame, seed: int, use_hmm_extension: bool = USE_HMM_EXTENSION) -> pd.DataFrame:
    ohlcv = ohlcv.copy()
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    ohlcv["instrument"] = ohlcv["instrument"].str.lower()
    ohlcv["asset_class"] = ohlcv["instrument"].map(ASSET_CLASS)
    ohlcv = ohlcv.sort_values(["instrument", "date"])

    feature_frames = []
    for _, group in ohlcv.groupby("instrument", sort=False):
        feature_frames.append(add_group_features(group))
    features = pd.concat(feature_frames, ignore_index=True)
    features = add_unsupervised_features(features, seed, use_hmm_extension=use_hmm_extension)
    features = add_cross_sectional_features(features)
    return features.sort_values(["date", "instrument"]).reset_index(drop=True)


def add_signal_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["instrument", "date"]).copy()
    out["signal_abs"] = out["primary_signal"].abs()
    out["signal_long"] = (out["primary_signal"] == 1).astype(int)
    out["signal_short"] = (out["primary_signal"] == -1).astype(int)
    out["signal_zero"] = (out["primary_signal"] == 0).astype(int)

    pieces = []
    for _, group in out.groupby("instrument", sort=False):
        group = group.copy()
        prev = group["primary_signal"].shift(1).fillna(0)
        group["prev_signal"] = prev
        group["signal_changed"] = (group["primary_signal"] != prev).astype(int)
        run_id = group["signal_changed"].cumsum()
        group["signal_run_length"] = group.groupby(run_id).cumcount() + 1
        group["signal_nonzero_rate_20"] = group["signal_abs"].shift(1).rolling(20, min_periods=1).mean()
        group["signal_long_rate_20"] = group["signal_long"].shift(1).rolling(20, min_periods=1).mean()
        group["signal_short_rate_20"] = group["signal_short"].shift(1).rolling(20, min_periods=1).mean()
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True).sort_values(["date", "instrument"]).reset_index(drop=True)


def add_side_adjusted_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    side = out["primary_signal"].astype(float)
    out["side_return_1d"] = side * out.get("ret_1", np.nan)
    out["side_return_5d"] = side * out.get("mom_5", np.nan)
    out["side_momentum_20d"] = side * out.get("mom_20", np.nan)
    out["side_vol_adj_momentum"] = side * safe_divide(out.get("mom_20", pd.Series(np.nan, index=out.index)), out.get("rv_20", pd.Series(np.nan, index=out.index)))
    out["side_rsi"] = side * ((out.get("rsi_14", pd.Series(np.nan, index=out.index)) - 50.0) / 50.0)
    out["side_macd"] = side * out.get("macd_hist_pct", np.nan)
    out["side_bollinger_position"] = side * out.get("bollinger_pos_20", np.nan)
    return out


def add_regime_interaction_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["instrument", "date"]).copy()
    side = out["primary_signal"].astype(float)
    gmm_cols = [f"gmm_regime_{i}" for i in range(3) if f"gmm_regime_{i}" in out.columns]
    for i, col in enumerate(gmm_cols):
        out[f"gmm_state_{i}_x_signal"] = side * out[col]
    for i in range(1, 4):
        col = f"pca_ohlcv_{i}"
        if col in out.columns:
            out[f"{col}_x_signal"] = side * out[col]
    if gmm_cols:
        out["gmm_max_prob"] = out[gmm_cols].max(axis=1)
    else:
        out["gmm_max_prob"] = np.nan
    if "gmm_regime_id" in out.columns:
        out["gmm_state_change"] = (
            out.groupby("instrument")["gmm_regime_id"]
            .transform(lambda s: (s != s.shift(1)).astype(float))
            .where(out["gmm_regime_id"].notna(), np.nan)
        )
    else:
        out["gmm_state_change"] = np.nan
    return out.sort_values(["date", "instrument"]).reset_index(drop=True)


def recent_streak(history: list[dict[str, Any]], target: int) -> int:
    streak = 0
    for event in reversed(history):
        if int(event["label"]) == target:
            streak += 1
        else:
            break
    return streak


def add_signal_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.sort_values(["instrument", "date"]).copy()
    for col in SIGNAL_HISTORY_FEATURES + ["regime_conditioned_hit_rate_60d"]:
        out[col] = np.nan

    pieces = []
    for _, group in out.groupby("instrument", sort=False):
        group = group.sort_values("date").copy()
        events = (
            group[
                (group["primary_signal"] != 0)
                & group["label"].notna()
                & group["label_end_date"].notna()
            ][["date", "label_end_date", "primary_signal", "label", "side_return", "gmm_regime_id"]]
            .sort_values(["label_end_date", "date"])
            .to_dict("records")
        )
        history: list[dict[str, Any]] = []
        event_idx = 0
        values: dict[str, list[float]] = {col: [] for col in SIGNAL_HISTORY_FEATURES + ["regime_conditioned_hit_rate_60d"]}

        for row in group.itertuples(index=False):
            current_date = row.date
            while event_idx < len(events) and pd.Timestamp(events[event_idx]["label_end_date"]) < current_date:
                history.append(events[event_idx])
                event_idx += 1

            def window_events(days: int) -> list[dict[str, Any]]:
                cutoff = current_date - pd.Timedelta(days=days)
                return [event for event in history if pd.Timestamp(event["label_end_date"]) >= cutoff]

            recent_20 = window_events(20)
            recent_60 = window_events(60)
            recent_120 = window_events(120)
            values["signal_hit_rate_20d"].append(float(np.mean([e["label"] for e in recent_20])) if recent_20 else np.nan)
            values["signal_hit_rate_60d"].append(float(np.mean([e["label"] for e in recent_60])) if recent_60 else np.nan)
            values["signal_hit_rate_120d"].append(float(np.mean([e["label"] for e in recent_120])) if recent_120 else np.nan)
            values["signal_count_60d"].append(float(len(recent_60)))
            values["signal_avg_return_60d"].append(float(np.nanmean([e["side_return"] for e in recent_60])) if recent_60 else np.nan)

            long_60 = [e for e in recent_60 if int(e["primary_signal"]) == 1]
            short_60 = [e for e in recent_60 if int(e["primary_signal"]) == -1]
            values["signal_win_rate_long_60d"].append(float(np.mean([e["label"] for e in long_60])) if long_60 else np.nan)
            values["signal_win_rate_short_60d"].append(float(np.mean([e["label"] for e in short_60])) if short_60 else np.nan)
            values["signal_recent_fail_streak"].append(float(recent_streak(history, 0)))
            values["signal_recent_success_streak"].append(float(recent_streak(history, 1)))

            current_regime = getattr(row, "gmm_regime_id", np.nan)
            if np.isfinite(current_regime):
                regime_events = [
                    e for e in recent_60
                    if np.isfinite(e.get("gmm_regime_id", np.nan)) and int(e["gmm_regime_id"]) == int(current_regime)
                ]
                values["regime_conditioned_hit_rate_60d"].append(float(np.mean([e["label"] for e in regime_events])) if regime_events else np.nan)
            else:
                values["regime_conditioned_hit_rate_60d"].append(np.nan)

        for col, col_values in values.items():
            group[col] = col_values
        pieces.append(group)

    return pd.concat(pieces, ignore_index=True).sort_values(["date", "instrument"]).reset_index(drop=True)


def add_controlled_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = add_side_adjusted_features(frame)
    out = add_regime_interaction_features(out)
    out = add_signal_history_features(out)
    out["side_trend_tstat"] = (
        out["primary_signal"].astype(float)
        * out["trend_scan_sign"].fillna(0)
        * out["trend_scan_abs_tstat"]
    )
    return out


def build_feature_matrix(
    raw_data: dict[str, pd.DataFrame],
    config: CourseworkConfig,
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    """Build the lagged feature matrix for the official coursework pipeline.

    This step uses OHLCV and primary signals only. Signal-history features that
    depend on completed labels are added after labeling in
    `labeling.create_triple_barrier_labels`.
    """

    features = build_features(raw_data["ohlcv"], config.seed, use_hmm_extension=config.enable_hmm)
    feature_data = merge_price_and_signal_data(features, raw_data["signals_long"])
    feature_data = add_signal_features(feature_data)
    feature_data.attrs["ohlcv"] = raw_data["ohlcv"]
    feature_data.attrs["signals_long"] = raw_data["signals_long"]
    feature_data.attrs["engineered_features"] = features
    feature_columns = [
        col
        for col in feature_data.select_dtypes(include=[np.number]).columns
        if col not in {"label", "side_return", "holding_days"}
    ]
    return feature_data, sorted(set(feature_columns)), CONTROLLED_FEATURE_GROUPS
