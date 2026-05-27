"""Cluster-level feature importance tools.

This module groups correlated or economically related features and computes
permutation-style importance summaries for coursework interpretation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import CourseworkConfig, HMM_EXTENSION_FEATURES, SIDE_ADJUSTED_FEATURES, SIGNAL_HISTORY_FEATURES, TREND_SCANNING_FEATURES, VOL_STRESS_FEATURES
from .evaluation import metric_bundle
from .models import InstrumentModel, chronological_split, clean_feature_matrix, probability_of_positive


def feature_family(feature: str) -> str:
    if feature.startswith("signal") or feature in {"primary_signal", "prev_signal"}:
        return "primary_signal_context"
    if feature.startswith("gmm"):
        return "latent_gmm_regime"
    if feature.startswith("hmm"):
        return "latent_hmm_regime"
    if feature.startswith("pca"):
        return "latent_pca"
    if feature.startswith("cs_") or feature.startswith("asset_class"):
        return "cross_sectional"
    if any(key in feature for key in ["vol", "atr", "range", "bollinger"]):
        return "volatility_range"
    if any(key in feature for key in ["mom", "ma_gap", "price_z", "macd", "rsi", "stoch", "ret"]):
        return "trend_momentum"
    if "volume" in feature or "dollar" in feature or "amihud" in feature:
        return "volume_liquidity"
    if "open_interest" in feature or feature == "volume_oi_ratio":
        return "positioning_open_interest"
    return "other"


def correlated_clusters(x: pd.DataFrame, feature_cols: list[str], threshold: float = 0.75) -> list[list[str]]:
    corr = x[feature_cols].replace([np.inf, -np.inf], np.nan).corr().abs().fillna(0)
    unused = set(feature_cols)
    clusters: list[list[str]] = []
    while unused:
        start = unused.pop()
        stack = [start]
        cluster = {start}
        while stack:
            current = stack.pop()
            neighbours = set(corr.index[(corr.loc[current] >= threshold) & (corr.index != current)])
            new_nodes = neighbours & unused
            unused -= new_nodes
            cluster |= new_nodes
            stack.extend(new_nodes)
        clusters.append(sorted(cluster))
    return sorted(clusters, key=lambda c: (-len(c), c[0]))


def cluster_importance_rows(
    inst: str,
    model: InstrumentModel,
    inst_data: pd.DataFrame,
    prediction_start: pd.Timestamp,
) -> list[dict[str, Any]]:
    if model.rf_for_importance is None:
        return []
    _, _, final_train = chronological_split(inst_data, prediction_start)
    if len(final_train) < 10:
        return []
    x_train = clean_feature_matrix(final_train, model.feature_cols)
    try:
        importances = model.rf_for_importance.named_steps["model"].feature_importances_
    except Exception:
        return []
    imp_map = dict(zip(model.feature_cols, importances))
    clusters = correlated_clusters(x_train, model.feature_cols)
    rows = []
    for i, cluster in enumerate(clusters, start=1):
        importance = float(sum(imp_map.get(f, 0.0) for f in cluster))
        families = pd.Series([feature_family(f) for f in cluster]).value_counts()
        rows.append(
            {
                "instrument": inst,
                "cluster_id": i,
                "dominant_family": families.index[0] if len(families) else "unknown",
                "n_features": len(cluster),
                "importance_sum": importance,
                "features": "; ".join(cluster),
            }
        )
    rows.sort(key=lambda r: r["importance_sum"], reverse=True)
    return rows


def cluster_name_for_feature(feature: str) -> str:
    if feature in SIDE_ADJUSTED_FEATURES:
        return "side_adjusted"
    if feature in SIGNAL_HISTORY_FEATURES or feature == "regime_conditioned_hit_rate_60d":
        return "signal_history"
    if feature in VOL_STRESS_FEATURES or feature.startswith("drawdown_"):
        return "volatility_stress"
    if feature in TREND_SCANNING_FEATURES:
        return "trend_scanning"
    if feature in HMM_EXTENSION_FEATURES or feature.startswith("hmm_"):
        return "hmm_extension"
    if feature.startswith("gmm") or feature.startswith("gmm_state"):
        return "gmm_regime"
    if feature.startswith("pca_"):
        return "pca_latent"
    if feature.startswith("cs_") or feature.startswith("asset_class"):
        return "cross_sectional"
    if "open_interest" in feature or feature == "volume_oi_ratio":
        return "open_interest"
    if "volume" in feature or "dollar" in feature or "amihud" in feature:
        return "volume_liquidity"
    if any(key in feature for key in ["rsi", "macd", "bollinger", "stoch"]):
        return "technical_indicators"
    if any(key in feature for key in ["ret", "mom", "ma_gap", "price_z"]):
        return "return_momentum"
    if any(key in feature for key in ["vol", "atr", "range"]):
        return "volatility_stress"
    return "technical_indicators"


def cluster_permutation_importance(
    full_frame: pd.DataFrame,
    models: dict[str, InstrumentModel],
    prediction_start: pd.Timestamp,
    prediction_end: pd.Timestamp,
    seed: int,
    repeats: int = 3,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for inst, model in models.items():
        test = full_frame[
            (full_frame["instrument"] == inst)
            & (full_frame["date"] >= prediction_start)
            & (full_frame["date"] <= prediction_end)
            & full_frame["label"].notna()
        ].copy()
        if len(test) < 5:
            continue
        x_test = clean_feature_matrix(test, model.feature_cols)
        y_test = test["label"].astype(int).to_numpy()
        baseline_proba = probability_of_positive(model.model, x_test)
        baseline_metrics = metric_bundle(y_test, baseline_proba, threshold=model.threshold)

        cluster_map: dict[str, list[str]] = {}
        for feature in model.feature_cols:
            cluster_map.setdefault(cluster_name_for_feature(feature), []).append(feature)

        for cluster, features in sorted(cluster_map.items()):
            auc_values = []
            f1_values = []
            for _ in range(repeats):
                if len(x_test) <= 1:
                    continue
                permuted = x_test.copy()
                order = rng.permutation(len(permuted))
                permuted.loc[:, features] = permuted.loc[:, features].iloc[order].to_numpy()
                permuted_proba = probability_of_positive(model.model, permuted)
                permuted_metrics = metric_bundle(y_test, permuted_proba, threshold=model.threshold)
                auc_values.append(permuted_metrics["roc_auc"])
                f1_values.append(permuted_metrics["f1"])
            perm_auc = float(np.nanmean(auc_values)) if auc_values else np.nan
            perm_f1 = float(np.nanmean(f1_values)) if f1_values else np.nan
            rows.append(
                {
                    "experiment": model.experiment_name,
                    "instrument": inst,
                    "cluster": cluster,
                    "n_features": len(features),
                    "features": "; ".join(features),
                    "baseline_roc_auc": baseline_metrics["roc_auc"],
                    "permuted_roc_auc": perm_auc,
                    "roc_auc_drop": baseline_metrics["roc_auc"] - perm_auc if np.isfinite(baseline_metrics["roc_auc"]) and np.isfinite(perm_auc) else np.nan,
                    "baseline_f1": baseline_metrics["f1"],
                    "permuted_f1": perm_f1,
                    "f1_drop": baseline_metrics["f1"] - perm_f1 if np.isfinite(baseline_metrics["f1"]) and np.isfinite(perm_f1) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def compute_cluster_level_importance(
    final_model: dict[str, Any],
    labeled_data: pd.DataFrame,
    feature_columns: list[str],
    feature_groups: dict[str, list[str]],
    config: CourseworkConfig,
) -> pd.DataFrame:
    """Load cluster-level feature importance for the final submission."""

    if config.cluster_importance_path.exists():
        return pd.read_csv(config.cluster_importance_path)
    return pd.DataFrame()
