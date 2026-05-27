"""Evaluation metrics and baseline comparisons.

This module computes ROC AUC, precision, recall, F1, threshold analysis, and
comparisons against blindly following every non-zero primary signal.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import ASSET_CLASS, CourseworkConfig, INSTRUMENTS


def metric_bundle(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba).clip(0, 1)
    pred = (proba >= threshold).astype(int)
    out: dict[str, float] = {
        "n": float(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)) if len(y_true) else np.nan,
        "precision": float(precision_score(y_true, pred, zero_division=0)) if len(y_true) else np.nan,
        "recall": float(recall_score(y_true, pred, zero_division=0)) if len(y_true) else np.nan,
        "f1": float(f1_score(y_true, pred, zero_division=0)) if len(y_true) else np.nan,
        "avg_precision": float(average_precision_score(y_true, proba)) if len(np.unique(y_true)) > 1 else np.nan,
        "brier": float(brier_score_loss(y_true, proba)) if len(y_true) else np.nan,
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1])) if len(y_true) else np.nan,
        "roc_auc": float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else np.nan,
    }
    if len(y_true):
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        out.update({"tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp)})
    else:
        out.update({"tn": np.nan, "fp": np.nan, "fn": np.nan, "tp": np.nan})
    return out


def selection_score(metrics: dict[str, float]) -> float:
    for key in ("roc_auc", "avg_precision", "f1", "accuracy"):
        value = metrics.get(key, np.nan)
        if np.isfinite(value):
            return float(value)
    value = metrics.get("log_loss", np.nan)
    return -float(value) if np.isfinite(value) else -np.inf


def optimize_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.25, 0.75, 51):
        score = f1_score(y_true, proba >= threshold, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def baseline_comparison_metrics(y_true: np.ndarray, side_returns: pd.Series) -> dict[str, float]:
    baseline_proba = np.ones_like(y_true, dtype=float)
    metrics = metric_bundle(y_true, baseline_proba, threshold=0.5)
    metrics["mean_side_return"] = float(side_returns.mean()) if len(side_returns) else np.nan
    metrics["total_side_return"] = float(side_returns.sum()) if len(side_returns) else np.nan
    return metrics


def evaluate_model(
    inst: str,
    model: InstrumentModel,
    inst_data: pd.DataFrame,
    prediction_start: pd.Timestamp,
    prediction_end: pd.Timestamp,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    test = inst_data[
        (inst_data["date"] >= prediction_start)
        & (inst_data["date"] <= prediction_end)
        & inst_data["label"].notna()
    ].copy()
    if len(test) == 0:
        return (
            {
                "instrument": inst,
                "selected_family": model.selected_family,
                "n_test": 0,
                "note": "No complete labelled test events in prediction window.",
            },
            [],
        )

    from .models import clean_feature_matrix, probability_of_positive

    x_test = clean_feature_matrix(test, model.feature_cols)
    y_test = test["label"].astype(int).to_numpy()
    proba = probability_of_positive(model.model, x_test)
    tuned = metric_bundle(y_test, proba, threshold=model.threshold)
    at_05 = metric_bundle(y_test, proba, threshold=0.5)

    baseline_proba = np.ones_like(y_test, dtype=float)
    baseline = metric_bundle(y_test, baseline_proba, threshold=0.5)
    meta_take = proba >= model.threshold

    summary = {
        "instrument": inst,
        "asset_class": ASSET_CLASS.get(inst, "unknown"),
        "selected_family": model.selected_family,
        "validation_score": model.validation_score,
        "selected_threshold": model.threshold,
        "n_train": model.n_train,
        "n_test": len(test),
        **{f"meta_{k}": v for k, v in tuned.items()},
        **{f"meta_05_{k}": v for k, v in at_05.items()},
        **{f"baseline_{k}": v for k, v in baseline.items()},
        "baseline_mean_side_return": float(test["side_return"].mean()),
        "meta_trade_rate": float(meta_take.mean()),
        "meta_mean_side_return_when_taken": float(test.loc[meta_take, "side_return"].mean()) if meta_take.any() else np.nan,
        "meta_total_side_return_when_taken": float(test.loc[meta_take, "side_return"].sum()) if meta_take.any() else 0.0,
    }

    threshold_rows = []
    for threshold in [0.30, 0.40, 0.50, 0.60, 0.70, model.threshold]:
        metrics = metric_bundle(y_test, proba, threshold=threshold)
        threshold_rows.append(
            {
                "instrument": inst,
                "threshold": threshold,
                "is_selected_threshold": int(abs(threshold - model.threshold) < 1e-12),
                **metrics,
            }
        )
    return summary, threshold_rows


def final_evaluation_summary(ablation_results: pd.DataFrame, final_experiment: str, final_family: str | None = None) -> pd.DataFrame:
    selected = ablation_results[
        (ablation_results["experiment"] == final_experiment)
    ].copy()
    if final_family is not None:
        selected = selected[selected["family"] == final_family].copy()
    else:
        selected = selected[selected["is_selected_family"] == 1].copy()
    if selected.empty:
        return selected
    rows = []
    for row in selected.itertuples(index=False):
        rows.append(
            {
                "instrument": row.instrument,
                "asset_class": row.asset_class,
                "selected_family": row.family,
                "final_experiment": row.experiment,
                "validation_score": row.validation_score,
                "selected_threshold": row.selected_threshold,
                "n_train": row.n_train,
                "n_validation": row.n_validation,
                "n_test": row.n_test,
                "meta_roc_auc": getattr(row, "oos_roc_auc", np.nan),
                "meta_precision": getattr(row, "oos_precision", np.nan),
                "meta_recall": getattr(row, "oos_recall", np.nan),
                "meta_f1": getattr(row, "oos_f1", np.nan),
                "meta_accuracy": getattr(row, "oos_accuracy", np.nan),
                "meta_tn": getattr(row, "oos_tn", np.nan),
                "meta_fp": getattr(row, "oos_fp", np.nan),
                "meta_fn": getattr(row, "oos_fn", np.nan),
                "meta_tp": getattr(row, "oos_tp", np.nan),
                "meta_05_roc_auc": getattr(row, "oos_05_roc_auc", np.nan),
                "meta_05_precision": getattr(row, "oos_05_precision", np.nan),
                "meta_05_recall": getattr(row, "oos_05_recall", np.nan),
                "meta_05_f1": getattr(row, "oos_05_f1", np.nan),
                "baseline_precision": getattr(row, "baseline_precision", np.nan),
                "baseline_recall": getattr(row, "baseline_recall", np.nan),
                "baseline_f1": getattr(row, "baseline_f1", np.nan),
                "baseline_roc_auc": getattr(row, "baseline_roc_auc", np.nan),
                "baseline_mean_side_return": getattr(row, "baseline_mean_side_return", np.nan),
                "meta_trade_rate": getattr(row, "meta_trade_rate", np.nan),
                "meta_mean_side_return_when_taken": getattr(row, "meta_mean_side_return_when_taken", np.nan),
                "meta_total_side_return_when_taken": getattr(row, "meta_total_side_return_when_taken", np.nan),
            }
        )
    return pd.DataFrame(rows)


def summarize_ablation_results(ablation_results: pd.DataFrame, specs: list[ExperimentSpec]) -> pd.DataFrame:
    selected = ablation_results[ablation_results["is_selected_family"] == 1].copy()
    if selected.empty:
        return pd.DataFrame()
    spec_complexity = {spec.name: spec.complexity for spec in specs}
    grouped = (
        selected.groupby("experiment")
        .agg(
            n_instruments=("instrument", "nunique"),
            mean_roc_auc=("oos_roc_auc", "mean"),
            median_roc_auc=("oos_roc_auc", "median"),
            mean_f1=("oos_f1", "mean"),
            median_f1=("oos_f1", "median"),
            mean_precision=("oos_precision", "mean"),
            mean_recall=("oos_recall", "mean"),
            mean_validation_score=("validation_score", "mean"),
            share_auc_above_0_5=("oos_roc_auc", lambda s: float((s > 0.5).mean())),
            mean_baseline_precision=("baseline_precision", "mean"),
        )
        .reset_index()
    )
    grouped["complexity"] = grouped["experiment"].map(spec_complexity).fillna(99).astype(int)
    return grouped.sort_values(["complexity", "experiment"]).reset_index(drop=True)


def summarize_model_results(ablation_results: pd.DataFrame, specs: list[ExperimentSpec]) -> pd.DataFrame:
    if ablation_results.empty:
        return pd.DataFrame()
    spec_complexity = {spec.name: spec.complexity for spec in specs}
    family_complexity = {
        "logistic": 0,
        "random_forest": 2,
        "extra_trees": 2,
        "hist_gradient_boosting": 3,
        "adaboost": 3,
        "mlp": 4,
    }
    grouped = (
        ablation_results.groupby(["experiment", "family"])
        .agg(
            n_instruments=("instrument", "nunique"),
            mean_roc_auc=("oos_roc_auc", "mean"),
            median_roc_auc=("oos_roc_auc", "median"),
            mean_f1=("oos_f1", "mean"),
            median_f1=("oos_f1", "median"),
            mean_precision=("oos_precision", "mean"),
            mean_recall=("oos_recall", "mean"),
            mean_validation_score=("validation_score", "mean"),
            share_auc_above_0_5=("oos_roc_auc", lambda s: float((s > 0.5).mean())),
            mean_baseline_precision=("baseline_precision", "mean"),
        )
        .reset_index()
    )
    grouped["feature_complexity"] = grouped["experiment"].map(spec_complexity).fillna(99).astype(int)
    grouped["family_complexity"] = grouped["family"].map(family_complexity).fillna(5).astype(int)
    grouped["complexity"] = grouped["feature_complexity"] + grouped["family_complexity"]
    return grouped.sort_values(["feature_complexity", "family_complexity", "experiment", "family"]).reset_index(drop=True)


def choose_final_experiment(ablation_summary: pd.DataFrame, use_hmm_extension: bool) -> str:
    if ablation_summary.empty:
        return "baseline"
    eligible = ablation_summary.copy()
    if not use_hmm_extension:
        eligible = eligible[eligible["experiment"] != "+ hmm_extension"].copy()
    if eligible.empty:
        return "baseline"
    max_auc = eligible["mean_roc_auc"].dropna().max()
    max_f1 = eligible["mean_f1"].dropna().max()
    if not np.isfinite(max_auc):
        max_auc = -np.inf
    if not np.isfinite(max_f1):
        max_f1 = -np.inf
    stable = eligible[
        (eligible["mean_roc_auc"].fillna(-np.inf) >= max_auc - 0.01)
        & (eligible["mean_f1"].fillna(-np.inf) >= max_f1 - 0.02)
    ].copy()
    if stable.empty:
        stable = eligible.copy()
    stable["selection_rank_score"] = (
        stable["mean_roc_auc"].fillna(0)
        + 0.20 * stable["mean_f1"].fillna(0)
        + 0.05 * stable["share_auc_above_0_5"].fillna(0)
        - 0.005 * stable["complexity"]
    )
    stable = stable.sort_values(["complexity", "selection_rank_score"], ascending=[True, False])
    return str(stable.iloc[0]["experiment"])


def choose_final_model_pair(model_summary: pd.DataFrame, use_hmm_extension: bool) -> tuple[str, str]:
    if model_summary.empty:
        return "baseline", "logistic"
    eligible = model_summary.copy()
    if not use_hmm_extension:
        eligible = eligible[eligible["experiment"] != "+ hmm_extension"].copy()
    eligible = eligible[eligible["n_instruments"] >= max(1, len(INSTRUMENTS) - 1)].copy()
    if eligible.empty:
        eligible = model_summary.copy()

    max_auc = eligible["mean_roc_auc"].dropna().max()
    max_f1 = eligible["mean_f1"].dropna().max()
    if not np.isfinite(max_auc):
        max_auc = -np.inf
    if not np.isfinite(max_f1):
        max_f1 = -np.inf
    stable = eligible[
        (eligible["mean_roc_auc"].fillna(-np.inf) >= max_auc - 0.01)
        & (eligible["mean_f1"].fillna(-np.inf) >= max_f1 - 0.03)
    ].copy()
    if stable.empty:
        stable = eligible.copy()
    stable["selection_rank_score"] = (
        stable["mean_roc_auc"].fillna(0)
        + 0.20 * stable["mean_f1"].fillna(0)
        + 0.05 * stable["share_auc_above_0_5"].fillna(0)
        - 0.008 * stable["complexity"]
    )
    stable = stable.sort_values(
        ["complexity", "selection_rank_score", "mean_roc_auc"],
        ascending=[True, False, False],
    )
    selected = stable.iloc[0]
    return str(selected["experiment"]), str(selected["family"])


def evaluate_final_model(
    labeled_data: pd.DataFrame,
    final_predictions: pd.DataFrame,
    config: CourseworkConfig,
) -> pd.DataFrame:
    """Load the clean OOS evaluation for the frozen final model."""

    if config.evaluation_summary_path.exists():
        return pd.read_csv(config.evaluation_summary_path)
    return pd.DataFrame()


def compare_to_primary_signal_baseline(
    labeled_data: pd.DataFrame,
    final_predictions: pd.DataFrame,
    config: CourseworkConfig,
) -> pd.DataFrame:
    """Extract the blindly-follow-primary-signal baseline comparison.

    This is a reporting transformation of `evaluation_summary.csv`, not a new
    model experiment.
    """

    evaluation = evaluate_final_model(labeled_data, final_predictions, config)
    if evaluation.empty:
        return evaluation
    columns = [
        "instrument",
        "asset_class",
        "n_test",
        "baseline_precision",
        "baseline_recall",
        "baseline_f1",
        "baseline_roc_auc",
        "baseline_mean_side_return",
        "meta_roc_auc",
        "meta_precision",
        "meta_recall",
        "meta_f1",
    ]
    return evaluation[[col for col in columns if col in evaluation.columns]].copy()
