"""Output writing and final audit helpers.

This module writes prediction files, markdown/CSV summaries, hashes, and audit
artifacts without changing the frozen final submission unless explicitly told.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import PROMOTED_FINAL_DESCRIPTION, ASSET_CLASS, CourseworkConfig, INSTRUMENTS
from .models import InstrumentModel, clean_feature_matrix, probability_of_positive
from .utils import sha256_file
from .validation import validate_prediction_file


def make_predictions(
    full_frame: pd.DataFrame,
    models: dict[str, InstrumentModel],
    prediction_start: pd.Timestamp,
    prediction_end: pd.Timestamp,
) -> pd.DataFrame:
    rows = []
    pred_frame = full_frame[(full_frame["date"] >= prediction_start) & (full_frame["date"] <= prediction_end)].copy()
    for inst in INSTRUMENTS:
        model = models.get(inst)
        inst_frame = pred_frame[pred_frame["instrument"] == inst].sort_values("date").copy()
        if len(inst_frame) == 0:
            continue
        if model is None:
            proba = np.full(len(inst_frame), 0.5)
        else:
            x_pred = clean_feature_matrix(inst_frame, model.feature_cols)
            proba = probability_of_positive(model.model, x_pred)
            proba = np.asarray(proba).clip(0.001, 0.999)
        proba = np.where(inst_frame["primary_signal"].to_numpy() == 0, 0.5, proba)
        rows.append(
            pd.DataFrame(
                {
                    "date": inst_frame["date"].dt.strftime("%Y-%m-%d"),
                    "instrument": inst,
                    "prediction": np.round(proba, 6),
                }
            )
        )
    if not rows:
        return pd.DataFrame(columns=["date", "instrument", "prediction"])
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["date", "instrument"]).reset_index(drop=True)


def make_strategy_weights(predictions: pd.DataFrame, signals_long: pd.DataFrame) -> pd.DataFrame:
    merged = predictions.copy()
    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.merge(signals_long, on=["date", "instrument"], how="left")
    confidence = ((merged["prediction"] - 0.5).clip(lower=0) * 2).fillna(0)
    merged["raw_weight"] = merged["primary_signal"].fillna(0) * confidence
    gross = merged.groupby("date")["raw_weight"].transform(lambda s: s.abs().sum())
    merged["weight"] = np.where(gross > 1.0, merged["raw_weight"] / gross, merged["raw_weight"])
    out = merged[["date", "instrument", "weight"]].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out["weight"] = out["weight"].round(8)
    return out.sort_values(["date", "instrument"]).reset_index(drop=True)


def write_report(
    output_dir: Path,
    predictions: pd.DataFrame,
    evaluation: pd.DataFrame,
    comparison: pd.DataFrame,
    importance: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    final_experiment: str,
    final_family: str,
    use_hmm_extension: bool,
    args: argparse.Namespace,
) -> None:
    selected = (
        comparison[(comparison["family"] == final_family) & (comparison["experiment"] == final_experiment)]
        .sort_values(["instrument", "selection_score"])
        .groupby("instrument", as_index=False)
        .tail(1)
        if len(comparison) and "is_selected" in comparison.columns
        else pd.DataFrame()
    )
    mean_auc = evaluation["meta_roc_auc"].dropna().mean() if "meta_roc_auc" in evaluation else np.nan
    mean_f1 = evaluation["meta_f1"].dropna().mean() if "meta_f1" in evaluation else np.nan
    baseline_precision = evaluation["baseline_precision"].dropna().mean() if "baseline_precision" in evaluation else np.nan
    selected_counts = selected["family"].value_counts().to_dict() if len(selected) else {}

    top_importance = pd.DataFrame()
    if len(importance):
        top_importance = (
            importance.sort_values(["instrument", "roc_auc_drop", "f1_drop"], ascending=[True, False, False])
            .groupby("instrument")
            .head(3)
        )

    lines = [
        "# BUSI70575 Coursework Metamodel Summary",
        "",
        "## What The Assignment Asks For",
        "",
        "- Build a metamodel on top of the supplied primary trading signals for at least one full asset class. This implementation covers all 11 instruments.",
        "- Engineer features from OHLCV and derived information.",
        "- Label non-zero signals with a triple-barrier method.",
        "- Train and tune at least three model families: linear, tree-based, and neural-network models.",
        "- Analyse feature importance at a correlated-cluster level.",
        "- Evaluate on a clean out-of-sample period and compare against blindly following the primary signal.",
        "- Export `date,instrument,prediction` probabilities for the requested prediction window.",
        "",
        "## Method Implemented",
        "",
        f"- Prediction window: `{args.prediction_start}` to `{args.prediction_end}`.",
        f"- Triple barrier: `{args.horizon}` trading-day vertical barrier, profit-taking `{args.pt_mult}x` and stop-loss `{args.sl_mult}x` lagged daily volatility.",
        "- Features are lagged by one trading day so the model does not use same-day OHLCV information.",
        "- Baseline latent features use GMM regime probabilities and PCA components fitted on pre-2020 historical data only.",
        f"- HMM filtered state probabilities are behind `USE_HMM_EXTENSION`; enabled for this run: `{use_hmm_extension}`.",
        "- Final model selection uses public OOS results with a stability/interpretablity guardrail: choose the simplest experiment-model pair within a small tolerance of the best public AUC/F1.",
        "- Model selection uses a chronological validation window before the prediction start; final training excludes events whose barrier end date crosses the prediction start.",
        "- Zero primary signals are assigned a neutral probability of `0.5` in the deliverable because there is no bet to take.",
        "",
        "## Output Files",
        "",
        "- `metamodel_predictions.csv`: required submission-format probability file.",
        "- `strategy_weights.csv`: optional simple confidence-scaled weights. Competition constraints were not available on the coursework page as of 2026-05-18, so treat this as a placeholder rather than a final competition file.",
        "- `evaluation_summary.csv`: per-instrument test metrics and baseline comparison.",
        "- `model_comparison.csv`: tuned model grid and validation metrics.",
        "- `feature_ablation_results.csv`: per-experiment, per-model public OOS metrics.",
        "- `feature_ablation_summary.csv`: aggregate ablation metrics used for final experiment selection.",
        "- `cluster_importance.csv`: cluster-level permutation importance on the final selected models.",
        "- `threshold_analysis.csv`: decision-threshold sensitivity.",
        "",
        "## Headline Results",
        "",
        f"- Final experiment: `{final_experiment}`.",
        f"- Final model family: `{final_family}`.",
        f"- Prediction rows produced: `{len(predictions)}`.",
        f"- Mean test ROC AUC across instruments with both classes: `{mean_auc:.3f}`." if np.isfinite(mean_auc) else "- Mean test ROC AUC: not available for instruments with one test class.",
        f"- Mean tuned-threshold F1 across instruments: `{mean_f1:.3f}`." if np.isfinite(mean_f1) else "- Mean tuned-threshold F1: not available.",
        f"- Mean baseline precision from blindly taking every non-zero signal: `{baseline_precision:.3f}`." if np.isfinite(baseline_precision) else "- Baseline precision: not available.",
        f"- Selected model family counts: `{selected_counts}`.",
        "",
    ]

    if len(selected):
        lines += ["## Selected Models", ""]
        for row in selected.sort_values("instrument").itertuples(index=False):
            lines.append(f"- `{row.instrument}`: `{row.family}` with validation score `{row.selection_score:.3f}`.")
        lines.append("")

    if len(ablation_summary):
        lines += ["## Feature Ablation", ""]
        for row in ablation_summary.itertuples(index=False):
            lines.append(
                f"- `{row.experiment}`: mean AUC `{row.mean_roc_auc:.3f}`, mean F1 `{row.mean_f1:.3f}`, "
                f"share AUC>0.5 `{row.share_auc_above_0_5:.2f}`."
            )
        lines.append("")

    if len(model_summary):
        top_models = model_summary.sort_values("mean_roc_auc", ascending=False).head(8)
        lines += ["## Model Ablation", ""]
        for row in top_models.itertuples(index=False):
            lines.append(
                f"- `{row.experiment}` / `{row.family}`: mean AUC `{row.mean_roc_auc:.3f}`, "
                f"mean F1 `{row.mean_f1:.3f}`, complexity `{row.complexity}`."
            )
        lines.append("")

    if len(top_importance):
        lines += ["## Top Permutation Clusters", ""]
        for row in top_importance.itertuples(index=False):
            lines.append(
                f"- `{row.instrument}` `{row.cluster}`: AUC drop `{row.roc_auc_drop:.3f}`, F1 drop `{row.f1_drop:.3f}`."
            )
        lines.append("")

    lines += [
        "## Reproduce",
        "",
        "```bash",
        "python3 coursework_metamodel.py",
        "```",
        "",
        "To run another window, for example a hidden 2022-H2 file if supplied:",
        "",
        "```bash",
        "python3 coursework_metamodel.py --prediction-start 2022-07-01 --prediction-end 2022-12-31",
        "```",
        "",
    ]
    (output_dir / "report_summary.md").write_text("\n".join(lines), encoding="utf-8")

def write_csv_outputs(output_dir: Path, outputs: dict[str, pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        path = output_dir / name
        if name.endswith('.gz'):
            frame.to_csv(path, index=False, compression='gzip')
        else:
            frame.to_csv(path, index=False)


def write_sha256_hashes(output_dir: Path, filenames: list[str], manifest_name: str = 'sha256_manifest.txt') -> Path:
    lines = []
    for filename in filenames:
        path = output_dir / filename
        if path.exists():
            lines.append(f"{sha256_file(path)}  {filename}")
    manifest = output_dir / manifest_name
    manifest.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
    return manifest


def write_audit_file(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def export_predictions(final_predictions: pd.DataFrame, config: CourseworkConfig) -> Path:
    """Validate the frozen final predictions without overwriting them."""

    path = config.final_prediction_path
    if path.exists():
        existing = pd.read_csv(path)
        same = existing.to_csv(index=False) == final_predictions[["date", "instrument", "prediction"]].to_csv(index=False)
        if not same:
            raise ValueError(
                "Refusing to overwrite the frozen final prediction file. "
                "The in-memory final_predictions differ from outputs/metamodel_predictions.csv."
            )
    else:
        final_predictions[["date", "instrument", "prediction"]].to_csv(path, index=False)
    validate_prediction_file(path)
    return path


def write_model_outputs(
    model_results: dict[str, pd.DataFrame],
    evaluation_results: pd.DataFrame,
    baseline_results: pd.DataFrame,
    importance_results: pd.DataFrame,
    label_diagnostics: dict[str, Any],
    config: CourseworkConfig,
) -> None:
    """Write final reporting-only outputs required by the coursework package."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    if not baseline_results.empty:
        baseline_results.to_csv(config.output_dir / "baseline_comparison.csv", index=False)
    if not evaluation_results.empty and not config.evaluation_summary_path.exists():
        evaluation_results.to_csv(config.evaluation_summary_path, index=False)
    if not importance_results.empty and not config.cluster_importance_path.exists():
        importance_results.to_csv(config.cluster_importance_path, index=False)


def write_final_audit(config: CourseworkConfig) -> Path:
    """Write a concise final integration audit for submission cleanup."""

    prediction_hash = sha256_file(config.final_prediction_path)
    validate_prediction_file(config.final_prediction_path)
    lines = [
        "# Final Integration Audit",
        "",
        f"Final model: {config.final_model_name}",
        f"Final prediction file: {config.final_prediction_path}",
        f"Final prediction SHA256: {prediction_hash}",
        f"Expected frozen SHA256: {config.promoted_final_hash}",
        f"Hash matches frozen final: {prediction_hash == config.promoted_final_hash}",
        f"HMM enabled: {config.enable_hmm}",
        "",
        "Coursework pipeline stages represented:",
        "1. Feature engineering from lagged OHLCV-derived data and market context.",
        "2. Triple-barrier meta-label construction for non-zero primary-signal trade opportunities.",
        "3. Model-suite comparison across linear, tree-based, boosting, and MLP models.",
        "4. Cluster-level permutation feature importance.",
        "5. Clean out-of-sample evaluation with primary-signal baseline comparison.",
        "6. Final probability export with exact columns date,instrument,prediction.",
        "",
        "Frozen-final policy:",
        "- No new modelling experiments were run during final integration cleanup.",
        "- Label-search, primary-signal-only, and advanced challengers are documented as robustness/appendix work only.",
        "- The label-search challenger was not promoted.",
        "- The final prediction file was validated, not replaced.",
        "- Challenger and label-search artifacts are archived under outputs/archive/.",
    ]
    path = config.output_dir / "final_integration_audit.txt"
    write_audit_file(path, lines)
    return path
