"""Appendix robustness checks for universe and feature pruning.

This module evaluates diagnostic robustness variants without promoting them or
overwriting the frozen final prediction file.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    ASSET_CLASS,
    CONFIG,
    INSTRUMENTS,
    PROMOTED_FINAL_MEAN_AUC,
    PROMOTED_FINAL_MEAN_F1,
    PROMOTED_FINAL_HASH,
)
from .evaluation import metric_bundle, optimize_threshold, selection_score
from .importance import cluster_name_for_feature
from .models import (
    ExperimentSpec,
    build_estimator,
    candidate_grid,
    chronological_split,
    clean_feature_matrix,
    feature_columns_for_experiment,
    probability_of_positive,
)
from .utils import sha256_file


PREDICTION_START = pd.Timestamp(CONFIG.prediction_start)
PREDICTION_END = pd.Timestamp(CONFIG.prediction_end)
NEAR_ZERO_IMPORTANCE_CUTOFF = 0.002
WORST_INSTRUMENT_MATERIAL_DROP = 0.03
VALIDATION_SIMILAR_AUC_TOL = 0.002
VALIDATION_SIMILAR_F1_TOL = 0.005


@dataclass(frozen=True)
class BlendFrames:
    name: str
    logistic_features: list[str]
    mlp_features: list[str]
    valid: pd.DataFrame
    public: pd.DataFrame


def modelling_dataset_path() -> Path:
    active_path = CONFIG.output_dir / "modelling_dataset.csv.gz"
    archived_path = CONFIG.output_dir / "archive" / "debug_or_temporary" / "modelling_dataset.csv.gz"
    if active_path.exists():
        return active_path
    if archived_path.exists():
        return archived_path
    raise FileNotFoundError(
        "Could not find modelling_dataset.csv.gz in outputs/ or outputs/archive/debug_or_temporary/."
    )


def load_modelling_dataset() -> pd.DataFrame:
    full = pd.read_csv(modelling_dataset_path())
    full["date"] = pd.to_datetime(full["date"])
    full["label_end_date"] = pd.to_datetime(full["label_end_date"])
    return full


def feature_columns(full: pd.DataFrame, name: str, groups: tuple[str, ...]) -> list[str]:
    return feature_columns_for_experiment(
        full,
        ExperimentSpec(name=name, groups=groups, complexity=len(groups)),
        use_hmm_extension=False,
    )


def fit_sigmoid_calibrator(valid_frame: pd.DataFrame) -> Pipeline | None:
    if valid_frame.empty or valid_frame["label"].nunique() < 2:
        return None
    calibrator = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=CONFIG.seed),
            ),
        ]
    )
    calibrator.fit(valid_frame[["probability"]].to_numpy(), valid_frame["label"].astype(int).to_numpy())
    return calibrator


def apply_sigmoid_calibrator(calibrator: Pipeline | None, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if calibrator is not None and not out.empty:
        out["probability"] = calibrator.predict_proba(out[["probability"]].to_numpy())[:, 1]
    out["probability"] = out["probability"].clip(0.001, 0.999)
    return out


def best_params_for_family(
    family: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
) -> tuple[dict[str, Any], float, float]:
    best: dict[str, Any] | None = None
    for params in candidate_grid(CONFIG.seed).get(family, []):
        estimator = build_estimator(family, params, CONFIG.seed)
        try:
            estimator.fit(x_train, y_train)
            valid_proba = probability_of_positive(estimator, x_valid)
            threshold = optimize_threshold(y_valid, valid_proba)
            metrics = metric_bundle(y_valid, valid_proba, threshold=0.5)
            score = selection_score(metrics)
            if best is None or score > best["score"]:
                best = {"params": params, "threshold": threshold, "score": score}
        except Exception:
            continue
    if best is None:
        raise RuntimeError(f"No valid model for family={family}")
    return best["params"], float(best["threshold"]), float(best["score"])


def fit_family_frames(
    full: pd.DataFrame,
    instruments: list[str],
    feature_cols: list[str],
    family: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_rows: list[pd.DataFrame] = []
    public_rows: list[pd.DataFrame] = []

    for inst in instruments:
        inst_data = full[full["instrument"] == inst].sort_values("date").copy()
        train, validation, final_train = chronological_split(inst_data, PREDICTION_START)
        if len(final_train) < 10 or final_train["label"].nunique() < 2:
            continue
        if train["label"].nunique() < 2 or validation["label"].nunique() < 2:
            split_point = max(1, int(len(final_train) * 0.75))
            train = final_train.iloc[:split_point].copy()
            validation = final_train.iloc[split_point:].copy()
        if train["label"].nunique() < 2 or validation["label"].nunique() < 2:
            continue

        x_train = clean_feature_matrix(train, feature_cols)
        y_train = train["label"].astype(int).to_numpy()
        x_valid = clean_feature_matrix(validation, feature_cols)
        y_valid = validation["label"].astype(int).to_numpy()
        params, _, _ = best_params_for_family(family, x_train, y_train, x_valid, y_valid)

        validation_estimator = build_estimator(family, params, CONFIG.seed)
        validation_estimator.fit(x_train, y_train)
        valid_proba = probability_of_positive(validation_estimator, x_valid)
        valid_rows.append(
            pd.DataFrame(
                {
                    "date": validation["date"].to_numpy(),
                    "instrument": inst,
                    "asset_class": ASSET_CLASS.get(inst, "unknown"),
                    "label": y_valid,
                    "side_return": validation["side_return"].to_numpy(),
                    "primary_signal": validation["primary_signal"].to_numpy(),
                    "probability": valid_proba,
                }
            )
        )

        final_estimator = build_estimator(family, params, CONFIG.seed)
        x_final = clean_feature_matrix(final_train, feature_cols)
        y_final = final_train["label"].astype(int).to_numpy()
        final_estimator.fit(x_final, y_final)

        public = inst_data[
            (inst_data["date"] >= PREDICTION_START)
            & (inst_data["date"] <= PREDICTION_END)
            & inst_data["label"].notna()
        ].copy()
        if not public.empty:
            public_proba = probability_of_positive(
                final_estimator,
                clean_feature_matrix(public, feature_cols),
            )
            public_rows.append(
                pd.DataFrame(
                    {
                        "date": public["date"].to_numpy(),
                        "instrument": inst,
                        "asset_class": ASSET_CLASS.get(inst, "unknown"),
                        "label": public["label"].astype(int).to_numpy(),
                        "side_return": public["side_return"].to_numpy(),
                        "primary_signal": public["primary_signal"].to_numpy(),
                        "probability": public_proba,
                    }
                )
            )

    valid = pd.concat(valid_rows, ignore_index=True) if valid_rows else empty_probability_frame()
    public = pd.concat(public_rows, ignore_index=True) if public_rows else empty_probability_frame()
    return valid, public


def empty_probability_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "instrument",
            "asset_class",
            "label",
            "side_return",
            "primary_signal",
            "probability",
        ]
    )


def merge_probability_frames(left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str) -> pd.DataFrame:
    if left.empty or right.empty:
        return empty_probability_frame()
    merged = left.rename(columns={"probability": left_name}).merge(
        right.rename(columns={"probability": right_name}),
        on=["date", "instrument", "asset_class", "label", "side_return", "primary_signal"],
        how="inner",
    )
    out = merged[["date", "instrument", "asset_class", "label", "side_return", "primary_signal"]].copy()
    out["probability"] = 0.5 * merged[left_name] + 0.5 * merged[right_name]
    return out


def fit_calibrated_blend(
    full: pd.DataFrame,
    instruments: list[str],
    logistic_cols: list[str],
    mlp_cols: list[str],
    name: str,
) -> BlendFrames:
    logistic_valid, logistic_public = fit_family_frames(full, instruments, logistic_cols, "logistic")
    mlp_valid, mlp_public = fit_family_frames(full, instruments, mlp_cols, "mlp")
    valid = merge_probability_frames(logistic_valid, mlp_valid, "logistic_prob", "mlp_prob")
    public = merge_probability_frames(logistic_public, mlp_public, "logistic_prob", "mlp_prob")
    calibrator = fit_sigmoid_calibrator(valid)
    valid = apply_sigmoid_calibrator(calibrator, valid)
    public = apply_sigmoid_calibrator(calibrator, public)
    return BlendFrames(name=name, logistic_features=logistic_cols, mlp_features=mlp_cols, valid=valid, public=public)


def thresholds_from_validation(valid_frame: pd.DataFrame, instruments: list[str]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for inst in instruments:
        group = valid_frame[valid_frame["instrument"] == inst]
        if group.empty:
            thresholds[inst] = 0.5
            continue
        y = group["label"].astype(int).to_numpy()
        proba = group["probability"].to_numpy()
        thresholds[inst] = optimize_threshold(y, proba)
    return thresholds


def instrument_metrics(
    valid_frame: pd.DataFrame,
    public_frame: pd.DataFrame,
    instruments: list[str],
    experiment: str,
) -> pd.DataFrame:
    thresholds = thresholds_from_validation(valid_frame, instruments)
    rows: list[dict[str, Any]] = []
    for inst in instruments:
        valid_group = valid_frame[valid_frame["instrument"] == inst]
        public_group = public_frame[public_frame["instrument"] == inst]
        threshold = thresholds.get(inst, 0.5)

        if valid_group.empty:
            valid_metrics: dict[str, float] = {"n": 0.0, "roc_auc": np.nan, "f1": np.nan}
        else:
            valid_metrics = metric_bundle(
                valid_group["label"].astype(int).to_numpy(),
                valid_group["probability"].to_numpy(),
                threshold=threshold,
            )
        if public_group.empty:
            public_metrics = {"n": 0.0, "roc_auc": np.nan, "f1": np.nan}
        else:
            public_metrics = metric_bundle(
                public_group["label"].astype(int).to_numpy(),
                public_group["probability"].to_numpy(),
                threshold=threshold,
            )

        rows.append(
            {
                "experiment": experiment,
                "instrument": inst,
                "asset_class": ASSET_CLASS.get(inst, "unknown"),
                "validation_threshold": threshold,
                **{f"validation_{k}": v for k, v in valid_metrics.items()},
                **{f"public_{k}": v for k, v in public_metrics.items()},
                "public_auc_minus_validation_auc": public_metrics.get("roc_auc", np.nan)
                - valid_metrics.get("roc_auc", np.nan),
            }
        )
    return pd.DataFrame(rows)


def complete_asset_class_retained(instruments: list[str]) -> bool:
    selected = set(instruments)
    for asset_class in sorted(set(ASSET_CLASS.values())):
        class_instruments = {inst for inst, cls in ASSET_CLASS.items() if cls == asset_class}
        if class_instruments and class_instruments.issubset(selected):
            return True
    return False


def aggregate_from_per_instrument(per_inst: pd.DataFrame, experiment: str, instruments: list[str]) -> dict[str, Any]:
    public_auc = per_inst["public_roc_auc"].dropna()
    valid_auc = per_inst["validation_roc_auc"].dropna()
    worst_public = per_inst.sort_values("public_roc_auc", na_position="last").head(3)
    worst_valid = per_inst.sort_values("validation_roc_auc", na_position="last").head(3)
    retained_classes = sorted({ASSET_CLASS.get(inst, "unknown") for inst in instruments})
    return {
        "experiment": experiment,
        "n_instruments": len(instruments),
        "instruments": ",".join(instruments),
        "asset_classes_retained": ",".join(retained_classes),
        "has_complete_asset_class": complete_asset_class_retained(instruments),
        "direct_output_format_full_assignment_valid": set(instruments) == set(INSTRUMENTS),
        "output_format_valid_if_dropped_instruments_filled_neutral": True,
        "validation_mean_roc_auc": float(valid_auc.mean()) if len(valid_auc) else np.nan,
        "validation_mean_f1": float(per_inst["validation_f1"].mean()),
        "public_mean_roc_auc": float(public_auc.mean()) if len(public_auc) else np.nan,
        "public_mean_f1": float(per_inst["public_f1"].mean()),
        "validation_share_auc_above_0_5": float((valid_auc > 0.5).mean()) if len(valid_auc) else np.nan,
        "public_share_auc_above_0_5": float((public_auc > 0.5).mean()) if len(public_auc) else np.nan,
        "worst_3_public_auc": "; ".join(
            f"{row.instrument}:{row.public_roc_auc:.3f}"
            for row in worst_public.itertuples(index=False)
            if pd.notna(row.public_roc_auc)
        ),
        "worst_3_validation_auc": "; ".join(
            f"{row.instrument}:{row.validation_roc_auc:.3f}"
            for row in worst_valid.itertuples(index=False)
            if pd.notna(row.validation_roc_auc)
        ),
        "n_validation_events": int(per_inst["validation_n"].sum()),
        "n_public_events": int(per_inst["public_n"].sum()),
    }


def subset_frames(frames: BlendFrames, instruments: list[str], name: str) -> BlendFrames:
    return BlendFrames(
        name=name,
        logistic_features=frames.logistic_features,
        mlp_features=frames.mlp_features,
        valid=frames.valid[frames.valid["instrument"].isin(instruments)].copy(),
        public=frames.public[frames.public["instrument"].isin(instruments)].copy(),
    )


def evaluate_frames(frames: BlendFrames, instruments: list[str], name: str) -> tuple[dict[str, Any], pd.DataFrame]:
    per_inst = instrument_metrics(frames.valid, frames.public, instruments, name)
    aggregate = aggregate_from_per_instrument(per_inst, name, instruments)
    return aggregate, per_inst


def universe_definitions(base_per_inst: pd.DataFrame) -> dict[str, dict[str, Any]]:
    all_instruments = list(INSTRUMENTS)
    weak_drop = base_per_inst.loc[
        base_per_inst["validation_roc_auc"].fillna(0.0) >= 0.5,
        "instrument",
    ].tolist()
    if not weak_drop:
        weak_drop = all_instruments

    deterioration = base_per_inst["validation_roc_auc"] - base_per_inst["public_roc_auc"]
    unstable_drop = base_per_inst.loc[
        ~((deterioration > 0.10) | ((base_per_inst["validation_roc_auc"] >= 0.55) & (base_per_inst["public_roc_auc"] < 0.50))),
        "instrument",
    ].tolist()
    if not unstable_drop:
        unstable_drop = all_instruments

    return {
        "all_11_instruments": {
            "instruments": all_instruments,
            "selection_basis": "all coursework instruments",
            "diagnostic_only": False,
        },
        "equity_index_futures_only": {
            "instruments": [inst for inst in all_instruments if ASSET_CLASS.get(inst) == "equity_index"],
            "selection_basis": "complete asset class mapping",
            "diagnostic_only": False,
        },
        "energy_only": {
            "instruments": [inst for inst in all_instruments if ASSET_CLASS.get(inst) == "energy"],
            "selection_basis": "complete asset class mapping",
            "diagnostic_only": False,
        },
        "metals_only": {
            "instruments": [inst for inst in all_instruments if ASSET_CLASS.get(inst) == "metals"],
            "selection_basis": "complete asset class mapping",
            "diagnostic_only": False,
        },
        "drop_validation_weak_instruments": {
            "instruments": weak_drop,
            "selection_basis": "validation AUC >= 0.5 only",
            "diagnostic_only": False,
        },
        "drop_unstable_instruments": {
            "instruments": unstable_drop,
            "selection_basis": "validation-public deterioration diagnostic",
            "diagnostic_only": True,
        },
    }


def validation_cluster_importance(full: pd.DataFrame, feature_cols: list[str], repeats: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(CONFIG.seed)
    rows: list[dict[str, Any]] = []
    cluster_map: dict[str, list[str]] = {}
    for feature in feature_cols:
        if feature.startswith("hmm_"):
            continue
        cluster_map.setdefault(cluster_name_for_feature(feature), []).append(feature)

    for inst in INSTRUMENTS:
        inst_data = full[full["instrument"] == inst].sort_values("date").copy()
        train, validation, final_train = chronological_split(inst_data, PREDICTION_START)
        if train["label"].nunique() < 2 or validation["label"].nunique() < 2:
            split_point = max(1, int(len(final_train) * 0.75))
            train = final_train.iloc[:split_point].copy()
            validation = final_train.iloc[split_point:].copy()
        if train["label"].nunique() < 2 or validation["label"].nunique() < 2:
            continue

        x_train = clean_feature_matrix(train, feature_cols)
        y_train = train["label"].astype(int).to_numpy()
        x_valid = clean_feature_matrix(validation, feature_cols)
        y_valid = validation["label"].astype(int).to_numpy()
        params, threshold, _ = best_params_for_family("logistic", x_train, y_train, x_valid, y_valid)
        estimator = build_estimator("logistic", params, CONFIG.seed)
        estimator.fit(x_train, y_train)
        baseline_proba = probability_of_positive(estimator, x_valid)
        baseline = metric_bundle(y_valid, baseline_proba, threshold=threshold)

        for cluster, features in sorted(cluster_map.items()):
            auc_values: list[float] = []
            f1_values: list[float] = []
            for _ in range(repeats):
                permuted = x_valid.copy()
                if len(permuted) <= 1:
                    continue
                order = rng.permutation(len(permuted))
                permuted.loc[:, features] = permuted.loc[:, features].iloc[order].to_numpy()
                permuted_proba = probability_of_positive(estimator, permuted)
                permuted_metrics = metric_bundle(y_valid, permuted_proba, threshold=threshold)
                auc_values.append(permuted_metrics["roc_auc"])
                f1_values.append(permuted_metrics["f1"])
            permuted_auc = float(np.nanmean(auc_values)) if auc_values else np.nan
            permuted_f1 = float(np.nanmean(f1_values)) if f1_values else np.nan
            rows.append(
                {
                    "instrument": inst,
                    "asset_class": ASSET_CLASS.get(inst, "unknown"),
                    "cluster": cluster,
                    "n_features": len(features),
                    "features": "; ".join(features),
                    "baseline_validation_roc_auc": baseline["roc_auc"],
                    "permuted_validation_roc_auc": permuted_auc,
                    "roc_auc_drop": baseline["roc_auc"] - permuted_auc
                    if np.isfinite(baseline["roc_auc"]) and np.isfinite(permuted_auc)
                    else np.nan,
                    "baseline_validation_f1": baseline["f1"],
                    "permuted_validation_f1": permuted_f1,
                    "f1_drop": baseline["f1"] - permuted_f1
                    if np.isfinite(baseline["f1"]) and np.isfinite(permuted_f1)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def keep_features_by_cluster(feature_cols: list[str], retained_clusters: set[str]) -> list[str]:
    out = [
        feature
        for feature in feature_cols
        if cluster_name_for_feature(feature) in retained_clusters or feature == "primary_signal"
    ]
    return sorted(set(out))


def feature_set_definitions(full_features: list[str], importance: pd.DataFrame) -> dict[str, dict[str, Any]]:
    cluster_mean = importance.groupby("cluster")["roc_auc_drop"].mean().sort_values(ascending=False)
    all_clusters = {cluster_name_for_feature(feature) for feature in full_features}
    non_negative = set(cluster_mean[cluster_mean >= 0.0].index)
    above_near_zero = set(cluster_mean[cluster_mean > NEAR_ZERO_IMPORTANCE_CUTOFF].index)
    top_5 = set(cluster_mean.head(5).index)
    economic_core = {
        "return_momentum",
        "volatility_range",
        "volatility_stress",
        "technical_indicators",
        "volume_liquidity",
        "cross_sectional",
        "gmm_regime",
        "pca_latent",
        "signal_history",
    }
    return {
        "full_features": {
            "features": sorted(full_features),
            "retained_clusters": sorted(all_clusters),
            "selection_basis": "all non-HMM feature clusters",
        },
        "drop_negative_importance_clusters": {
            "features": keep_features_by_cluster(full_features, non_negative),
            "retained_clusters": sorted(non_negative),
            "selection_basis": "validation cluster AUC drop >= 0",
        },
        "drop_near_zero_importance_clusters": {
            "features": keep_features_by_cluster(full_features, above_near_zero),
            "retained_clusters": sorted(above_near_zero),
            "selection_basis": f"validation cluster AUC drop > {NEAR_ZERO_IMPORTANCE_CUTOFF}",
        },
        "keep_top_5_clusters": {
            "features": keep_features_by_cluster(full_features, top_5),
            "retained_clusters": sorted(top_5),
            "selection_basis": "top 5 validation cluster AUC drops",
        },
        "keep_economic_core_clusters": {
            "features": keep_features_by_cluster(full_features, economic_core),
            "retained_clusters": sorted(economic_core & all_clusters),
            "selection_basis": "pre-specified economically interpretable clusters",
        },
    }


def append_feature_metadata(
    aggregate: dict[str, Any],
    feature_name: str,
    feature_cols: list[str],
    full_features: list[str],
) -> dict[str, Any]:
    retained = sorted({cluster_name_for_feature(feature) for feature in feature_cols})
    all_clusters = sorted({cluster_name_for_feature(feature) for feature in full_features})
    dropped = sorted(set(all_clusters) - set(retained))
    aggregate.update(
        {
            "feature_set": feature_name,
            "n_features_retained": len(feature_cols),
            "clusters_retained": ",".join(retained),
            "clusters_dropped": ",".join(dropped),
            "interpretability_improves": feature_name != "full_features" and len(feature_cols) < len(full_features),
        }
    )
    return aggregate


def feature_results_to_csv_rows(aggregates: list[dict[str, Any]], per_frames: list[pd.DataFrame]) -> pd.DataFrame:
    aggregate_rows = []
    for row in aggregates:
        aggregate_rows.append({"row_type": "aggregate", "instrument": "__mean__", **row})

    instrument_rows = []
    for per_inst in per_frames:
        feature_set = per_inst["feature_set"].iloc[0]
        for row in per_inst.to_dict("records"):
            instrument_rows.append({"row_type": "instrument", **row})
    return pd.DataFrame(aggregate_rows + instrument_rows)


def guardrail_status(
    row: dict[str, Any],
    base_validation_auc: float,
    base_validation_f1: float,
    base_worst_public_auc: float,
    selection_uses_public: bool,
) -> dict[str, Any]:
    public_auc_improvement = row["public_mean_roc_auc"] - PROMOTED_FINAL_MEAN_AUC
    public_f1_change = row["public_mean_f1"] - PROMOTED_FINAL_MEAN_F1
    validation_auc_similar = row["validation_mean_roc_auc"] >= base_validation_auc - VALIDATION_SIMILAR_AUC_TOL
    validation_f1_similar = row["validation_mean_f1"] >= base_validation_f1 - VALIDATION_SIMILAR_F1_TOL
    worst_public_auc = np.nan
    if row.get("worst_3_public_auc"):
        try:
            worst_public_auc = min(float(part.split(":")[1]) for part in row["worst_3_public_auc"].split("; "))
        except Exception:
            worst_public_auc = np.nan
    worst_not_materially_worse = (
        pd.notna(worst_public_auc)
        and pd.notna(base_worst_public_auc)
        and worst_public_auc >= base_worst_public_auc - WORST_INSTRUMENT_MATERIAL_DROP
    )
    passes = all(
        [
            public_auc_improvement >= 0.005,
            public_f1_change >= 0.0,
            validation_auc_similar or validation_f1_similar,
            bool(row["has_complete_asset_class"]),
            not selection_uses_public,
            worst_not_materially_worse,
            bool(row["direct_output_format_full_assignment_valid"]),
        ]
    )
    return {
        "public_auc_improvement_vs_frozen": public_auc_improvement,
        "public_f1_change_vs_frozen": public_f1_change,
        "validation_similar_to_full_final_style": bool(validation_auc_similar or validation_f1_similar),
        "selection_uses_public": selection_uses_public,
        "worst_public_auc_not_materially_worse": bool(worst_not_materially_worse),
        "passes_promotion_guardrails": bool(passes),
    }


def write_universe_summary(rows: pd.DataFrame, path: Path) -> None:
    best_validation = rows.sort_values("validation_mean_roc_auc", ascending=False).iloc[0]
    best_public = rows.sort_values("public_mean_roc_auc", ascending=False).iloc[0]
    lines = [
        "# Universe Ablation Summary",
        "",
        "This is a robustness appendix. It does not overwrite `outputs/metamodel_predictions.csv`.",
        "",
        "Instrument universes were evaluated with the frozen final model style: calibrated 0.50 Logistic + 0.50 signal-history MLP probability blend. Because the project trains models per instrument, a reduced universe is evaluated by retaining only the relevant instrument-level results.",
        "",
        "Asset-class mapping used from the coursework configuration:",
        "- Equity index futures: `es1s`, `nq1s`, `fesx1s`.",
        "- Energy: `cl1s`, `ho1s`, `rb1s`, `ng1s`.",
        "- Metals: `gc1s`, `si1s`, `hg1s`, `pl1s`.",
        "",
        f"- Best validation mean ROC AUC: `{best_validation.experiment}` at `{best_validation.validation_mean_roc_auc:.4f}`.",
        f"- Best public 2022H1 mean ROC AUC sanity check: `{best_public.experiment}` at `{best_public.public_mean_roc_auc:.4f}`.",
        "- Reduced universes are useful diagnostics, but a direct reduced-universe prediction file would not satisfy the full all-date/all-instrument deliverable unless dropped instruments were explicitly filled with neutral probabilities.",
        "- `drop_unstable_instruments` uses public 2022H1 deterioration information and is diagnostic only.",
        "",
        "## Results",
        "",
        "| Universe | Instruments | Validation AUC | Validation F1 | Public AUC | Public F1 | Guardrails |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows.itertuples(index=False):
        lines.append(
            f"| `{row.experiment}` | {row.n_instruments} | {row.validation_mean_roc_auc:.4f} | "
            f"{row.validation_mean_f1:.4f} | {row.public_mean_roc_auc:.4f} | {row.public_mean_f1:.4f} | "
            f"{row.passes_promotion_guardrails} |"
        )
    lines.extend(
        [
            "",
            "Conclusion: universe reduction remains a robustness appendix unless it passes all promotion guardrails and can still produce a complete coursework-valid prediction file.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_feature_summary(rows: pd.DataFrame, path: Path) -> None:
    aggregates = rows[rows["row_type"] == "aggregate"].copy()
    best_validation = aggregates.sort_values("validation_mean_roc_auc", ascending=False).iloc[0]
    lines = [
        "# Feature-Pruning Summary",
        "",
        "Feature-cluster pruning uses validation-period cluster permutation importance as the pruning signal. Public 2022H1 is reported only as a sanity check.",
        "",
        f"- Best validation mean ROC AUC: `{best_validation.feature_set}` at `{best_validation.validation_mean_roc_auc:.4f}`.",
        "- HMM features remain disabled.",
        "- Pruned feature sets may improve interpretability by using fewer clusters, but they are not promoted unless they pass all guardrails.",
        "",
        "| Feature Set | Features | Validation AUC | Validation F1 | Public AUC | Public F1 | Interpretability | Guardrails |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in aggregates.itertuples(index=False):
        lines.append(
            f"| `{row.feature_set}` | {int(row.n_features_retained)} | {row.validation_mean_roc_auc:.4f} | "
            f"{row.validation_mean_f1:.4f} | {row.public_mean_roc_auc:.4f} | {row.public_mean_f1:.4f} | "
            f"{row.interpretability_improves} | {row.passes_promotion_guardrails} |"
        )
    lines.append("")
    lines.append("Conclusion: feature pruning is reported as robustness evidence; final predictions remain frozen.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_combined_summary(rows: pd.DataFrame, path: Path) -> None:
    best_validation = rows.sort_values("validation_mean_roc_auc", ascending=False).iloc[0]
    best_public = rows.sort_values("public_mean_roc_auc", ascending=False).iloc[0]
    lines = [
        "# Universe + Feature-Pruning Combined Summary",
        "",
        "Combined tests reuse the best validation-selected pruned feature set and safe universe diagnostics. They do not create or overwrite final predictions.",
        "",
        f"- Best validation combination: `{best_validation.experiment}` at validation AUC `{best_validation.validation_mean_roc_auc:.4f}`.",
        f"- Best public sanity-check combination: `{best_public.experiment}` at public AUC `{best_public.public_mean_roc_auc:.4f}`.",
        "",
        "| Combination | Instruments | Features | Validation AUC | Validation F1 | Public AUC | Public F1 | Guardrails |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows.itertuples(index=False):
        lines.append(
            f"| `{row.experiment}` | {row.n_instruments} | {row.n_features_retained} | "
            f"{row.validation_mean_roc_auc:.4f} | {row.validation_mean_f1:.4f} | "
            f"{row.public_mean_roc_auc:.4f} | {row.public_mean_f1:.4f} | {row.passes_promotion_guardrails} |"
        )
    lines.append("")
    lines.append("Conclusion: no combined diagnostic should replace the frozen final unless all promotion guardrails pass.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def update_report_summary(universe_rows: pd.DataFrame, feature_rows: pd.DataFrame, combined_rows: pd.DataFrame) -> None:
    marker = "## Universe and Feature-Pruning Robustness"
    feature_aggregates = feature_rows[feature_rows["row_type"] == "aggregate"].copy()
    best_universe = universe_rows.sort_values("validation_mean_roc_auc", ascending=False).iloc[0]
    best_feature = feature_aggregates.sort_values("validation_mean_roc_auc", ascending=False).iloc[0]
    any_pass = bool(
        universe_rows["passes_promotion_guardrails"].any()
        or feature_aggregates["passes_promotion_guardrails"].any()
        or combined_rows["passes_promotion_guardrails"].any()
    )
    section = "\n".join(
        [
            marker,
            "",
            "A controlled robustness appendix tested reduced instrument universes and validation-driven feature-cluster pruning. This was not a new final submission run, HMM remained disabled, and `outputs/metamodel_predictions.csv` was not overwritten.",
            "",
            "Dropping weak instruments can lift average metrics, but it risks cherry-picking and may fail the full coursework output requirement if dropped instruments are not filled with neutral probabilities. Feature-cluster pruning can reduce overfitting and improve interpretability, but pruning decisions were based on validation-period cluster importance rather than public 2022H1.",
            "",
            f"- Best validation universe diagnostic: `{best_universe.experiment}` with validation AUC `{best_universe.validation_mean_roc_auc:.3f}` and public AUC `{best_universe.public_mean_roc_auc:.3f}`.",
            f"- Best validation feature-pruning diagnostic: `{best_feature.feature_set}` with validation AUC `{best_feature.validation_mean_roc_auc:.3f}` and public AUC `{best_feature.public_mean_roc_auc:.3f}`.",
            f"- Any diagnostic passed all promotion guardrails: `{any_pass}`.",
            "- Final submission remains unchanged because this experiment is an appendix robustness check and the frozen prediction hash is still preserved.",
            "",
        ]
    )
    package_root = Path(__file__).resolve().parents[1]
    workspace_root = package_root.parent
    for report_path in [workspace_root / "report_summary.md", package_root / "report_summary.md"]:
        if not report_path.exists():
            continue
        text = report_path.read_text(encoding="utf-8")
        if marker in text:
            text = text.split(marker)[0].rstrip() + "\n\n"
        report_path.write_text(text.rstrip() + "\n\n" + section, encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    output_dir = CONFIG.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    original_hash = sha256_file(CONFIG.final_prediction_path)
    if original_hash != PROMOTED_FINAL_HASH:
        raise RuntimeError(
            f"Frozen prediction hash changed before robustness run: {original_hash} != {PROMOTED_FINAL_HASH}"
        )

    full = load_modelling_dataset()
    core_logistic_features = feature_columns(full, "core_logistic", tuple())
    signal_history_mlp_features = feature_columns(full, "signal_history_mlp", ("side_adjusted", "signal_history"))
    full_features = feature_columns(
        full,
        "full_features",
        ("side_adjusted", "signal_history", "vol_stress", "trend_scanning", "regime_interactions"),
    )

    final_style = fit_calibrated_blend(
        full,
        list(INSTRUMENTS),
        core_logistic_features,
        signal_history_mlp_features,
        "final_style_all_11",
    )
    base_aggregate, base_per_inst = evaluate_frames(final_style, list(INSTRUMENTS), "all_11_instruments")
    base_worst_public_auc = base_per_inst["public_roc_auc"].dropna().nsmallest(3).min()

    universe_rows: list[dict[str, Any]] = []
    universe_per_rows: list[pd.DataFrame] = []
    for name, definition in universe_definitions(base_per_inst).items():
        instruments = definition["instruments"]
        frames = subset_frames(final_style, instruments, name)
        aggregate, per_inst = evaluate_frames(frames, instruments, name)
        aggregate.update(
            {
                "selection_basis": definition["selection_basis"],
                "diagnostic_only": definition["diagnostic_only"],
                "satisfies_at_least_one_complete_asset_class": aggregate["has_complete_asset_class"],
            }
        )
        aggregate.update(
            guardrail_status(
                aggregate,
                base_aggregate["validation_mean_roc_auc"],
                base_aggregate["validation_mean_f1"],
                base_worst_public_auc,
                selection_uses_public=definition["diagnostic_only"],
            )
        )
        universe_rows.append(aggregate)
        per_inst["universe"] = name
        per_inst["selection_basis"] = definition["selection_basis"]
        universe_per_rows.append(per_inst)

    universe_summary = pd.DataFrame(universe_rows)
    universe_per_instrument = pd.concat(universe_per_rows, ignore_index=True)
    universe_summary.to_csv(output_dir / "universe_ablation_results.csv", index=False)
    universe_per_instrument.to_csv(output_dir / "universe_ablation_per_instrument.csv", index=False)
    write_universe_summary(universe_summary, output_dir / "universe_ablation_summary.md")

    importance = validation_cluster_importance(full, full_features, repeats=3)
    importance.to_csv(output_dir / "feature_pruning_validation_cluster_importance.csv", index=False)
    feature_defs = feature_set_definitions(full_features, importance)

    feature_aggregates: list[dict[str, Any]] = []
    feature_per_frames: list[pd.DataFrame] = []
    feature_blends: dict[str, BlendFrames] = {}
    for feature_name, definition in feature_defs.items():
        feature_cols = definition["features"]
        frames = fit_calibrated_blend(full, list(INSTRUMENTS), feature_cols, feature_cols, feature_name)
        feature_blends[feature_name] = frames
        aggregate, per_inst = evaluate_frames(frames, list(INSTRUMENTS), feature_name)
        aggregate = append_feature_metadata(aggregate, feature_name, feature_cols, full_features)
        aggregate["selection_basis"] = definition["selection_basis"]
        aggregate.update(
            guardrail_status(
                aggregate,
                base_aggregate["validation_mean_roc_auc"],
                base_aggregate["validation_mean_f1"],
                base_worst_public_auc,
                selection_uses_public=False,
            )
        )
        feature_aggregates.append(aggregate)
        per_inst["feature_set"] = feature_name
        per_inst["n_features_retained"] = len(feature_cols)
        per_inst["clusters_retained"] = aggregate["clusters_retained"]
        per_inst["clusters_dropped"] = aggregate["clusters_dropped"]
        per_inst["interpretability_improves"] = aggregate["interpretability_improves"]
        feature_per_frames.append(per_inst)

    feature_results = feature_results_to_csv_rows(feature_aggregates, feature_per_frames)
    feature_results.to_csv(output_dir / "feature_pruning_results.csv", index=False)
    write_feature_summary(feature_results, output_dir / "feature_pruning_summary.md")

    feature_aggregate_df = pd.DataFrame(feature_aggregates)
    pruned_candidates = feature_aggregate_df[feature_aggregate_df["feature_set"] != "full_features"].copy()
    best_pruned = pruned_candidates.sort_values("validation_mean_roc_auc", ascending=False).iloc[0]
    best_pruned_name = str(best_pruned["feature_set"])

    asset_only = universe_summary[
        universe_summary["experiment"].isin(
            ["equity_index_futures_only", "energy_only", "metals_only"]
        )
    ].copy()
    best_asset_universe = asset_only.sort_values("validation_mean_roc_auc", ascending=False).iloc[0]
    safe_universes = universe_summary[~universe_summary["diagnostic_only"]].copy()
    best_validation_universe = safe_universes.sort_values("validation_mean_roc_auc", ascending=False).iloc[0]

    combined_specs = [
        ("all_11_plus_pruned_features", "all_11_instruments", list(INSTRUMENTS)),
        (
            f"{best_asset_universe.experiment}_plus_pruned_features",
            str(best_asset_universe.experiment),
            str(best_asset_universe.instruments).split(","),
        ),
        (
            f"{best_validation_universe.experiment}_plus_pruned_features",
            str(best_validation_universe.experiment),
            str(best_validation_universe.instruments).split(","),
        ),
    ]
    seen_combined: set[str] = set()
    combined_rows: list[dict[str, Any]] = []
    pruned_frames = feature_blends[best_pruned_name]
    pruned_feature_cols = feature_defs[best_pruned_name]["features"]
    for combined_name, universe_name, instruments in combined_specs:
        if combined_name in seen_combined:
            continue
        seen_combined.add(combined_name)
        frames = subset_frames(pruned_frames, instruments, combined_name)
        aggregate, _ = evaluate_frames(frames, instruments, combined_name)
        aggregate = append_feature_metadata(aggregate, best_pruned_name, pruned_feature_cols, full_features)
        aggregate.update(
            {
                "universe_source": universe_name,
                "feature_set": best_pruned_name,
                "selection_basis": "validation-selected universe/features; public used only for reporting",
            }
        )
        aggregate.update(
            guardrail_status(
                aggregate,
                base_aggregate["validation_mean_roc_auc"],
                base_aggregate["validation_mean_f1"],
                base_worst_public_auc,
                selection_uses_public=False,
            )
        )
        combined_rows.append(aggregate)

    combined_summary = pd.DataFrame(combined_rows)
    combined_summary.to_csv(output_dir / "universe_feature_pruning_combined_results.csv", index=False)
    write_combined_summary(combined_summary, output_dir / "universe_feature_pruning_combined_summary.md")
    update_report_summary(universe_summary, feature_results, combined_summary)

    final_hash = sha256_file(CONFIG.final_prediction_path)
    if final_hash != original_hash:
        raise RuntimeError(f"Frozen prediction file changed during robustness run: {final_hash} != {original_hash}")

    print(json.dumps(
        {
            "status": "completed",
            "frozen_prediction_hash": final_hash,
            "universe_best_validation": universe_summary.sort_values("validation_mean_roc_auc", ascending=False).iloc[0][
                "experiment"
            ],
            "feature_best_validation": best_pruned_name,
            "any_guardrail_pass": bool(
                universe_summary["passes_promotion_guardrails"].any()
                or feature_aggregate_df["passes_promotion_guardrails"].any()
                or combined_summary["passes_promotion_guardrails"].any()
            ),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
