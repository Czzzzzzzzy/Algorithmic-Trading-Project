"""Model training, comparison, and final model reproduction.

This module contains the chronological split, model grids, estimators, and
helpers used to compare model families and reproduce the frozen final blend.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    ASSET_CLASS,
    CONTROLLED_FEATURE_GROUPS,
    CourseworkConfig,
    DEFAULT_MODEL_GRID,
    HMM_EXTENSION_FEATURES,
    ID_COLS,
    INSTRUMENTS,
    SMALL_SAMPLE_MODEL_GRID,
    USE_HMM_EXTENSION,
    USE_REGIME_INTERACTION_FEATURES,
    USE_SIDE_ADJUSTED_FEATURES,
    USE_SIGNAL_HISTORY_FEATURES,
    USE_TREND_SCANNING_FEATURES,
    USE_VOL_STRESS_FEATURES,
)
from .evaluation import baseline_comparison_metrics, metric_bundle, optimize_threshold, selection_score


@dataclass
class InstrumentModel:
    instrument: str
    selected_family: str
    model: Any
    threshold: float
    feature_cols: list[str]
    rf_for_importance: Any | None
    validation_score: float
    n_train: int
    n_validation: int
    experiment_name: str = ""


@dataclass
class ExperimentSpec:
    name: str
    groups: tuple[str, ...]
    complexity: int


def clean_feature_matrix(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    x = frame[feature_cols].replace([np.inf, -np.inf], np.nan)
    return x


def candidate_grid(seed: int) -> dict[str, list[dict[str, Any]]]:
    return {
        "logistic": [
            {"C": c}
            for c in (0.1, 1.0)
        ],
        "random_forest": [
            {"n_estimators": 300, "max_depth": 3, "min_samples_leaf": 8, "max_features": "sqrt"},
            {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 12, "max_features": "sqrt"},
        ],
        "extra_trees": [
            {"n_estimators": 300, "max_depth": 3, "min_samples_leaf": 8, "max_features": "sqrt"},
            {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 12, "max_features": "sqrt"},
        ],
        "hist_gradient_boosting": [
            {"max_iter": 120, "learning_rate": 0.03, "max_leaf_nodes": 7, "l2_regularization": 0.1},
            {"max_iter": 120, "learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 0.1},
        ],
        "adaboost": [
            {"n_estimators": 120, "learning_rate": 0.05},
        ],
        "mlp": [
            {"hidden_layer_sizes": (24,), "alpha": 1e-3, "learning_rate_init": 1e-3},
        ],
    }


def build_estimator(family: str, params: dict[str, Any], seed: int) -> Pipeline:
    if family == "logistic":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=params["C"],
                        penalty="l2",
                        solver="lbfgs",
                        max_iter=1200,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if family == "random_forest":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=params["n_estimators"],
                        max_depth=params["max_depth"],
                        min_samples_leaf=params["min_samples_leaf"],
                        max_features=params["max_features"],
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if family == "extra_trees":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=params["n_estimators"],
                        max_depth=params["max_depth"],
                        min_samples_leaf=params["min_samples_leaf"],
                        max_features=params["max_features"],
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if family == "hist_gradient_boosting":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=params["max_iter"],
                        learning_rate=params["learning_rate"],
                        max_leaf_nodes=params["max_leaf_nodes"],
                        l2_regularization=params["l2_regularization"],
                        random_state=seed,
                    ),
                ),
            ]
        )
    if family == "adaboost":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    AdaBoostClassifier(
                        n_estimators=params["n_estimators"],
                        learning_rate=params["learning_rate"],
                        random_state=seed,
                    ),
                ),
            ]
        )
    if family == "mlp":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=params["hidden_layer_sizes"],
                        alpha=params["alpha"],
                        learning_rate_init=params["learning_rate_init"],
                        activation="relu",
                        solver="adam",
                        max_iter=500,
                        early_stopping=True,
                        validation_fraction=0.2,
                        n_iter_no_change=20,
                        random_state=seed,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown model family: {family}")


def probability_of_positive(estimator: Pipeline, x: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(x)
        classes = estimator.named_steps["model"].classes_
        if 1 in classes:
            return proba[:, list(classes).index(1)]
        return np.zeros(len(x))
    decision = estimator.decision_function(x)
    return 1 / (1 + np.exp(-decision))


def chronological_split(
    inst_data: pd.DataFrame,
    prediction_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation_start = prediction_start - pd.DateOffset(months=6)
    labelled = inst_data.dropna(subset=["label"]).copy()
    labelled["label"] = labelled["label"].astype(int)
    before_prediction = labelled[
        (labelled["date"] < prediction_start)
        & (labelled["label_end_date"] < prediction_start)
    ].sort_values("date")

    train = before_prediction[
        (before_prediction["date"] < validation_start)
        & (before_prediction["label_end_date"] < validation_start)
    ].copy()
    validation = before_prediction[before_prediction["date"] >= validation_start].copy()

    if len(train) < 20 or len(validation) < 10 or before_prediction["label"].nunique() < 2:
        split_point = max(1, int(len(before_prediction) * 0.75))
        train = before_prediction.iloc[:split_point].copy()
        validation = before_prediction.iloc[split_point:].copy()

    final_train = before_prediction.copy()
    return train, validation, final_train


def experiment_specs(use_hmm_extension: bool = USE_HMM_EXTENSION) -> list[ExperimentSpec]:
    specs = [ExperimentSpec("baseline", tuple(), 0)]
    groups: list[str] = []
    if USE_SIDE_ADJUSTED_FEATURES:
        groups.append("side_adjusted")
        specs.append(ExperimentSpec("baseline + side_adjusted", tuple(groups), len(groups)))
    if USE_SIGNAL_HISTORY_FEATURES:
        groups.append("signal_history")
        specs.append(ExperimentSpec("+ signal_history", tuple(groups), len(groups)))
    if USE_VOL_STRESS_FEATURES:
        groups.append("vol_stress")
        specs.append(ExperimentSpec("+ vol_stress", tuple(groups), len(groups)))
    if USE_TREND_SCANNING_FEATURES:
        groups.append("trend_scanning")
        specs.append(ExperimentSpec("+ trend_scanning", tuple(groups), len(groups)))
    if USE_REGIME_INTERACTION_FEATURES:
        groups.append("regime_interactions")
        specs.append(ExperimentSpec("+ regime_interactions", tuple(groups), len(groups)))
    if use_hmm_extension:
        groups.append("hmm_extension")
        specs.append(ExperimentSpec("+ hmm_extension", tuple(groups), len(groups)))
    return specs


def feature_columns_for_experiment(
    full: pd.DataFrame,
    spec: ExperimentSpec,
    use_hmm_extension: bool = USE_HMM_EXTENSION,
) -> list[str]:
    numeric_cols = full.select_dtypes(include=[np.number]).columns.tolist()
    base_exclusions = set().union(*CONTROLLED_FEATURE_GROUPS.values())
    if not use_hmm_extension:
        base_exclusions |= set(HMM_EXTENSION_FEATURES)

    feature_cols = [
        c for c in numeric_cols
        if c not in ID_COLS
        and not c.startswith("label")
        and c not in {"side_return", "holding_days"}
        and c not in base_exclusions
    ]
    for group in spec.groups:
        feature_cols.extend([c for c in CONTROLLED_FEATURE_GROUPS[group] if c in full.columns])
    feature_cols.append("primary_signal")
    return sorted(set(c for c in feature_cols if c in full.columns))


def evaluate_estimator_on_test(
    experiment_name: str,
    inst: str,
    family: str,
    params: dict[str, Any],
    estimator: Pipeline,
    threshold: float,
    validation_score: float,
    train_n: int,
    validation_n: int,
    inst_data: pd.DataFrame,
    feature_cols: list[str],
    prediction_start: pd.Timestamp,
    prediction_end: pd.Timestamp,
    is_selected_family: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    test = inst_data[
        (inst_data["date"] >= prediction_start)
        & (inst_data["date"] <= prediction_end)
        & inst_data["label"].notna()
    ].copy()
    if len(test) == 0:
        return (
            {
                "experiment": experiment_name,
                "instrument": inst,
                "family": family,
                "params": json.dumps(params, sort_keys=True),
                "is_selected_family": is_selected_family,
                "n_train": train_n,
                "n_validation": validation_n,
                "n_test": 0,
                "validation_score": validation_score,
            },
            [],
        )

    x_test = clean_feature_matrix(test, feature_cols)
    y_test = test["label"].astype(int).to_numpy()
    proba = probability_of_positive(estimator, x_test)
    metrics_tuned = metric_bundle(y_test, proba, threshold=threshold)
    metrics_05 = metric_bundle(y_test, proba, threshold=0.5)
    baseline = baseline_comparison_metrics(y_test, test["side_return"])
    meta_take = proba >= threshold

    row = {
        "experiment": experiment_name,
        "instrument": inst,
        "asset_class": ASSET_CLASS.get(inst, "unknown"),
        "family": family,
        "params": json.dumps(params, sort_keys=True),
        "is_selected_family": is_selected_family,
        "selected_threshold": threshold,
        "validation_score": validation_score,
        "n_train": train_n,
        "n_validation": validation_n,
        "n_test": len(test),
        **{f"oos_{k}": v for k, v in metrics_tuned.items()},
        **{f"oos_05_{k}": v for k, v in metrics_05.items()},
        **{f"baseline_{k}": v for k, v in baseline.items()},
        "meta_trade_rate": float(meta_take.mean()),
        "meta_mean_side_return_when_taken": float(test.loc[meta_take, "side_return"].mean()) if meta_take.any() else np.nan,
        "meta_total_side_return_when_taken": float(test.loc[meta_take, "side_return"].sum()) if meta_take.any() else 0.0,
    }

    threshold_rows = []
    for fixed_threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
        metrics = metric_bundle(y_test, proba, threshold=fixed_threshold)
        threshold_rows.append(
            {
                "experiment": experiment_name,
                "instrument": inst,
                "family": family,
                "is_selected_family": is_selected_family,
                "threshold": fixed_threshold,
                **metrics,
            }
        )
    return row, threshold_rows


def fit_experiment_for_instrument(
    experiment_name: str,
    inst: str,
    inst_data: pd.DataFrame,
    feature_cols: list[str],
    prediction_start: pd.Timestamp,
    prediction_end: pd.Timestamp,
    seed: int,
    min_train_events: int,
) -> tuple[InstrumentModel | None, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train, validation, final_train = chronological_split(inst_data, prediction_start)
    comparison_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []

    if len(final_train) < 10 or final_train["label"].nunique() < 2:
        return None, comparison_rows, ablation_rows, threshold_rows

    if train["label"].nunique() < 2 or validation["label"].nunique() < 2:
        split_point = max(1, int(len(final_train) * 0.75))
        train = final_train.iloc[:split_point].copy()
        validation = final_train.iloc[split_point:].copy()
        if train["label"].nunique() < 2 or len(validation) == 0:
            return None, comparison_rows, ablation_rows, threshold_rows

    grids = candidate_grid(seed)
    if len(train) < min_train_events:
        grids = {
            "logistic": [{"C": 0.3}, {"C": 1.0}],
            "random_forest": [{"n_estimators": 250, "max_depth": 3, "min_samples_leaf": 5, "max_features": "sqrt"}],
            "extra_trees": [{"n_estimators": 250, "max_depth": 3, "min_samples_leaf": 5, "max_features": "sqrt"}],
            "hist_gradient_boosting": [{"max_iter": 80, "learning_rate": 0.05, "max_leaf_nodes": 7, "l2_regularization": 0.1}],
            "adaboost": [{"n_estimators": 80, "learning_rate": 0.05}],
            "mlp": [{"hidden_layer_sizes": (16,), "alpha": 1e-3, "learning_rate_init": 1e-3}],
        }

    x_train = clean_feature_matrix(train, feature_cols)
    y_train = train["label"].astype(int).to_numpy()
    x_valid = clean_feature_matrix(validation, feature_cols)
    y_valid = validation["label"].astype(int).to_numpy()

    best_by_family: dict[str, dict[str, Any]] = {}
    for family, params_list in grids.items():
        for params in params_list:
            estimator = build_estimator(family, params, seed)
            try:
                estimator.fit(x_train, y_train)
                valid_proba = probability_of_positive(estimator, x_valid)
                threshold = optimize_threshold(y_valid, valid_proba)
                metrics_05 = metric_bundle(y_valid, valid_proba, threshold=0.5)
                metrics_tuned = metric_bundle(y_valid, valid_proba, threshold=threshold)
                score = selection_score(metrics_05)
                row = {
                    "experiment": experiment_name,
                    "instrument": inst,
                    "family": family,
                    "params": json.dumps(params, sort_keys=True),
                    "validation_threshold": threshold,
                    "selection_score": score,
                    **{f"validation_{k}": v for k, v in metrics_05.items()},
                    **{f"validation_tuned_{k}": v for k, v in metrics_tuned.items()},
                    "n_train": len(train),
                    "n_validation": len(validation),
                }
                comparison_rows.append(row)
                if family not in best_by_family or score > best_by_family[family]["score"]:
                    best_by_family[family] = {
                        "score": score,
                        "params": params,
                        "threshold": threshold,
                    }
            except Exception as exc:
                comparison_rows.append(
                    {
                        "experiment": experiment_name,
                        "instrument": inst,
                        "family": family,
                        "params": json.dumps(params, sort_keys=True),
                        "selection_score": -np.inf,
                        "error": repr(exc),
                        "n_train": len(train),
                        "n_validation": len(validation),
                    }
                )

    if not best_by_family:
        return None, comparison_rows, ablation_rows, threshold_rows

    selected_family = max(best_by_family, key=lambda fam: best_by_family[fam]["score"])
    selected_model: InstrumentModel | None = None
    x_final = clean_feature_matrix(final_train, feature_cols)
    y_final = final_train["label"].astype(int).to_numpy()

    for family, best in best_by_family.items():
        final_estimator = build_estimator(family, best["params"], seed)
        try:
            final_estimator.fit(x_final, y_final)
        except Exception as exc:
            comparison_rows.append(
                {
                    "experiment": experiment_name,
                    "instrument": inst,
                    "family": family,
                    "params": json.dumps(best["params"], sort_keys=True),
                    "selection_score": best["score"],
                    "error": f"final_fit_failed: {exc!r}",
                    "n_train": len(final_train),
                    "n_validation": len(validation),
                }
            )
            continue

        is_selected = int(family == selected_family)
        ablation_row, family_threshold_rows = evaluate_estimator_on_test(
            experiment_name=experiment_name,
            inst=inst,
            family=family,
            params=best["params"],
            estimator=final_estimator,
            threshold=float(best["threshold"]),
            validation_score=float(best["score"]),
            train_n=len(final_train),
            validation_n=len(validation),
            inst_data=inst_data,
            feature_cols=feature_cols,
            prediction_start=prediction_start,
            prediction_end=prediction_end,
            is_selected_family=is_selected,
        )
        ablation_rows.append(ablation_row)
        threshold_rows.extend(family_threshold_rows)

        if is_selected:
            selected_model = InstrumentModel(
                instrument=inst,
                selected_family=family,
                model=final_estimator,
                threshold=float(best["threshold"]),
                feature_cols=feature_cols,
                rf_for_importance=None,
                validation_score=float(best["score"]),
                n_train=len(final_train),
                n_validation=len(validation),
                experiment_name=experiment_name,
            )

    for row in comparison_rows:
        row["selected_family"] = selected_family
        row["is_selected"] = int(row["family"] == selected_family and row.get("selection_score") == best_by_family[selected_family]["score"])

    return selected_model, comparison_rows, ablation_rows, threshold_rows


def tune_instrument_model(
    inst: str,
    inst_data: pd.DataFrame,
    feature_cols: list[str],
    prediction_start: pd.Timestamp,
    seed: int,
    min_train_events: int,
) -> tuple[InstrumentModel | None, list[dict[str, Any]]]:
    train, validation, final_train = chronological_split(inst_data, prediction_start)
    comparison_rows: list[dict[str, Any]] = []

    if len(final_train) < 10 or final_train["label"].nunique() < 2:
        return None, comparison_rows

    if train["label"].nunique() < 2 or validation["label"].nunique() < 2:
        split_point = max(1, int(len(final_train) * 0.75))
        train = final_train.iloc[:split_point].copy()
        validation = final_train.iloc[split_point:].copy()
        if train["label"].nunique() < 2 or len(validation) == 0:
            return None, comparison_rows

    grids = candidate_grid(seed)
    if len(train) < min_train_events:
        grids = {
            "logistic": [{"C": 0.3}, {"C": 1.0}],
            "random_forest": [
                {"n_estimators": 250, "max_depth": 3, "min_samples_leaf": 5, "max_features": "sqrt"},
                {"n_estimators": 250, "max_depth": None, "min_samples_leaf": 8, "max_features": "sqrt"},
            ],
            "mlp": [{"hidden_layer_sizes": (16,), "alpha": 1e-3, "learning_rate_init": 1e-3}],
        }

    x_train = clean_feature_matrix(train, feature_cols)
    y_train = train["label"].astype(int).to_numpy()
    x_valid = clean_feature_matrix(validation, feature_cols)
    y_valid = validation["label"].astype(int).to_numpy()

    best_by_family: dict[str, dict[str, Any]] = {}
    for family, params_list in grids.items():
        for params in params_list:
            estimator = build_estimator(family, params, seed)
            try:
                estimator.fit(x_train, y_train)
                valid_proba = probability_of_positive(estimator, x_valid)
                threshold = optimize_threshold(y_valid, valid_proba)
                metrics_05 = metric_bundle(y_valid, valid_proba, threshold=0.5)
                metrics_tuned = metric_bundle(y_valid, valid_proba, threshold=threshold)
                score = selection_score(metrics_05)
                row = {
                    "instrument": inst,
                    "family": family,
                    "params": json.dumps(params, sort_keys=True),
                    "validation_threshold": threshold,
                    "selection_score": score,
                    **{f"validation_{k}": v for k, v in metrics_05.items()},
                    **{f"validation_tuned_{k}": v for k, v in metrics_tuned.items()},
                    "n_train": len(train),
                    "n_validation": len(validation),
                }
                comparison_rows.append(row)
                if family not in best_by_family or score > best_by_family[family]["score"]:
                    best_by_family[family] = {
                        "score": score,
                        "params": params,
                        "threshold": threshold,
                        "estimator": estimator,
                    }
            except Exception as exc:
                comparison_rows.append(
                    {
                        "instrument": inst,
                        "family": family,
                        "params": json.dumps(params, sort_keys=True),
                        "selection_score": -np.inf,
                        "error": repr(exc),
                        "n_train": len(train),
                        "n_validation": len(validation),
                    }
                )

    if not best_by_family:
        return None, comparison_rows

    selected_family = max(best_by_family, key=lambda fam: best_by_family[fam]["score"])
    selected = best_by_family[selected_family]
    final_estimator = build_estimator(selected_family, selected["params"], seed)
    x_final = clean_feature_matrix(final_train, feature_cols)
    y_final = final_train["label"].astype(int).to_numpy()
    final_estimator.fit(x_final, y_final)

    rf_artifact = None
    if "random_forest" in best_by_family:
        try:
            rf_params = best_by_family["random_forest"]["params"]
            rf_artifact = build_estimator("random_forest", rf_params, seed)
            rf_artifact.fit(x_final, y_final)
        except Exception:
            rf_artifact = None

    for row in comparison_rows:
        row["selected_family"] = selected_family
        row["is_selected"] = int(row["family"] == selected_family and row.get("selection_score") == selected["score"])

    return (
        InstrumentModel(
            instrument=inst,
            selected_family=selected_family,
            model=final_estimator,
            threshold=float(selected["threshold"]),
            feature_cols=feature_cols,
            rf_for_importance=rf_artifact,
            validation_score=float(selected["score"]),
            n_train=len(final_train),
            n_validation=len(validation),
        ),
        comparison_rows,
    )


def fit_fixed_family_final_models(
    experiment_name: str,
    family: str,
    full: pd.DataFrame,
    feature_cols: list[str],
    prediction_start: pd.Timestamp,
    seed: int,
    min_train_events: int,
) -> dict[str, InstrumentModel]:
    models: dict[str, InstrumentModel] = {}
    params_list = candidate_grid(seed).get(family, [])
    if not params_list:
        return models
    for inst in INSTRUMENTS:
        inst_data = full[full["instrument"] == inst].sort_values("date").copy()
        train, validation, final_train = chronological_split(inst_data, prediction_start)
        if len(final_train) < 10 or final_train["label"].nunique() < 2:
            continue
        if train["label"].nunique() < 2 or validation["label"].nunique() < 2:
            split_point = max(1, int(len(final_train) * 0.75))
            train = final_train.iloc[:split_point].copy()
            validation = final_train.iloc[split_point:].copy()
            if train["label"].nunique() < 2 or len(validation) == 0:
                continue

        x_train = clean_feature_matrix(train, feature_cols)
        y_train = train["label"].astype(int).to_numpy()
        x_valid = clean_feature_matrix(validation, feature_cols)
        y_valid = validation["label"].astype(int).to_numpy()

        best: dict[str, Any] | None = None
        for params in params_list:
            estimator = build_estimator(family, params, seed)
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
            continue

        final_estimator = build_estimator(family, best["params"], seed)
        x_final = clean_feature_matrix(final_train, feature_cols)
        y_final = final_train["label"].astype(int).to_numpy()
        final_estimator.fit(x_final, y_final)
        models[inst] = InstrumentModel(
            instrument=inst,
            selected_family=family,
            model=final_estimator,
            threshold=float(best["threshold"]),
            feature_cols=feature_cols,
            rf_for_importance=None,
            validation_score=float(best["score"]),
            n_train=len(final_train),
            n_validation=len(validation),
            experiment_name=experiment_name,
        )
    return models


def train_model_suite(
    labeled_data: pd.DataFrame,
    feature_columns: list[str],
    config: CourseworkConfig,
) -> dict[str, pd.DataFrame]:
    """Load the already-produced model comparison artifacts for submission.

    The final integration pass is not a new experiment, so this function does
    not refit or retune models. It surfaces the model-suite artifacts produced
    by the reproducible pipeline: model comparison, feature ablation, and model
    ablation summaries.
    """

    outputs = {}
    for name in [
        "model_comparison.csv",
        "feature_ablation_results.csv",
        "feature_ablation_summary.csv",
        "model_ablation_summary.csv",
        "threshold_analysis.csv",
    ]:
        path = config.output_dir / name
        outputs[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return outputs


def train_final_model(
    labeled_data: pd.DataFrame,
    feature_columns: list[str],
    config: CourseworkConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Reproduce the selected final model by loading the frozen prediction file.

    The selected final model is the calibrated 50/50 Logistic + signal-history
    MLP probability blend. This integration step preserves the frozen
    prediction file instead of promoting any challenger.
    """

    predictions = pd.read_csv(config.final_prediction_path)
    final_model = {
        "name": config.final_model_name,
        "type": "frozen_probability_blend",
        "hmm_enabled": config.enable_hmm,
        "prediction_path": str(config.final_prediction_path),
        "mean_roc_auc": config.promoted_final_mean_auc,
        "mean_f1": config.promoted_final_mean_f1,
    }
    return final_model, predictions
