"""New-period prediction utilities for the promoted metamodel recipe.

The final coursework submission keeps the public-2022H1 prediction file
frozen. This module is for later inference windows: it rebuilds the same
feature and labeling pipeline, trains only on data strictly before the new
prediction window, and exports fresh `date,instrument,prediction` probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import CONFIG, CourseworkConfig, INSTRUMENTS
from .data_loader import load_and_merge_data
from .evaluation import metric_bundle, optimize_threshold, selection_score
from .features import build_feature_matrix
from .labeling import create_triple_barrier_labels
from .models import (
    ExperimentSpec,
    build_estimator,
    candidate_grid,
    chronological_split,
    clean_feature_matrix,
    feature_columns_for_experiment,
    probability_of_positive,
)
from .validation import validate_input_files, validate_panel, validate_prediction_file


@dataclass(frozen=True)
class NewPeriodPredictionResult:
    output_path: Path
    rows: int
    date_count: int
    instrument_count: int
    used_sigmoid_calibration: bool


def _best_family_estimator(
    family: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    final_train: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
) -> tuple[Any, pd.DataFrame] | None:
    if train.empty or validation.empty or final_train.empty:
        return None
    if train["label"].nunique() < 2 or final_train["label"].nunique() < 2:
        return None

    x_train = clean_feature_matrix(train, feature_cols)
    y_train = train["label"].astype(int).to_numpy()
    x_valid = clean_feature_matrix(validation, feature_cols)
    y_valid = validation["label"].astype(int).to_numpy()

    best: dict[str, Any] | None = None
    for params in candidate_grid(seed).get(family, []):
        estimator = build_estimator(family, params, seed)
        try:
            estimator.fit(x_train, y_train)
            valid_proba = probability_of_positive(estimator, x_valid)
            metrics = metric_bundle(y_valid, valid_proba, threshold=0.5)
            score = selection_score(metrics)
            if best is None or score > best["score"]:
                best = {
                    "params": params,
                    "score": score,
                    "threshold": optimize_threshold(y_valid, valid_proba),
                    "valid_proba": valid_proba,
                }
        except Exception:
            continue

    if best is None:
        return None

    final_estimator = build_estimator(family, best["params"], seed)
    x_final = clean_feature_matrix(final_train, feature_cols)
    y_final = final_train["label"].astype(int).to_numpy()
    final_estimator.fit(x_final, y_final)

    validation_frame = validation[["date", "instrument", "label"]].copy()
    validation_frame.attrs.clear()
    validation_frame[f"{family}_probability"] = best["valid_proba"]
    return final_estimator, validation_frame


def _split_for_prediction(inst_data: pd.DataFrame, prediction_start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, validation, final_train = chronological_split(inst_data, prediction_start)
    if train["label"].nunique() < 2 or validation["label"].nunique() < 2:
        labelled = final_train.dropna(subset=["label"]).copy()
        split_point = max(1, int(len(labelled) * 0.75))
        train = labelled.iloc[:split_point].copy()
        validation = labelled.iloc[split_point:].copy()
    return train, validation, final_train


def fit_sigmoid_calibrator(validation_blend: pd.DataFrame) -> Pipeline | None:
    if validation_blend.empty or validation_blend["label"].nunique() < 2:
        return None
    calibrator = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=CONFIG.seed)),
        ]
    )
    calibrator.fit(validation_blend[["raw_probability"]].to_numpy(), validation_blend["label"].astype(int).to_numpy())
    return calibrator


def predict_new_period(
    config: CourseworkConfig = CONFIG,
    output_path: str | Path | None = None,
) -> NewPeriodPredictionResult:
    """Train the promoted blend recipe and export predictions for a new window."""

    validate_input_files(config)
    raw_data = load_and_merge_data(config)
    validate_panel(raw_data, config)

    feature_data, _, _ = build_feature_matrix(raw_data, config)
    labeled_data, _ = create_triple_barrier_labels(feature_data, config)

    prediction_start = pd.Timestamp(config.prediction_start)
    prediction_end = pd.Timestamp(config.prediction_end)
    base_cols = feature_columns_for_experiment(labeled_data, ExperimentSpec("baseline", tuple(), 0), use_hmm_extension=False)
    signal_history_cols = feature_columns_for_experiment(
        labeled_data,
        ExperimentSpec("baseline + side_adjusted + signal_history", ("side_adjusted", "signal_history"), 2),
        use_hmm_extension=False,
    )

    pred_rows: list[pd.DataFrame] = []
    valid_rows: list[pd.DataFrame] = []
    pred_frame = labeled_data[
        (labeled_data["date"] >= prediction_start)
        & (labeled_data["date"] <= prediction_end)
    ].copy()

    for instrument in INSTRUMENTS:
        inst_data = labeled_data[labeled_data["instrument"] == instrument].sort_values("date").copy()
        inst_pred = pred_frame[pred_frame["instrument"] == instrument].sort_values("date").copy()
        if inst_pred.empty:
            continue

        train, validation, final_train = _split_for_prediction(inst_data, prediction_start)
        logistic_fit = _best_family_estimator("logistic", train, validation, final_train, base_cols, config.seed)
        mlp_fit = _best_family_estimator("mlp", train, validation, final_train, signal_history_cols, config.seed)

        component_predictions: list[np.ndarray] = []
        if logistic_fit is not None:
            logistic_model, logistic_valid = logistic_fit
            component_predictions.append(
                probability_of_positive(logistic_model, clean_feature_matrix(inst_pred, base_cols))
            )
        if mlp_fit is not None:
            mlp_model, mlp_valid = mlp_fit
            component_predictions.append(
                probability_of_positive(mlp_model, clean_feature_matrix(inst_pred, signal_history_cols))
            )

        if len(component_predictions) == 2:
            raw_probability = 0.5 * component_predictions[0] + 0.5 * component_predictions[1]
            validation_blend = logistic_valid.merge(mlp_valid, on=["date", "instrument", "label"], how="inner")
            if not validation_blend.empty:
                validation_blend["raw_probability"] = (
                    0.5 * validation_blend["logistic_probability"]
                    + 0.5 * validation_blend["mlp_probability"]
                )
                valid_rows.append(validation_blend[["date", "instrument", "label", "raw_probability"]])
        elif len(component_predictions) == 1:
            raw_probability = component_predictions[0]
        else:
            raw_probability = np.full(len(inst_pred), 0.5)

        pred_rows.append(
            pd.DataFrame(
                {
                    "date": inst_pred["date"].dt.strftime("%Y-%m-%d"),
                    "instrument": instrument,
                    "primary_signal": inst_pred["primary_signal"].to_numpy(),
                    "raw_probability": np.asarray(raw_probability, dtype=float),
                }
            )
        )

    if not pred_rows:
        predictions = pd.DataFrame(columns=["date", "instrument", "prediction"])
        used_calibration = False
    else:
        predictions = pd.concat(pred_rows, ignore_index=True)
        validation_blend = pd.concat(valid_rows, ignore_index=True) if valid_rows else pd.DataFrame()
        calibrator = fit_sigmoid_calibrator(validation_blend)
        used_calibration = calibrator is not None
        if calibrator is not None:
            predictions["prediction"] = calibrator.predict_proba(predictions[["raw_probability"]].to_numpy())[:, 1]
        else:
            predictions["prediction"] = predictions["raw_probability"]

        predictions["prediction"] = predictions["prediction"].clip(0.001, 0.999).round(6)
        predictions = predictions[["date", "instrument", "prediction"]].sort_values(["date", "instrument"]).reset_index(drop=True)

    path = Path(output_path) if output_path is not None else config.output_dir / "new_period_predictions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False)
    validation = validate_prediction_file(path)
    return NewPeriodPredictionResult(
        output_path=path,
        rows=int(validation["rows"]),
        date_count=int(validation["n_dates"]),
        instrument_count=int(validation["n_instruments"]),
        used_sigmoid_calibration=used_calibration,
    )
