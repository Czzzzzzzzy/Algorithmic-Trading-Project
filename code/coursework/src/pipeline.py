"""Pipeline orchestration for the BUSI70575 coursework submission.

The implementation modules handle the modelling details; this file keeps the
end-to-end run order explicit and reusable from the command-line entry points.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import CONFIG, CourseworkConfig
from .data_loader import load_and_merge_data
from .evaluation import compare_to_primary_signal_baseline, evaluate_final_model
from .features import build_feature_matrix
from .importance import compute_cluster_level_importance
from .labeling import create_triple_barrier_labels
from .models import train_final_model, train_model_suite
from .reporting import export_predictions, write_final_audit, write_model_outputs
from .validation import validate_input_files, validate_panel, validate_prediction_file


@dataclass(frozen=True)
class PipelineArtifacts:
    """Paths and key in-memory outputs produced by a pipeline run."""

    prediction_path: Path
    audit_path: Path
    output_dir: Path
    final_model: dict[str, Any]
    final_predictions: pd.DataFrame


def run_coursework_pipeline(config: CourseworkConfig = CONFIG) -> PipelineArtifacts:
    """Run the official coursework pipeline in submission order."""

    validate_input_files(config)
    raw_data = load_and_merge_data(config)
    validate_panel(raw_data, config)

    feature_data, feature_columns, feature_groups = build_feature_matrix(raw_data, config)

    labeled_data, label_diagnostics = create_triple_barrier_labels(feature_data, config)

    model_results = train_model_suite(labeled_data, feature_columns, config)

    final_model, final_predictions = train_final_model(labeled_data, feature_columns, config)

    evaluation_results = evaluate_final_model(labeled_data, final_predictions, config)
    baseline_results = compare_to_primary_signal_baseline(labeled_data, final_predictions, config)

    importance_results = compute_cluster_level_importance(
        final_model,
        labeled_data,
        feature_columns,
        feature_groups,
        config,
    )

    prediction_path = export_predictions(final_predictions, config)
    validate_prediction_file(prediction_path)

    write_model_outputs(
        model_results=model_results,
        evaluation_results=evaluation_results,
        baseline_results=baseline_results,
        importance_results=importance_results,
        label_diagnostics=label_diagnostics,
        config=config,
    )

    audit_path = write_final_audit(config)
    return PipelineArtifacts(
        prediction_path=prediction_path,
        audit_path=audit_path,
        output_dir=config.output_dir,
        final_model=final_model,
        final_predictions=final_predictions,
    )
