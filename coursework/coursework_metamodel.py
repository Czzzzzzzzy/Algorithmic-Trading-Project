#!/usr/bin/env python3
"""Final BUSI70575 coursework pipeline runner.

This file is intentionally a simple map of the official coursework pipeline.
Implementation details live in `src/`; frozen final predictions are validated,
not retuned or replaced.
"""

from __future__ import annotations

try:
    from src.config import CONFIG
    from src.data_loader import load_and_merge_data
    from src.evaluation import evaluate_final_model, compare_to_primary_signal_baseline
    from src.features import build_feature_matrix
    from src.importance import compute_cluster_level_importance
    from src.labeling import create_triple_barrier_labels
    from src.models import train_final_model, train_model_suite
    from src.reporting import export_predictions, write_final_audit, write_model_outputs
    from src.validation import validate_input_files, validate_panel, validate_prediction_file
except ModuleNotFoundError:
    from .src.config import CONFIG
    from .src.data_loader import load_and_merge_data
    from .src.evaluation import evaluate_final_model, compare_to_primary_signal_baseline
    from .src.features import build_feature_matrix
    from .src.importance import compute_cluster_level_importance
    from .src.labeling import create_triple_barrier_labels
    from .src.models import train_final_model, train_model_suite
    from .src.reporting import export_predictions, write_final_audit, write_model_outputs
    from .src.validation import validate_input_files, validate_panel, validate_prediction_file


def main() -> None:
    # 1. Load and validate data
    validate_input_files(CONFIG)
    raw_data = load_and_merge_data(CONFIG)
    validate_panel(raw_data, CONFIG)

    # 2. Feature engineering
    feature_data, feature_columns, feature_groups = build_feature_matrix(
        raw_data,
        CONFIG,
    )

    # 3. Triple-barrier meta-labeling
    labeled_data, label_diagnostics = create_triple_barrier_labels(
        feature_data,
        CONFIG,
    )

    # 4. Train and compare model families
    model_results = train_model_suite(labeled_data, feature_columns, CONFIG)

    # 5. Reproduce final selected model
    final_model, final_predictions = train_final_model(
        labeled_data,
        feature_columns,
        CONFIG,
    )

    # 6. Clean OOS evaluation
    evaluation_results = evaluate_final_model(
        labeled_data,
        final_predictions,
        CONFIG,
    )

    baseline_results = compare_to_primary_signal_baseline(
        labeled_data,
        final_predictions,
        CONFIG,
    )

    # 7. Cluster-level feature importance
    importance_results = compute_cluster_level_importance(
        final_model,
        labeled_data,
        feature_columns,
        feature_groups,
        CONFIG,
    )

    # 8. Export predictions without replacing the frozen file
    export_predictions(final_predictions, CONFIG)

    # 9. Validate final output
    validate_prediction_file(CONFIG.final_prediction_path)

    # 10. Write audit files and final reporting summaries
    write_model_outputs(
        model_results=model_results,
        evaluation_results=evaluation_results,
        baseline_results=baseline_results,
        importance_results=importance_results,
        label_diagnostics=label_diagnostics,
        config=CONFIG,
    )

    audit_path = write_final_audit(CONFIG)
    print(f"Final coursework integration audit written to {audit_path}")


if __name__ == "__main__":
    main()
