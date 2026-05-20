# BUSI70575 Metamodel Coursework

Project purpose: this project builds a metamodel on top of supplied primary trading signals.

Primary signal:

- `+1` = long
- `-1` = short
- `0` = no trade

Metamodel prediction: probability that the primary signal should be followed.

Final model:

```text
Calibrated 0.50 Logistic + 0.50 signal-history MLP probability blend
```

Final output:

```text
outputs/metamodel_predictions.csv
```

Final prediction SHA256:

```text
c5c7ca869d905b384ef3c9072c3377e0f43c7a7ad03c9125aa062077f0f9b369
```

## 1. Coursework Pipeline

The task is meta-labeling, not direct return prediction.

The supplied primary signals provide the trade direction. The metamodel decides whether each non-zero primary signal should be trusted.

Pipeline:

```text
OHLCV + primary signals
-> feature engineering
-> triple-barrier meta-labeling
-> model-family comparison and validation tuning
-> final calibrated Logistic-MLP probability blend
-> clean out-of-sample evaluation
-> cluster-level feature importance
-> final prediction export
```

The final CSV must contain exactly:

```text
date,instrument,prediction
```

where `prediction` is in `[0, 1]`.

## 2. Folder Structure

```text
coursework/
├── README.md
├── report_summary.md
├── coursework_metamodel.py
├── requirements.txt
├── docs/
│   ├── PROJECT_GUIDE_FOR_TEAMMATES.md
│   ├── PIPELINE_FLOW.md
│   └── FINAL_SUBMISSION_CHECKLIST.md
├── notebooks/
│   ├── 01_pipeline_walkthrough.ipynb
│   ├── 02_label_search_appendix.ipynb
│   └── 03_robustness_experiments.ipynb
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── features.py
│   ├── labeling.py
│   ├── models.py
│   ├── evaluation.py
│   ├── importance.py
│   ├── validation.py
│   ├── reporting.py
│   └── utils.py
└── outputs/
    ├── metamodel_predictions.csv
    ├── evaluation_summary.csv
    ├── model_comparison.csv
    ├── cluster_importance.csv
    ├── threshold_analysis.csv
    ├── baseline_comparison.csv
    ├── final_integration_audit.txt
    └── archive/
```

In this local workspace, `coursework/outputs` points to the shared top-level `outputs/` folder. The root-level `coursework_metamodel.py` is a convenience wrapper for running the final pipeline from the repository root.

## 3. Input Files

Expected raw inputs:

- `ohlcv_data.csv`
- `primary_signals.csv`

`primary_signals.csv` contains daily primary trading signals in `{-1, 0, +1}`.

## 4. How To Run

From the repository root:

```bash
python3 coursework_metamodel.py
```

Or from inside `coursework/`:

```bash
python3 coursework_metamodel.py
```

The final runner is submission-oriented. It validates and documents the frozen final output. It does not promote challengers or run a new modelling experiment.

## 5. Notebook Guide

- `notebooks/01_pipeline_walkthrough.ipynb`: main coursework pipeline walkthrough.
- `notebooks/02_label_search_appendix.ipynb`: label-specification search appendix.
- `notebooks/03_robustness_experiments.ipynb`: robustness and challenger appendix notes.

The notebooks call functions from `src/`; they should not duplicate large pipeline blocks.

## 6. Main Outputs

Final submission output:

- `outputs/metamodel_predictions.csv`

Core final outputs:

- `outputs/evaluation_summary.csv`
- `outputs/model_comparison.csv`
- `outputs/cluster_importance.csv`
- `outputs/threshold_analysis.csv`
- `outputs/baseline_comparison.csv`
- `outputs/final_integration_audit.txt`
- `outputs/final_github_preflight_audit.txt`

Optional robustness summaries:

- `outputs/universe_ablation_summary.md`
- `outputs/feature_pruning_summary.md`
- `outputs/universe_feature_pruning_combined_summary.md`

Archived experiments live in `outputs/archive/` and are not final submission outputs.

## 7. Final Model Explanation

The final model is the calibrated 0.50 Logistic + 0.50 signal-history MLP probability blend.

Logistic Regression provides a stable, interpretable anchor. The signal-history MLP adds nonlinear information about whether recent primary signals have tended to work or fail. Sigmoid calibration improves probability quality.

The final model is not baseline logistic only.

HMM is not the final model and remains disabled.

## 8. Robustness / Appendix Experiments

Several appendix experiments were run and documented, but not promoted:

- Label-specification search evaluated alternative triple-barrier settings.
- Advanced challengers such as stacking and autoencoder-style experiments did not pass guardrails.
- Universe ablation and feature-pruning robustness checks did not pass promotion guardrails.
- HMM remains disabled.

These experiments are useful for the report appendix, but the final prediction file remains unchanged.

## 9. Reproducibility Notes

Train/validation splitting is chronological. Public 2022H1 is used as a clean out-of-sample sanity-check window, not as the main tuning period.

Features are lagged to avoid look-ahead bias.

The target is a binary triple-barrier meta-label for a non-zero primary-signal trade opportunity. It should not be described as raw future return direction or "true ground truth".

## 10. Final Prediction Hash

The frozen final prediction hash is:

```text
c5c7ca869d905b384ef3c9072c3377e0f43c7a7ad03c9125aa062077f0f9b369
```

If this hash changes, stop and investigate before submission.

## 11. What Not To Change Before Submission

- Do not overwrite `outputs/metamodel_predictions.csv`.
- Do not change the final model.
- Do not enable HMM.
- Do not promote label-search challengers.
- Do not promote advanced challengers.
- Do not promote universe or feature-pruning robustness variants.
- Do not tune on public 2022H1 or hidden test data.
- Do not describe the final model as baseline logistic only.
- Do not treat archive files as final submission outputs.
