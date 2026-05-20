# BUSI70575 Pipeline Flow

This project follows the official coursework pipeline:

```text
ohlcv_data.csv + primary_signals.csv
        |
        v
data loading and panel validation
        |
        v
lagged OHLCV and market-context feature engineering
        |
        v
triple-barrier meta-label construction
        |
        v
model-family comparison and validation-period tuning
        |
        v
final calibrated Logistic-MLP probability blend
        |
        v
clean out-of-sample evaluation on 2022H1
        |
        v
cluster-level feature importance analysis
        |
        v
final prediction export
        |
        v
outputs/metamodel_predictions.csv
```

## Step-By-Step Map

1. Raw inputs

   `ohlcv_data.csv` contains prices, volume, and open interest.

   `primary_signals.csv` contains daily primary trading signals in `{-1, 0, +1}`.

2. Data loading

   `src/data_loader.py` loads the raw files, standardizes date/instrument columns, converts primary signals into long format, and merges price and signal data.

3. Validation

   `src/validation.py` checks input files, panel consistency, duplicate date-instrument rows, final prediction format, and leakage-related assumptions.

4. Feature engineering

   `src/features.py` builds lagged OHLCV-derived features, momentum, volatility, technical indicators, volume/liquidity features, cross-sectional ranks, GMM/PCA context, signal-history features, and controlled extension features.

5. Triple-barrier labeling

   `src/labeling.py` creates binary triple-barrier meta-labels for non-zero primary signals. The label asks whether following the primary signal hits profit-taking before stop-loss within the vertical barrier.

6. Model training and comparison

   `src/models.py` handles chronological train/validation splits, model grids, Logistic Regression, tree-based models, boosting models, MLP, and final model reproduction.

7. Evaluation

   `src/evaluation.py` computes ROC AUC, precision, recall, F1, confusion matrices, threshold analysis, per-instrument metrics, and baseline comparison against blindly following every non-zero primary signal.

8. Feature importance

   `src/importance.py` computes and loads cluster-level feature importance. Cluster-level analysis is used because many financial features are correlated.

9. Reporting and audit files

   `src/reporting.py` writes CSV outputs, markdown summaries, prediction exports, final audits, and SHA256 hashes.

10. Shared configuration and helpers

   `src/config.py` stores paths, dates, feature flags, final model metadata, label defaults, and HMM disabled status.

   `src/utils.py` stores shared helper functions such as seed setting, safe division, rolling calculations, directory creation, and SHA256 hashing.

## Main Runner

The main runner is:

```text
coursework/coursework_metamodel.py
```

It is intentionally written as a pipeline map:

1. Load and validate data.
2. Feature engineering.
3. Triple-barrier labeling.
4. Train and compare model families.
5. Reproduce final selected model.
6. Clean OOS evaluation.
7. Cluster-level feature importance.
8. Export predictions.
9. Validate final output.
10. Write audit files.

From the project root, run:

```bash
python3 coursework_metamodel.py
```

## Final Output

The final coursework prediction file is:

```text
outputs/metamodel_predictions.csv
```

It must contain exactly:

```text
date,instrument,prediction
```

The prediction is a probability in `[0, 1]` that the primary signal should be followed.
