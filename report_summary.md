# BUSI70575 Coursework Metamodel Summary

## 1. Executive Summary

This project builds a metamodel on top of supplied primary trading signals. The final model is the frozen calibrated `0.50 Logistic + 0.50 signal-history MLP` probability blend.

Final prediction file:

```text
outputs/metamodel_predictions.csv
```

Final SHA256:

```text
c5c7ca869d905b384ef3c9072c3377e0f43c7a7ad03c9125aa062077f0f9b369
```

The final output has columns `date,instrument,prediction`, where `prediction` is a probability in `[0, 1]`.

## 2. Assignment Interpretation

The task is meta-labeling, not direct return prediction.

The supplied primary signals provide the trade direction:

- `+1` = long
- `-1` = short
- `0` = no trade

The metamodel predicts whether a non-zero primary signal should be followed. A training row is a `date x instrument x non-zero primary signal` trade opportunity.

`X` = primary signal + lagged OHLCV-derived features + market context features.

`y` = binary triple-barrier meta-label.

`prediction` = probability that the primary signal should be followed.

## 3. Data and Preprocessing

The raw inputs are `ohlcv_data.csv` and `primary_signals.csv`.

The code standardizes date and instrument columns, reshapes primary signals into a long date-instrument panel, and validates the panel before feature construction.

Zero primary signals are not genuine trade opportunities for training. If the final deliverable requires all date-instrument rows, zero-signal rows receive neutral probability `0.5`.

## 4. Feature Engineering

Feature engineering uses lagged OHLCV and derived market information. Feature groups include:

- return and momentum features
- volatility and range features
- RSI, MACD, Bollinger-style, and stochastic technical indicators
- volume, liquidity, and open-interest features
- cross-sectional rank features
- GMM regime and PCA latent context
- signal-history features
- side-adjusted and volatility-stress features

Features are lagged to avoid look-ahead bias.

## 5. Triple-Barrier Labeling

The target is a binary triple-barrier meta-label, not raw future return.

For each non-zero primary signal, the label checks whether following that signal hits profit-taking before stop-loss within the vertical barrier. If profit-taking is hit first, `y = 1`. Otherwise, `y = 0`.

Barrier widths use lagged volatility, so the label is path-aware and risk-adjusted without using future information at prediction time.

## 6. Model Training and Hyperparameter Tuning

Model selection uses chronological train/validation splits. The effective training period is mostly 2020 through 2021H1, and validation is 2021H2, with label end-date checks to avoid event overlap into future periods.

The project compares multiple model families:

- Logistic Regression
- Random Forest
- Extra Trees
- HistGradientBoosting
- AdaBoost
- MLP

Hyperparameter tuning, threshold selection, and probability calibration are based on the validation period, not hidden test data.

## 7. Final Model Selection

The final selected model is:

```text
Calibrated 0.50 Logistic + 0.50 signal-history MLP probability blend
```

The Logistic component provides a stable, interpretable anchor. The MLP component adds nonlinear signal-history information. The blended probability is sigmoid calibrated.

The final model is not baseline logistic only.

HMM was tested as an extension but remains disabled.

## 8. Testing and Out-of-Sample Evaluation

Public 2022H1 is used as a clean out-of-sample sanity-check window, not as the main tuning window.

The frozen final model has public 2022H1 mean ROC AUC around `0.583` and mean F1 around `0.626`.

Small public score differences should be interpreted cautiously because the public test sample is limited.

## 9. Cluster-Level Feature Importance

Cluster-level permutation importance is saved in:

```text
outputs/cluster_importance.csv
```

Cluster-level analysis is used because many financial features are correlated. The strongest average AUC-drop clusters in the final summaries include technical indicators, cross-sectional features, GMM regime features, and volume/liquidity features.

## 10. Robustness Checks

Label-specification search evaluated 144 triple-barrier specifications. The best global specification found was `vertical=10`, `profit_taking_multiplier=1.0`, `stop_loss_multiplier=1.0`, `vol_lookback=10`.

The label-search challenger showed strong validation performance but weaker public 2022H1 sanity-check performance, so it was not promoted.

Advanced challengers such as stacking and autoencoder-style experiments did not pass guardrails.

The primary-signal-only baseline had public mean ROC AUC about `0.471` and mean F1 about `0.641`; the high F1 mainly reflects near-always accepting non-zero trades rather than strong ranking power.

## 11. Universe and Feature-Pruning Robustness

A controlled robustness appendix tested reduced instrument universes and validation-driven feature-cluster pruning. This was not a new final submission run, HMM remained disabled, and `outputs/metamodel_predictions.csv` was not overwritten.

Dropping weak instruments can lift average metrics, but it risks cherry-picking and may fail the full coursework output requirement if dropped instruments are not filled with neutral probabilities. Feature-cluster pruning can reduce overfitting and improve interpretability, but pruning decisions were based on validation-period cluster importance rather than public 2022H1.

- Best validation universe diagnostic: `drop_validation_weak_instruments` with validation AUC `0.590` and public AUC `0.577`.
- Best validation feature-pruning diagnostic: `drop_near_zero_importance_clusters` with validation AUC `0.603` and public AUC `0.560`.
- No universe, feature-pruning, or combined diagnostic passed all promotion guardrails.

## 12. Final Submission Validation

The final submission remains unchanged.

Validation checks confirm:

- final output exists
- columns are exactly `date,instrument,prediction`
- predictions are in `[0, 1]`
- no missing prediction values
- no duplicate date-instrument rows
- final SHA256 matches the frozen hash
- HMM is disabled
- archived challengers are not promoted

The final GitHub preflight audit is saved in:

```text
outputs/final_github_preflight_audit.txt
```

## 13. Limitations

The triple-barrier label is a selected label specification, not an absolute truth.

Financial labels are noisy, and public 2022H1 contains a limited number of labeled events. Small public score changes can be unstable.

Selection overfitting risk is controlled through chronological validation, leakage checks, promotion guardrails, and robustness appendices. Even so, the final model should be interpreted as a disciplined coursework metamodel rather than a production trading system.
