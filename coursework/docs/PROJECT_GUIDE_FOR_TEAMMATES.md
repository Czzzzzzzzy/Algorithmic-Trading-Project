# BUSI70575 Coursework Project Guide

## 1. Big Picture: What Are We Trying To Do?

The coursework asks us to build a metamodel on top of supplied primary trading signals.

The primary signal already gives a trading direction:

- `+1` means the primary model wants to go long.
- `-1` means the primary model wants to go short.
- `0` means no trade.

Our metamodel does not predict market direction from scratch. It predicts the probability that the primary signal should be followed.

A simple analogy:

- Primary signal = someone says "buy" or "sell".
- Metamodel = asks "is this a good situation to listen to that signal?"

So this project is meta-labeling, not direct return prediction.

## 2. What Is One Training Sample?

One row is one date and one instrument. If `primary_signal` is non-zero, that row is a trade opportunity.

- If `primary_signal = +1`, the primary model wants to go long.
- If `primary_signal = -1`, the primary model wants to go short.
- If `primary_signal = 0`, there is no real trade opportunity.

For the model:

`X` includes:

- `primary_signal`
- lagged OHLCV features
- momentum and return features
- volatility and range features
- technical indicators
- volume, liquidity, and open-interest features
- cross-sectional rank features
- GMM and PCA regime/context features
- signal-history features

`y` means:

- `1` = the primary signal was worth following under the labeling rule.
- `0` = the primary signal was not worth following under the labeling rule.

The target is a triple-barrier meta-label. Do not call it "true ground truth". It is the selected label specification for the coursework task.

## 3. Triple-Barrier Labeling Explained Simply

Triple-barrier labeling creates `y`.

For each non-zero primary signal, we look forward for a fixed number of trading days. We set three barriers:

1. A profit-taking barrier.
2. A stop-loss barrier.
3. A vertical time barrier.

If the trade hits profit-taking first, `y = 1`.

If the trade hits stop-loss first, `y = 0`.

If time runs out before profit-taking is hit, `y = 0` in our binary meta-label setup.

Long example:

- Entry price = `100`.
- `primary_signal = +1`.
- Lagged volatility = `2%`.
- Multiplier = `1.5`.
- Barrier width = `3%`.
- Profit-taking price = `103`.
- Stop-loss price = `97`.
- If price reaches `103` first, `y = 1`.
- If price reaches `97` first, `y = 0`.

Short example:

- Entry price = `100`.
- `primary_signal = -1`.
- Profit-taking for a short means price falls.
- Profit-taking price = `97`.
- Stop-loss price = `103`.
- If price reaches `97` first, `y = 1`.
- If price reaches `103` first, `y = 0`.

This label is not raw future return. It asks whether following the primary signal worked under a simple risk-management rule.

## 4. Why Do We Use Lagged Features?

We must avoid look-ahead bias.

A feature used on date `t` should only use information available before the trade decision. The model cannot peek at tomorrow's price when making today's decision.

That is why rolling volatility, returns, technical indicators, signal-history features, GMM/PCA context, and cross-sectional features are lagged or fitted only on the appropriate training history.

## 5. Feature Engineering

The project uses several feature groups:

1. Return and momentum: captures recent price direction.
2. Volatility and range: captures how noisy or risky the market is.
3. Technical indicators: RSI, MACD, Bollinger-style indicators, and stochastic indicators.
4. Volume, liquidity, and open interest: captures market activity and participation.
5. Cross-sectional ranks: compares one instrument against the others on the same date.
6. GMM regime features: captures soft market regime states.
7. PCA latent features: compresses correlated OHLCV information into lower-dimensional components.
8. Signal-history features: captures whether similar recent primary signals have worked or failed.

Features are grouped because many financial features are correlated. The report therefore uses cluster-level feature importance, not only individual feature importance.

## 6. Models We Compare

The coursework expects several model families. We compare:

1. Logistic Regression.
2. Tree-based models such as Random Forest and Extra Trees.
3. Gradient boosting style models.
4. MLP neural network.

Logistic Regression is stable and interpretable. Tree-based models can capture nonlinear relationships. MLP can capture nonlinear interactions, but it can also overfit.

The final frozen model is:

```text
Calibrated 0.50 Logistic + 0.50 signal-history MLP probability blend
```

The Logistic model provides a stable anchor. The signal-history MLP adds nonlinear information from recent signal behavior. Sigmoid calibration improves probability quality.

## 7. Why This Final Model Was Selected

The final model was selected using guardrails rather than chasing the highest public score.

The calibrated Logistic-MLP blend improved the final public sanity-check performance compared with simpler reference models while staying explainable enough for the report.

Several extensions were tested but not promoted:

- HMM extension: implemented as an extension, but disabled in the final run.
- Label-search challenger: tested different triple-barrier label specifications, but not promoted because public sanity-check guardrails were not strong enough.
- Advanced challengers: stacking, autoencoder, and other complex challengers were tested but did not pass guardrails.
- Universe ablation: some reduced universes improved selected averages, but they risk cherry-picking and may not satisfy the full deliverable without neutral fills.
- Feature pruning: validation-driven pruning improved some validation metrics, but did not pass promotion guardrails.

The final submission remains the frozen calibrated Logistic-MLP probability blend.

## 8. Project Folder Structure

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

In this local workspace, `coursework/outputs` points to the shared top-level `outputs/` folder so the final prediction file is not duplicated.

## 9. How To Run The Code

From the project root:

```bash
python3 coursework_metamodel.py
```

Or from inside the `coursework/` folder:

```bash
python3 coursework_metamodel.py
```

The runner validates the existing frozen final prediction file and writes audit information. It does not retune the final submission.

## 10. Outputs That Matter

The final submission file is:

```text
outputs/metamodel_predictions.csv
```

It must have exactly these columns:

```text
date,instrument,prediction
```

The final prediction hash is:

```text
c5c7ca869d905b384ef3c9072c3377e0f43c7a7ad03c9125aa062077f0f9b369
```

Other useful final outputs:

- `outputs/evaluation_summary.csv`
- `outputs/model_comparison.csv`
- `outputs/cluster_importance.csv`
- `outputs/threshold_analysis.csv`
- `outputs/baseline_comparison.csv`
- `outputs/final_integration_audit.txt`
- `outputs/final_github_preflight_audit.txt`

Archived files are for appendix experiments only. They are not the final submission.

## 11. What Not To Change Before Submission

Do not overwrite `outputs/metamodel_predictions.csv`.

Do not describe the final model as baseline logistic only.

Do not enable HMM.

Do not promote label-search results.

Do not promote advanced challengers.

Do not promote universe or feature-pruning robustness results.

Do not tune on public 2022H1 or hidden test data.

Do not use future OHLCV features.

Do not call triple-barrier labels "true ground truth".
