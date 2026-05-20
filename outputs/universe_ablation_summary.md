# Universe Ablation Summary

This is a robustness appendix. It does not overwrite `outputs/metamodel_predictions.csv`.

Instrument universes were evaluated with the frozen final model style: calibrated 0.50 Logistic + 0.50 signal-history MLP probability blend. Because the project trains models per instrument, a reduced universe is evaluated by retaining only the relevant instrument-level results.

Asset-class mapping used from the coursework configuration:
- Equity index futures: `es1s`, `nq1s`, `fesx1s`.
- Energy: `cl1s`, `ho1s`, `rb1s`, `ng1s`.
- Metals: `gc1s`, `si1s`, `hg1s`, `pl1s`.

- Best validation mean ROC AUC: `drop_validation_weak_instruments` at `0.5903`.
- Best public 2022H1 mean ROC AUC sanity check: `metals_only` at `0.6372`.
- Reduced universes are useful diagnostics, but a direct reduced-universe prediction file would not satisfy the full all-date/all-instrument deliverable unless dropped instruments were explicitly filled with neutral probabilities.
- `drop_unstable_instruments` uses public 2022H1 deterioration information and is diagnostic only.

## Results

| Universe | Instruments | Validation AUC | Validation F1 | Public AUC | Public F1 | Guardrails |
|---|---:|---:|---:|---:|---:|---|
| `all_11_instruments` | 11 | 0.5140 | 0.7273 | 0.5830 | 0.6255 | False |
| `equity_index_futures_only` | 3 | 0.5443 | 0.6972 | 0.4611 | 0.6750 | False |
| `energy_only` | 4 | 0.5379 | 0.7672 | 0.6327 | 0.5308 | False |
| `metals_only` | 4 | 0.4674 | 0.7100 | 0.6372 | 0.6832 | False |
| `drop_validation_weak_instruments` | 6 | 0.5903 | 0.7090 | 0.5774 | 0.6693 | False |
| `drop_unstable_instruments` | 9 | 0.4961 | 0.7213 | 0.6198 | 0.6349 | False |

Conclusion: universe reduction remains a robustness appendix unless it passes all promotion guardrails and can still produce a complete coursework-valid prediction file.
