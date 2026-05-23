# Final Submission Checklist

## Final Model

- Final model: `Calibrated 0.50 Logistic + 0.50 signal-history MLP probability blend`
- HMM enabled: `False`
- Label-search challenger promoted: `False`
- Advanced challengers promoted: `False`
- Universe and feature-pruning robustness promoted: `False`

## Final Prediction File

- Final prediction file path: `outputs/metamodel_predictions.csv`
- Expected columns: `date,instrument,prediction`
- Prediction values must be in `[0, 1]`
- No missing values
- No duplicate `date,instrument` rows
- Zero primary signals receive neutral probability `0.5` where required

Final SHA256:

```text
c5c7ca869d905b384ef3c9072c3377e0f43c7a7ad03c9125aa062077f0f9b369
```

## Reproduce / Validate

Run from the project root:

```bash
python3 coursework_metamodel.py
```

Preflight audit file:

```text
outputs/final_github_preflight_audit.txt
```

Final integration audit:

```text
outputs/final_integration_audit.txt
```

## Files To Use For Submission

Use:

```text
outputs/metamodel_predictions.csv
```

Do not use challenger files in `outputs/archive/`.

Do not use label-search challenger predictions.

Do not use advanced challenger predictions.

Do not replace the frozen final prediction file unless there is a confirmed formatting bug.

## Optional Strategy Bonus

Strategy construction is optional bonus work, not the required metamodel deliverable.

Run:

```bash
python3 coursework_strategy_bonus.py
```

Optional output:

```text
outputs/strategy_weights.csv
```

Expected columns:

```text
date,instrument,weight
```

The current optional strategy method is `soft_allocation`. It uses primary-signal direction and metamodel probability confidence. It does not change `outputs/metamodel_predictions.csv`.

Optional bonus checks:

- `outputs/strategy_weights.csv` has exact columns `date,instrument,weight`
- no missing weights
- no duplicate `date,instrument` rows
- zero primary signals have zero weight
- weight signs agree with primary signals
- daily gross exposure is not above `1.0`
- single-instrument max absolute weight cap is respected
- HMM remains disabled
- archived challengers are not promoted
- final metamodel prediction hash remains `c5c7ca869d905b384ef3c9072c3377e0f43c7a7ad03c9125aa062077f0f9b369`

## GitHub Submission Checklist

- README explains the project purpose on the first screen.
- `report_summary.md` matches the final model and does not call it baseline logistic only.
- `docs/PROJECT_GUIDE_FOR_TEAMMATES.md` explains meta-labeling and triple-barrier labels in plain English.
- `docs/PIPELINE_FLOW.md` maps pipeline steps to source files.
- `docs/FINAL_SUBMISSION_CHECKLIST.md` lists the frozen model, file, and hash.
- Python modules compile.
- Notebooks parse as valid JSON.
- No `.env`, credentials, tokens, private keys, or hidden test labels are committed.
- No cache files such as `__pycache__`, `.DS_Store`, or notebook checkpoints are committed.
- Archive folders are ignored unless intentionally restored.
- Raw input files are included only if coursework policy allows sharing them.

## GitHub Repository

- Repository URL: `https://github.com/Czzzzzzzzy/Algorithmic-Trading-Project.git`
- Branch: `main`
- Final command to reproduce:

```bash
python3 coursework_metamodel.py
```

Final warning: the final submission should use `outputs/metamodel_predictions.csv`; archived challengers are appendix material only.
