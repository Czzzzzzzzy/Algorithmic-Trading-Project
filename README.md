# BUSI70575 Deliverables

Generated: 2026-05-27
Official coursework page: https://hm-ai.github.io/BUSI70575/coursework/

## Contents

- `report/BUSI70575_Report.pdf`: final report.
- `required/metamodel_predictions.csv`: required probability deliverable.
- `optional_strategy/strategy_weights.csv`: optional strategy-weight deliverable.
- `code/`: notebook-first reproducible workflow, clean `coursework/src` package, and frozen/supporting outputs for reruns.
- `supporting_outputs/`: metrics, audits, feature importance, threshold analysis, and strategy evidence.
- `sha256_manifest.txt`: hashes for all packaged files.

## Reproduce

Preferred reproduction path: open `code/coursework/notebooks/00_final_reproducible_pipeline.ipynb` and run all cells.

This public GitHub copy intentionally omits the official raw coursework files:

- `ohlcv_data.csv`
- `primary_signals.csv`

To fully rerun the notebook, place those two files in `code/` after cloning. The submitted prediction, report, strategy weights, metrics, and audits are included.

Use a Python environment with `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `jupyter` installed. This workspace was verified with `/opt/anaconda3/bin/python3`.

The package intentionally omits CLI wrapper scripts. The notebook calls the reusable implementation directly from `code/coursework/src`.

Required prediction SHA256: `c5c7ca869d905b384ef3c9072c3377e0f43c7a7ad03c9125aa062077f0f9b369`
