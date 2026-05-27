# BUSI70575 Code Package

Preferred and only packaged workflow: open `coursework/notebooks/00_final_reproducible_pipeline.ipynb` and run all cells. The notebook explains the modelling flow in markdown while calling the reusable code in `coursework/src`.

This public GitHub copy does not include the official raw coursework input CSVs. To fully rerun the notebook, place `ohlcv_data.csv` and `primary_signals.csv` in this `code/` directory. The frozen/supporting outputs used by the submission checks are included.

Use a Python environment with the packages in `coursework/requirements.txt` installed. In this workspace the verified interpreter is `/opt/anaconda3/bin/python3`.

The command-line wrapper scripts are intentionally not included in this cleaned deliverable package. The notebook is the readable execution entry point; `coursework/src` is the implementation library.

For a genuinely new prediction period, use the final section of the notebook. It calls `coursework.src.prediction.predict_new_period` with the same arguments shown here:

```bash
predict_new_period(new_config, output_path='outputs/new_period_predictions.csv')
```
