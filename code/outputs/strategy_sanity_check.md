# Strategy Sanity Check

This check validates the optional strategy bonus outputs. It does not modify the required metamodel prediction file.

## Summary

- Final metamodel prediction hash: `c5c7ca869d905b384ef3c9072c3377e0f43c7a7ad03c9125aa062077f0f9b369`.
- Hash matches frozen final: `True`.
- Selected strategy weights file: `/Users/ziyunjameschen/Downloads/IC_本地/algorithmic trading/outputs/strategy_weights.csv`.
- Rows: `1408`.
- Dates: `128`.
- Instruments: `11`.
- Min weight: `-0.25000000`.
- Max weight: `0.16666650`.
- Mean gross exposure: `0.976562`.
- Max gross exposure: `0.999999`.

## Checks

- Weight at date t uses only `prediction_t` and `primary_signal_t`: `PASS` by construction in `coursework/src/strategy.py`; selected weights validate against same-date panel rows.
- Return is computed from close_t to close_t+1: `PASS`; `strategy_daily_returns.csv` contains `date` and `next_date`, with weights applied from date to next_date.
- No future price is used to form weight_t: `PASS`; price data is used only for return calculation, not for position direction or confidence.
- `primary_signal = 0` always has `weight = 0`: `True`; count `0`.
- Weight sign agrees with primary signal direction: `True`; count `0`.
- Daily gross exposure <= 1: `True`; max `0.999999`.
- Max absolute instrument cap respected: `True`; cap `0.25`, observed `0.250000`.
- Transaction cost formula equals `turnover x cost_bps / 10000`: `True`.
- Holidays / missing close handling: prediction dates that are market holidays use the most recent available close at or before date t for return alignment. This is documented as a close-to-close assumption and does not use future prices to form weights.
- `strategy_weights.csv` columns exactly `date,instrument,weight`: `True`.
- No missing strategy weight values: `True`; count `0`.
- No duplicate date-instrument rows: `True`; count `0`.
- No hidden 2022H2 data used: `True`; max weight date `2022-06-30`, max return next_date `2022-06-30`.
- Required metamodel prediction hash unchanged: `True`.

## Interpretation

The optional strategy is an illustrative public-2022H1 close-to-close backtest. It is not a live trading system and should be interpreted cautiously because the public test window has only 128 prediction dates.
