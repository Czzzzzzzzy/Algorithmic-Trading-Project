# Optional Strategy Construction / Bonus Track

This is an optional extension. It does not change the required metamodel prediction file.

Final metamodel prediction file: `/Users/ziyunjameschen/Downloads/IC_本地/algorithmic trading/Deliverables/code/outputs/metamodel_predictions.csv`
Final prediction SHA256: `c5c7ca869d905b384ef3c9072c3377e0f43c7a7ad03c9125aa062077f0f9b369`

## Construction Rule

Weights are built from primary-signal direction multiplied by metamodel probability confidence.
The weight on date `t` earns close-to-close return from `t` to `t+1`; future returns are not used to form weights.
Gross exposure target: `1.0`.
Max absolute weight per instrument: `0.25`.

## Methods Tested

- `threshold_filter_050`
- `threshold_filter_055`
- `threshold_filter_060`
- `confidence_linear`
- `confidence_scaled`
- `soft_allocation`

## Selection

- Best gross Sharpe method: `soft_allocation`.
- Best net Sharpe method at 2 bps: `soft_allocation`.
- Most conservative method: `threshold_filter_060`.
- Recommended optional strategy: `soft_allocation`.
- Reason: Selected the best 2 bps net Sharpe strategy that passed simple drawdown and turnover guardrails.

## Selected Strategy Metrics

| Cost bps | CAGR | Ann. Vol | Sharpe | Sortino | Max Drawdown | Avg Turnover |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.8344 | 0.1407 | 4.3878 | 7.4799 | -0.0347 | 0.5427 |
| 1 | 0.8095 | 0.1407 | 4.2883 | 7.2844 | -0.0348 | 0.5427 |
| 2 | 0.7850 | 0.1408 | 4.1888 | 7.0941 | -0.0350 | 0.5427 |
| 5 | 0.7133 | 0.1411 | 3.8907 | 6.5483 | -0.0355 | 0.5427 |

## Headline

- Gross Sharpe for selected strategy: `4.3878`.
- 2 bps net Sharpe for selected strategy: `4.1888`.

## Baseline Comparison

The selected `soft_allocation` strategy matches the `blindly_follow_primary_signal_equal_weight` baseline under the current conservative cap and gross-exposure normalization. It does not improve over the blind baseline on 2 bps net Sharpe or cumulative return in public 2022H1.

Detailed comparison files:

- `outputs/strategy_baseline_comparison.csv`
- `outputs/strategy_baseline_comparison.md`

## Limitations

This is a simple close-to-close portfolio backtest for the optional competition track. It assumes weights are set at date `t` and earn return from `t` to `t+1`. It does not use hidden 2022H2 data, and it does not alter the required prediction submission. The high Sharpe should be interpreted cautiously because the public test window has only 128 prediction dates.
