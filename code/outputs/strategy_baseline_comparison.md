# Strategy Baseline Comparison

This compares the selected optional strategy with simple baselines. The required metamodel prediction file is unchanged.

- Selected strategy: `soft_allocation`.
- Blind baseline: `blindly_follow_primary_signal_equal_weight`.
- Selected strategy improves over blind baseline at 2 bps by Sharpe: `0.0000`.
- Selected strategy improves over blind baseline at 2 bps by cumulative return: `0.0000`.
- Overall improvement over blind baseline: `False`.

## Metrics

| Method | Cost bps | Cum. Return | CAGR | Ann. Vol | Sharpe | Sortino | Max DD | Avg Hold | Avg Turnover | Active Days | Avg Active Inst. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `blindly_follow_primary_signal_equal_weight` | 0 | 0.3577 | 0.8344 | 0.1407 | 4.3878 | 7.4799 | -0.0347 | 4.00 | 0.5427 | 127 | 7.89 |
| `blindly_follow_primary_signal_equal_weight` | 2 | 0.3391 | 0.7850 | 0.1408 | 4.1888 | 7.0941 | -0.0350 | 4.00 | 0.5427 | 127 | 7.89 |
| `threshold_filter_050` | 0 | 0.3200 | 0.7347 | 0.1440 | 3.9015 | 6.1879 | -0.0388 | 3.62 | 0.6133 | 127 | 7.09 |
| `threshold_filter_050` | 2 | 0.2996 | 0.6820 | 0.1441 | 3.6829 | 5.8184 | -0.0403 | 3.62 | 0.6133 | 127 | 7.09 |
| `threshold_filter_055` | 0 | 0.3045 | 0.6946 | 0.1513 | 3.5643 | 6.0998 | -0.0441 | 2.90 | 0.5705 | 122 | 3.42 |
| `threshold_filter_055` | 2 | 0.2858 | 0.6467 | 0.1513 | 3.3756 | 5.7213 | -0.0448 | 2.90 | 0.5705 | 122 | 3.42 |
| `threshold_filter_060` | 0 | -0.0202 | -0.0397 | 0.0332 | -1.2046 | -0.6585 | -0.0251 | 2.62 | 0.0295 | 21 | 0.17 |
| `threshold_filter_060` | 2 | -0.0209 | -0.0411 | 0.0332 | -1.2462 | -0.7295 | -0.0255 | 2.62 | 0.0295 | 21 | 0.17 |
| `confidence_linear` | 0 | 0.3505 | 0.8152 | 0.1538 | 3.9583 | 7.1645 | -0.0413 | 3.62 | 0.6111 | 127 | 7.09 |
| `confidence_linear` | 2 | 0.3297 | 0.7603 | 0.1539 | 3.7556 | 6.7741 | -0.0430 | 3.62 | 0.6111 | 127 | 7.09 |
| `confidence_scaled` | 0 | 0.3505 | 0.8152 | 0.1538 | 3.9583 | 7.1645 | -0.0413 | 3.62 | 0.6111 | 127 | 7.09 |
| `confidence_scaled` | 2 | 0.3297 | 0.7603 | 0.1539 | 3.7556 | 6.7741 | -0.0430 | 3.62 | 0.6111 | 127 | 7.09 |
| `soft_allocation` | 0 | 0.3577 | 0.8344 | 0.1407 | 4.3878 | 7.4799 | -0.0347 | 4.00 | 0.5427 | 127 | 7.89 |
| `soft_allocation` | 2 | 0.3391 | 0.7850 | 0.1408 | 4.1888 | 7.0941 | -0.0350 | 4.00 | 0.5427 | 127 | 7.89 |

## Caution

These are public-2022H1 close-to-close backtest results. The window has only 128 prediction dates, so high Sharpe ratios should be interpreted cautiously. Transaction costs are simplified as turnover times bps divided by 10000.
