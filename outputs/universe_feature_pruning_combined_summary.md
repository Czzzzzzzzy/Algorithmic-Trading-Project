# Universe + Feature-Pruning Combined Summary

Combined tests reuse the best validation-selected pruned feature set and safe universe diagnostics. They do not create or overwrite final predictions.

- Best validation combination: `equity_index_futures_only_plus_pruned_features` at validation AUC `0.6210`.
- Best public sanity-check combination: `all_11_plus_pruned_features` at public AUC `0.5604`.

| Combination | Instruments | Features | Validation AUC | Validation F1 | Public AUC | Public F1 | Guardrails |
|---|---:|---:|---:|---:|---:|---:|---|
| `all_11_plus_pruned_features` | 11 | 48 | 0.6028 | 0.7369 | 0.5604 | 0.6122 | False |
| `equity_index_futures_only_plus_pruned_features` | 3 | 48 | 0.6210 | 0.7273 | 0.3867 | 0.6338 | False |
| `drop_validation_weak_instruments_plus_pruned_features` | 6 | 48 | 0.6157 | 0.7173 | 0.5312 | 0.6292 | False |

Conclusion: no combined diagnostic should replace the frozen final unless all promotion guardrails pass.
