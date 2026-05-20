# Feature-Pruning Summary

Feature-cluster pruning uses validation-period cluster permutation importance as the pruning signal. Public 2022H1 is reported only as a sanity check.

- Best validation mean ROC AUC: `drop_near_zero_importance_clusters` at `0.6028`.
- HMM features remain disabled.
- Pruned feature sets may improve interpretability by using fewer clusters, but they are not promoted unless they pass all guardrails.

| Feature Set | Features | Validation AUC | Validation F1 | Public AUC | Public F1 | Interpretability | Guardrails |
|---|---:|---:|---:|---:|---:|---|---|
| `full_features` | 110 | 0.5458 | 0.7252 | 0.4606 | 0.6214 | False | False |
| `drop_negative_importance_clusters` | 62 | 0.5530 | 0.7323 | 0.5505 | 0.5977 | True | False |
| `drop_near_zero_importance_clusters` | 48 | 0.6028 | 0.7369 | 0.5604 | 0.6122 | True | False |
| `keep_top_5_clusters` | 48 | 0.6028 | 0.7369 | 0.5604 | 0.6122 | True | False |
| `keep_economic_core_clusters` | 93 | 0.5302 | 0.7285 | 0.5793 | 0.6279 | True | False |

Conclusion: feature pruning is reported as robustness evidence; final predictions remain frozen.
