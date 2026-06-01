# Target-Drug Match Model Benchmark

## Dataset
- Examples: 36642
- Positive pairs: 18321
- Structures: 4099
- Targets: 2729

## Results
| split | model | ROC-AUC | PR-AUC | Acc | Bal Acc | F1 | seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| random_pair_split | current_target_match_net_existing | 0.9253 |  | 0.8413 |  |  |  |
| random_pair_split | logistic_l2 | 0.7537 | 0.7329 | 0.6851 | 0.6851 | 0.6838 | 33.01 |
| random_pair_split | random_forest | 0.9189 | 0.9229 | 0.8421 | 0.8421 | 0.8331 | 3.21 |
| target_group_split | logistic_l2 | 0.6908 | 0.6556 | 0.6463 | 0.6455 | 0.6237 | 26.91 |
| target_group_split | random_forest | 0.8666 | 0.8662 | 0.7108 | 0.7064 | 0.6022 | 2.91 |
| drug_group_split | logistic_l2 | 0.749 | 0.7399 | 0.6889 | 0.6889 | 0.6951 | 29.59 |
| drug_group_split | random_forest | 0.8912 | 0.8988 | 0.8153 | 0.8153 | 0.7995 | 2.45 |

## Best Traditional Model Per Split
| split | model | ROC-AUC | PR-AUC | Acc | Bal Acc | F1 | seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| random_pair_split | current_target_match_net_existing | 0.9253 |  | 0.8413 |  |  |  |
| target_group_split | random_forest | 0.8666 | 0.8662 | 0.7108 | 0.7064 | 0.6022 | 2.91 |
| drug_group_split | random_forest | 0.8912 | 0.8988 | 0.8153 | 0.8153 | 0.7995 | 2.45 |

## Interpretation
- Random pair split is optimistic because drugs and targets can appear on both sides.
- Target group split is the most relevant stress test for new target generalization.
- FDA/EMA/PMDA approval status is not used as a training label; it remains candidate-library metadata only.
- Boundary: Computational target-drug matching only. This model does not prove biological activity, potency, efficacy, toxicity, safety, dosing, or clinical usefulness.
