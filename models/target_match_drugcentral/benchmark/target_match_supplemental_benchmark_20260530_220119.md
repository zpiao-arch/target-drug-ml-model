# Supplemental Target-Drug Model Benchmark

| split | model | ROC-AUC | PR-AUC | Acc | Bal Acc | Precision | Recall | F1 | seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random_pair_split | extra_trees | 0.9418 | 0.9438 | 0.8798 | 0.8798 | 0.9116 | 0.8412 | 0.8749 | 4.81 |
| random_pair_split | hist_gradient_boosting | 0.9039 | 0.9045 | 0.8213 | 0.8213 | 0.8509 | 0.7789 | 0.8133 | 27.74 |
| random_pair_split | catboost | 0.8868 | 0.8907 | 0.8032 | 0.8032 | 0.8398 | 0.7495 | 0.792 | 10.93 |
| target_group_split | extra_trees | 0.893 | 0.8902 | 0.7226 | 0.7184 | 0.943 | 0.4638 | 0.6218 | 6.71 |
| target_group_split | hist_gradient_boosting | 0.8403 | 0.8406 | 0.7364 | 0.7341 | 0.8196 | 0.5947 | 0.6893 | 36.58 |
| target_group_split | catboost | 0.8361 | 0.833 | 0.7371 | 0.7351 | 0.8049 | 0.6141 | 0.6967 | 13.68 |
| drug_group_split | extra_trees | 0.8931 | 0.901 | 0.8 | 0.8 | 0.8978 | 0.6771 | 0.772 | 5.78 |
| drug_group_split | hist_gradient_boosting | 0.8821 | 0.8866 | 0.8033 | 0.8033 | 0.8452 | 0.7426 | 0.7906 | 24.36 |
| drug_group_split | catboost | 0.8748 | 0.8825 | 0.7918 | 0.7918 | 0.8217 | 0.7455 | 0.7817 | 9.31 |

Boundary: Computational target-drug matching only. This model does not prove biological activity, potency, efficacy, toxicity, safety, dosing, or clinical usefulness.
