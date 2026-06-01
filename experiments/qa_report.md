# QA Report

- PPTX: `/Users/ruiny_park/Documents/药物分子证明/deliverables/extra_trees_model_ppt_20260601/ViroMol_Compass_ExtraTrees模型汇报.pptx`
- Slide count: 22
- Figures generated: 8
- Verification: PPTX reopened successfully with python-pptx.
- ROC curves, learning curve, feature importance, and SHAP plots were generated from local rerun experiments.
- Multi-disease ranking panel was generated from local DrugCentral structures using ExtraTrees plus known-target evidence calibration.
- Model comparison panel was generated from an existing local benchmark CSV, not manually entered.
- SHAP grouped summary used a fixed-seed validation sample of 40 rows with TreeExplainer approximate mode for reproducible model audit.
- ExtraTrees has no neural-network epoch loss; the deck uses log-loss vs number of trees as the valid training-stability curve.
- Boundary: Computational target-drug matching only; not biological activity, safety, dosing, or clinical evidence.
