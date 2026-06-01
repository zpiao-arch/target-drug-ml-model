# Target-Drug ML Model Package

本压缩包是当前项目中“靶点-药物匹配机器学习模型”的独立交付包，包含源码、已训练模型、训练数据、评估结果和复现说明。

## 1. 模型定位

该模型用于判断一个“药物结构 + 靶点描述”组合是否符合 DrugCentral 中已知 drug-target 关系模式，并输出候选药物排序分数。

它的产品用途是：

- 已知药物重定位候选召回
- 疾病/靶点输入后的 DrugCentral 药物排名
- Stage 3 候选初筛与 Stage 4 结构验证之间的排序桥梁
- 给智能体分析提供可追溯的机器学习证据

它不能用于直接声称药效、毒性、安全性、剂量或临床可用性。

## 2. 包内目录

```text
code/
  ai_mol_loop/
    target_match_model.py              # 神经匹配器与通用特征/数据构造函数
    test_target_match_model.py         # 模型相关单元测试快照
  scripts/
    build_extra_trees_model_ppt.py     # ExtraTrees 训练、评估、图表与 PPT 生成脚本
    extra_trees_ranker.py              # 独立 ExtraTrees 药物排序脚本
  webapp_tests/
    test_ml_rank_api.py                # 前端接入 API 测试
    test_stage3_frontend_contract.py   # Stage 3 前端契约测试

models/
  target_match_drugcentral/
    extra_trees/
      target_match_extra_trees.joblib  # 当前前端接入的一键机器学习模型
      metadata.json                    # 模型参数、数据规模、benchmark 摘要
      training_report.md
    target_match_model.pt              # 早期神经匹配器模型
    metadata.json                      # 神经匹配器元数据
    benchmark/                         # benchmark 记录
    *.csv / *.json                     # 验证和示例预测文件

data/
  drugcentral/
    2021_09_01/
      structures.smiles.tsv            # 4099 个 DrugCentral 药物结构
      drug.target.interaction.tsv.gz   # 18321 条 drug-target 正关系来源
    static/
      FDA_Approved.csv
      EMA_Approved.csv
      PMDA_Approved.csv
      FDA_EMA_PMDA_Approved.csv

experiments/
  model_comparison_benchmark.csv       # ExtraTrees / HistGradientBoosting / CatBoost 对比
  rerun_extra_trees_metrics.csv        # 三种划分下 ExtraTrees 复跑指标
  learning_curve_random_pair.csv       # 树数量学习曲线
  multi_disease_target_rankings.csv    # 多疾病 Top-K 排名示例
  experiment_summary.json              # PPT 实验汇总

docs/
  target_drug_match_model_report_2026-05-30.md
  target_drug_match_model_optimization_2026-05-30.md
```

## 3. 数据与样本划分

模型数据来自本地 DrugCentral：

- 药物结构：4099
- 靶点：2729
- 正样本：18321 条 DrugCentral 已知 drug-target 关系
- 负样本：18321 条确定性负采样
- 总样本：36642

ExtraTrees benchmark 使用三种 80/20 留出划分：

- `random_pair_split`：随机 drug-target pair 划分
- `target_group_split`：按靶点分组，验证集靶点训练时不可见
- `drug_group_split`：按药物结构分组，验证集药物训练时不可见

当前部署模型在完成 benchmark 后使用全量 36642 条样本重训，元数据中记录为 `trained_on_all_examples: true`。

## 4. 主要指标

当前主模型为 `sklearn.ensemble.ExtraTreesClassifier`：

- n_estimators = 180
- min_samples_leaf = 2
- max_features = sqrt
- random_state = 13

评估结果：

| 划分 | ROC-AUC | Accuracy | F1 |
| --- | ---: | ---: | ---: |
| random_pair_split | 0.9418 | 0.8798 | 0.8749 |
| target_group_split | 0.8930 | 0.7226 | 0.6218 |
| drug_group_split | 0.8931 | 0.8000 | 0.7720 |

## 5. 快速使用

在本包根目录运行：

```bash
python3 code/scripts/extra_trees_ranker.py \
  --target-query "Influenza A neuraminidase oseltamivir zanamivir peramivir P03468" \
  --top 10 \
  --output experiments/influenza_na_rank_from_package.csv
```

输出 CSV 字段包括：

- `rank`
- `struct_id`
- `drug_name`
- `smiles`
- `score`
- `model_probability`
- `known_target_similarity`
- `evidence_target`
- `evidence_organism`
- `confidence`

## 6. 依赖

推荐 Python 3.9+。核心依赖：

```bash
pip install numpy pandas scikit-learn joblib rdkit-pypi torch
```

如果已有 conda 环境，建议用 conda 安装 RDKit：

```bash
conda install -c conda-forge rdkit scikit-learn joblib numpy pandas pytorch
```

## 7. 合规边界

本模型输出的是计算筛选优先级，不代表真实结合强度、生物活性、药效、毒性、安全性、剂量或临床可用性。

后续需要结合 docking、Pose QC、对照校准、ChEMBL/BindingDB 活性数据和实验验证。
