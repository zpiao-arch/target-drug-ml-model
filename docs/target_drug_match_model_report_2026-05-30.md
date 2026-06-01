# 靶点-药物匹配模型阶段报告

生成时间：2026-05-30

## 目标

本阶段完成了一个本地靶点-药物匹配深度学习模型。它用于把“某个靶点/病毒蛋白/口袋描述”映射到本地已知药物库，并输出候选药物优先级。

边界：该模型只提供计算筛选排序信号，不证明生物活性、药效、毒性、安全性、剂量或临床可用性。

## 数据来源

当前使用本地 DrugCentral 数据：

- 结构文件：`data_lake/drugcentral/2021_09_01/structures.smiles.tsv`
- 靶点关系文件：`data_lake/drugcentral/2021_09_01/drug.target.interaction.tsv.gz`

训练集构建：

- 正样本：DrugCentral 中已有的 drug-target interaction。
- 负样本：对每个正样本进行确定性 target negative sampling。
- 药物数量：4099
- 靶点数量：2729
- 正样本关系：18321
- 负样本关系：18321
- 总样本：36642

## 模型结构

模型文件：`ai_mol_loop/target_match_model.py`

输入特征：

- 药物侧：RDKit Morgan fingerprint 512 维，加 8 个 RDKit 标量描述符。
- 靶点侧：靶点名称、靶点类别、accession、gene、SwissProt、作用类型、organism 等文本 hash 特征 256 维。

网络结构：

- 双塔 PyTorch 模型。
- 药物塔和靶点塔分别编码。
- 拼接 drug embedding、target embedding、绝对差值和乘积。
- 二分类输出 drug-target match probability。

排序校准：

- `model_probability` 来自神经网络。
- `known_target_similarity` 来自输入靶点描述和本地已知靶点证据的文本相似度。
- `match_probability` 为最终排序分，已做证据优先校准。

## 训练结果

训练命令：

```bash
python3 ai_mol_loop/target_match_model.py train \
  --epochs 8 \
  --negative-ratio 1 \
  --batch-size 512 \
  --hidden-dim 128 \
  --model-dir ai_mol_loop/models/target_match_drugcentral
```

指标：

- Train accuracy：0.8953
- Train ROC-AUC：0.9630
- Validation accuracy：0.8413
- Validation ROC-AUC：0.9253

模型产物：

- `ai_mol_loop/models/target_match_drugcentral/target_match_model.pt`
- `ai_mol_loop/models/target_match_drugcentral/metadata.json`
- `ai_mol_loop/models/target_match_drugcentral/training_report.md`

## 甲流 NA 验证示例

预测命令：

```bash
python3 ai_mol_loop/target_match_model.py predict \
  --model-dir ai_mol_loop/models/target_match_drugcentral \
  --target "Influenza A H1N1 neuraminidase NA P03468 oseltamivir pocket" \
  --top 10 \
  --output-csv ai_mol_loop/models/target_match_drugcentral/influenza_na_predictions_top10_verify.csv
```

Top 4 结果：

| rank | drug_name | match_probability | model_probability | known_target_similarity | evidence_target |
|---:|---|---:|---:|---:|---|
| 1 | oseltamivir | 0.526366 | 0.285050 | 0.606804 | P03468 / Neuraminidase |
| 2 | zanamivir | 0.514998 | 0.239579 | 0.606804 | P03468 / Neuraminidase |
| 3 | peramivir | 0.484401 | 0.117191 | 0.606804 | P03468 / Neuraminidase |
| 4 | rutoside | 0.449370 | 0.051672 | 0.581936 | P03469 / Neuraminidase |

这个结果说明：当输入甲流 H1N1 NA / P03468 / oseltamivir pocket 时，模型能把 DrugCentral 中已知 NA 相关药物排到最前面。它可作为后续“已知药物基准”和“老药重定位候选池”的入口。

## 测试记录

测试命令：

```bash
python3 -m unittest ai_mol_loop.tests.test_target_match_model
```

结果：

- 2 个单元测试通过。
- 覆盖训练样本构建、小型 fixture 训练、靶点输入预测和边界声明输出。

## 如何使用

训练模型：

```bash
python3 ai_mol_loop/target_match_model.py train \
  --model-dir ai_mol_loop/models/target_match_drugcentral
```

预测某个靶点可优先筛哪些已知药物：

```bash
python3 ai_mol_loop/target_match_model.py predict \
  --model-dir ai_mol_loop/models/target_match_drugcentral \
  --target "目标描述，例如 Influenza A neuraminidase P03468" \
  --top 25 \
  --output-csv outputs/target_match_predictions.csv
```

## 后续接入建议

1. 接 FastAPI：
   - `POST /api/target-match/train`
   - `POST /api/target-match/predict`

2. 接前端：
   - 在“已知药物库”或“药物重定位”页面增加靶点输入框。
   - 展示 match_probability、model_probability、known_target_similarity、证据靶点、证据来源和合规边界。

3. 接 Stage 3/4：
   - Stage 3 把 top 药物作为 seed drug。
   - Stage 4 把 top 药物进入 docking / positive control / decoy 同池校验。

4. 后续数据增强：
   - ChEMBL 活性数据。
   - BindingDB 亲和力数据。
   - PDB 共晶结构和配体口袋。
   - PubChem BioAssay。

## 风险和限制

- 负样本是采样构造，不等同于实验阴性样本。
- 文本 hash 靶点特征能处理泛化输入，但不等同于蛋白序列或三维结构建模。
- 当前概率更适合做排序，不适合解释为真实结合概率。
- 任何候选都必须继续通过 docking、pose QC、对照校准、回顾性 benchmark 和实验验证。
