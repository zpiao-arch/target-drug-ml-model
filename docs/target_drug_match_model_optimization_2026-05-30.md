# 靶点-药物匹配模型优化记录

日期：2026-05-30

## 问题

第一版模型的二分类验证指标较高，但真实使用时更关心“输入某个靶点后，已知相关药物能否排在前面”。原版本存在三个主要问题：

1. 只输入 accession，例如 `P03468` 时，已知 NA 药物不能稳定排前。
2. 靶点名证据层使用了带三字母片段的 tokenizer，容易把无关靶点误判成高相似。
3. SwissProt 后缀和单字母 token 会造成假精确命中，例如 `_HUMAN`、`Influenza A` 中的 `A`。

## 修复内容

代码位置：`ai_mol_loop/target_match_model.py`

1. 增加完整标识符匹配：
   - accession
   - gene
   - SwissProt
   - compact identifier

2. 区分两类特征：
   - 神经网络仍使用 hash/ngram 文本特征做泛化。
   - evidence boost 只使用真实词和完整标识符，避免假相似。

3. 增加证据优先校准：
   - accession/gene/SwissProt 精确命中优先级最高。
   - 靶点名相似不能压过精确 accession 命中。

4. 增加推理缓存：
   - 药物 RDKit 特征缓存。
   - 靶点证据记录缓存。
   - 批量 PyTorch 推理。

5. 新增 Top-K 检索评估：
   - `python3 ai_mol_loop/target_match_model.py evaluate ...`
   - 指标包括 hit@1、hit@3、hit@5、hit@10、MRR。

## 测试覆盖

测试文件：`ai_mol_loop/tests/test_target_match_model.py`

新增覆盖：

- `P03468` 这类 accession 短查询必须优先返回 oseltamivir / zanamivir。
- unrelated trigram overlap 不能产生高证据分。
- 单字母 token 不能触发 exact match。
- SwissProt 物种后缀 `_HUMAN` 不能触发 exact match。
- 精确证据必须压过模糊证据。
- 预测资产必须预计算 evidence records。
- Top-K 检索评估必须输出 hit@K 和 MRR。

验证命令：

```bash
python3 -m unittest ai_mol_loop.tests.test_target_match_model
```

结果：

- 9 tests passed

## 当前效果

甲流 NA 示例：

输入：

```text
Influenza A H1N1 neuraminidase NA P03468 oseltamivir pocket
```

Top 6：

| rank | drug | evidence | score |
|---:|---|---|---:|
| 1 | oseltamivir | P03468 / Neuraminidase | 0.928505 |
| 2 | zanamivir | P03468 / Neuraminidase | 0.923958 |
| 3 | peramivir | P03468 / Neuraminidase | 0.911719 |
| 4 | laninamivir octanoate hydrate | B4URF0 / Neuraminidase | 0.808300 |
| 5 | quercetin | B4URF0 / Neuraminidase | 0.803904 |
| 6 | rutoside | B4URF0 / Neuraminidase | 0.803594 |

50 个本地 DrugCentral 靶点抽样 Top-K 检索结果：

| metric | before optimization | after optimization |
|---|---:|---:|
| hit@1 | 0.32 | 0.70 |
| hit@3 | 0.48 | 0.80 |
| hit@5 | 0.60 | 0.92 |
| hit@10 | 0.74 | 0.98 |
| MRR | 0.4295 | 0.7756 |

评估文件：

- `ai_mol_loop/models/target_match_drugcentral/topk_eval_50_optimized.json`
- `ai_mol_loop/models/target_match_drugcentral/influenza_na_predictions_optimized.csv`

## 边界

这些指标衡量的是“本地 DrugCentral 已知关系检索能力”，不是药效验证。模型输出只能作为计算筛选排序信号，仍需继续进入 docking、pose QC、对照校准、回顾性 benchmark 和实验验证。

## 后续优化

1. 用 ChEMBL / BindingDB 加入亲和力标签，训练回归或排序模型。
2. 增加蛋白序列/结构 embedding，而不是只用靶点文本。
3. 引入 hard negative sampling，专门区分同类靶点、同物种靶点和同 gene family。
4. 把 Top-K evaluation 接入前端环境检查页或模型诊断页。
