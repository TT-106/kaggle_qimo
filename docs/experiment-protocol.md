# 实验与回顾规范

## 固定评价

`prompt-group-5fold-v1` 对官方公开训练集按 `prompt_id` 做五折，每篇文章恰有一个验证折。分类器以 `text` 为输入，`prompt_id` 用于分折和诊断，`id` 用于对齐。同折的 TF-IDF 词表、标准化、分类器与任何需要拟合的变换都从该折的训练侧获得。DeBERTa 的早停使用训练侧内部按题划分的验证集，或者按事先确定的轮数训练。

[data/protocols/](../data/protocols/) 内的 `manifest.json` 记录原始训练文件与分折文件的 SHA-256、标签顺序、数据概况和每折检查，`folds.csv` 是唯一的验证折清单。整体 BA 从完整折外预测计算，每折 BA 用于观察波动。论文表格逐项注明协议与数据来源。旧项目成绩使用各自原有协议引用，线上成绩使用 Kaggle 实际回执。

外部 Open Pool 使用 `open-pool-prompt-group-5fold-v1`。原始与归并题号有关联的记录放进同一折，五折均覆盖 18 类。以发布方的 model 字段评价外部池内的未见题组，语料过滤和分组细节见 [外部协议清单](../data/protocols/open-pool-prompt-group-5fold-v1/manifest.json)。

## 开跑前

为这次运行取新编号，例如 `E001-char-tfidf`。在 `run.json` 写明问题、预期、对照实验编号、本次改动、评价协议与两个文件指纹。记录实际运行命令、解析后的超参数、随机种子、模型与分词器版本、入口代码的提交号或文件指纹，以及依赖、设备和开始时间。涉及外部数据时再记录来源、确切模型版本、与原始数据的去重结果和各类实际训练用量。优先一次验证一个可归因的变化。

## 每次保存

| 文件 | 记录内容 |
|---|---|
| `run.json` | 配置与来源，运行类型、状态、退出码、开始结束时间、训练或编码与推理耗时、所用设备及可测的峰值内存或显存。 |
| `run.log` | 完整控制台输出。异常时保留报错与最后成功的步骤。 |
| `review.md` | 按模板记录观察、结论、依据和处理决定，并链接到这次的产物。 |
| `oof.parquet` | 每篇文章一行，固定列为 `id, prompt_id, fold, y_true, y_pred`。模型输出概率时附上 `p_<官方标签>` 列，记录使用的分类决策规则。 |
| `metrics.json` | 整体折外 `balanced_accuracy`、`macro_f1` 和 `accuracy`。 |
| `by_fold.csv` | 以 `fold` 标出各折，记录 `rows` 和三个指标。 |
| `by_model.csv` | 按标签顺序列出 `model`、`support`（真实篇数）、`predicted`（预测篇数）、`precision`、`recall` 和 `f1`。 |
| `confusion_counts.csv`、`confusion_by_true.csv` | 18×18 原始计数和按真实类归一化的比例。行是真实模型，列是预测模型，顺序沿用协议。 |
| `figures/confusion.png` | 按真实类归一化的混淆矩阵热图。回顾误判时结合原始计数。 |
| `paired_changes.csv` | 有同协议对照时保存逐篇配对，列为 `id, prompt_id, fold, y_true, baseline_pred, y_pred, outcome`。`outcome` 取 `corrected`、`regressed`、`both_correct` 或 `both_wrong`。 |
| `training_curve.csv` | 微调时保存训练和内部验证曲线数据，注明选定轮数或 checkpoint。 |

`run.json` 的核验字段为 `id`、`kind`、`status`。正式折外运行的 `kind` 为 `grouped_oof`，冒烟为 `smoke`。状态用 `running`、`completed`、`failed` 或 `incomplete`。正式运行完成时填写 `protocol_id`、`train_sha256`、`folds_sha256`、`exit_code` 和 `baseline_run`（没有对照时用 JSON 的 `null`）。同一次运行的配置、日志和回顾保留在原目录。重跑或调整参数时取新编号，下降结果照常归档。正式表格使用通过核验的完整折外运行。训练曲线和额外诊断按实际运行类型落盘。

微调的 checkpoint 放在该次运行的 `checkpoints/`，记录各折选中权重的路径和指纹。可复用的向量放在 `data/cache/`，随缓存记录输入数据指纹、模型版本、截断、池化和 `id` 顺序。混淆矩阵 CSV 的第一列为 `true_model`，其余列使用官方标签。指标和比例以计算精度保存，供核验时复算。

## 回顾顺序

1. **先确认实验做完了。** 看状态、日志、命令、对照和实际改动，再核对数据指纹、折号和标签。
2. **看效果与成本。** 看整体 BA、各折 BA、Macro-F1、最差类别召回，以及耗时和硬件。用百分点写 BA 变化。
3. **看错在哪里。** 混淆矩阵的每一行回答某个真实模型被判成了谁。结合行内比例和原始篇数，挑几对最常见的误判，回到源数据按 `id` 看原文。记录所见线索及证据范围。
4. **看具体变化。** 同协议对照逐篇核对 `id`、题号、折和真实标签，报告改对与改错数量。需要评估差值波动时按 Prompt 配对重抽，保留同题的全部样本。
5. **写出决定。** 进入下一阶段、保留作对照、或保留回退并说明理由。引用可核对的表、图或样本。

`check_run.py` 对完整折外运行检查必需文件、评价协议和逐篇覆盖，从预测重算整体、分折、分类别指标和两版混淆矩阵。声明了对照时还检查逐篇变化。热图由回顾者结合 CSV 检查坐标和标注。失败、未完成和冒烟运行通过其日志及回顾验收。

## 对应论文

| 论文位置 | 优先使用的证据 |
|---|---|
| 数据与方法 | 评价协议目录、分折检查、运行配置。 |
| 实验结果 | 各实验 `metrics.json`、`by_fold.csv`、`by_model.csv`。 |
| 误差分析 | 混淆矩阵、逐篇配对表、实际文本案例。 |
| 讨论 | 回退实验、成本和适用范围，附原始实验路径。 |
| 比赛结果 | `submissions/` 内的正式回执，附对应候选实验编号。 |
