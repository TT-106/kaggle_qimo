# E001-most-frequent-r3  常量预测

## 实验问题

固定五折和 18 类 BA 能否复现多数类常量预测的理论值？

## 实测结果

整体 BA **5.53%**。Macro-F1 **1.05%**。耗时 0.02 秒，包含特征构建、拟合与预测。

[各折](by_fold.csv) · [各类别](by_model.csv) · [逐篇预测](oof.parquet) · [实际配置](run.json)

## 常见误判

- claude-fable-5 → claude-opus-4-6 有 120 篇，占该真实类 80.54%。示例 ID 02b9b5daabe6。
- claude-opus-4-8 → claude-opus-4-6 有 120 篇，占该真实类 80.00%。示例 ID 01b5eec535ac。
- claude-opus-5 → claude-opus-4-6 有 120 篇，占该真实类 80.00%。示例 ID 030145156daf。
- claude-sonnet-5 → claude-opus-4-6 有 120 篇，占该真实类 80.00%。示例 ID 0041ac86ee93。

[混淆矩阵热图](figures/confusion.png) · [原始计数](confusion_counts.csv)

[原文案例](cases.csv)

