# E007-style-lightgbm  统计风格 + LightGBM

## 实验问题

在同样 20 项统计特征上，LightGBM 的效果和成本如何？

## 实测结果

整体 BA **24.89%**。Macro-F1 **24.51%**。耗时 8.17 秒，包含特征构建、拟合与预测。

[各折](by_fold.csv) · [各类别](by_model.csv) · [逐篇预测](oof.parquet) · [实际配置](run.json)

## 常见误判

- gemini-3.7-flash → gemini-3.8-flash 有 38 篇，占该真实类 25.33%。示例 ID 05204897132f。
- gemini-3.8-flash → gemini-3.7-flash 有 36 篇，占该真实类 24.00%。示例 ID 010967f56aaf。
- gemini-3.6-flash → gemini-3.8-flash 有 29 篇，占该真实类 19.33%。示例 ID 0e0d9e0c7a05。
- gemini-3.6-flash → gemini-3.7-flash 有 28 篇，占该真实类 18.67%。示例 ID 0433628fedae。

[混淆矩阵热图](figures/confusion.png) · [原始计数](confusion_counts.csv)

## 同篇对照

对照 E002-style-lr。255 篇改对，346 篇改错。可按题号追踪变化。

[逐篇配对](paired_changes.csv)

[原文案例](cases.csv)

