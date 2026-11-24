# E006-style-xgboost  统计风格 + XGBoost

## 实验问题

在同样 20 项统计特征上，XGBoost 的浅树交互能否改善归因？

## 实测结果

整体 BA **24.56%**。Macro-F1 **24.01%**。耗时 14.11 秒，包含特征构建、拟合与预测。

[各折](by_fold.csv) · [各类别](by_model.csv) · [逐篇预测](oof.parquet) · [实际配置](run.json)

## 常见误判

- gemini-3.7-flash → gemini-3.8-flash 有 36 篇，占该真实类 24.00%。示例 ID 05204897132f。
- gemini-3.8-flash → gemini-3.7-flash 有 34 篇，占该真实类 22.67%。示例 ID 010967f56aaf。
- gemini-3.8-flash → gemini-3.6-flash 有 29 篇，占该真实类 19.33%。示例 ID 017623d68134。
- gemini-3.6-flash → gemini-3.7-flash 有 28 篇，占该真实类 18.67%。示例 ID 0433628fedae。

[混淆矩阵热图](figures/confusion.png) · [原始计数](confusion_counts.csv)

## 同篇对照

对照 E002-style-lr。235 篇改对，335 篇改错。可按题号追踪变化。

[逐篇配对](paired_changes.csv)

[原文案例](cases.csv)

