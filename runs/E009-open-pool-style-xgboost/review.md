# E009-open-pool-style-xgboost  统计风格 + XGBoost

## 实验问题

使用同一份外部语料、分折和统计特征，换成 XGBoost 后效果如何？

## 实测结果

整体 BA **23.04%**。Macro-F1 **21.97%**。耗时 112.10 秒，包含每折特征处理、拟合与预测。 本组另外共用一次特征提取，实测 14.82 秒。

[各折](by_fold.csv) · [各类别](by_model.csv) · [逐篇预测](oof.parquet) · [实际配置](run.json)

## 常见误判

- gemini-3.6-flash → gemini-3.8-flash 有 2728 篇，占该真实类 56.76%。示例 ID ntlop_062155。
- gemini-3.7-flash → gemini-3.8-flash 有 1890 篇，占该真实类 60.75%。示例 ID ntlop_078296。
- glm-5.3-flash → claude-fable-5 有 1469 篇，占该真实类 22.85%。示例 ID ntlop_123655。
- glm-5.3 → glm-5.3-flash 有 1448 篇，占该真实类 23.12%。示例 ID ntlop_111811。

[混淆矩阵热图](figures/confusion.png) · [原始计数](confusion_counts.csv)

## 同篇对照

对照 E008-open-pool-style-lr。9951 篇改对，6910 篇改错。可按题号追踪变化。

[逐篇配对](paired_changes.csv)

[原文案例](cases.csv)

