# E008-open-pool-style-lr  统计风格 + 逻辑回归

## 实验问题

在外部 Open Pool 自身的按题五折中，20 项统计特征与逻辑回归的基准分数是多少？

## 实测结果

整体 BA **20.28%**。Macro-F1 **18.89%**。耗时 19.25 秒，包含每折特征处理、拟合与预测。 本组另外共用一次特征提取，实测 14.82 秒。

[各折](by_fold.csv) · [各类别](by_model.csv) · [逐篇预测](oof.parquet) · [实际配置](run.json)

## 常见误判

- gemini-3.6-flash → gemini-3.8-flash 有 2578 篇，占该真实类 53.64%。示例 ID ntlop_062156。
- glm-5.3-flash → claude-fable-5 有 2248 篇，占该真实类 34.96%。示例 ID ntlop_123658。
- glm-5.3 → claude-fable-5 有 2092 篇，占该真实类 33.40%。示例 ID ntlop_111809。
- gemini-3.7-flash → gemini-3.8-flash 有 1764 篇，占该真实类 56.70%。示例 ID ntlop_078296。

[混淆矩阵热图](figures/confusion.png) · [原始计数](confusion_counts.csv)

[原文案例](cases.csv)

[各折的高权重特征](top_features.csv)

