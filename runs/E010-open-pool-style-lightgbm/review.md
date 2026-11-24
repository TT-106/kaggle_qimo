# E010-open-pool-style-lightgbm  统计风格 + LightGBM

## 实验问题

使用同一份外部语料、分折和统计特征，换成 LightGBM 后效果如何？

## 实测结果

整体 BA **23.93%**。Macro-F1 **23.05%**。耗时 57.58 秒，包含每折特征处理、拟合与预测。 本组另外共用一次特征提取，实测 14.82 秒。

[各折](by_fold.csv) · [各类别](by_model.csv) · [逐篇预测](oof.parquet) · [实际配置](run.json)

## 常见误判

- gemini-3.6-flash → gemini-3.8-flash 有 2569 篇，占该真实类 53.45%。示例 ID ntlop_062156。
- gemini-3.7-flash → gemini-3.8-flash 有 1841 篇，占该真实类 59.18%。示例 ID ntlop_078296。
- glm-5.3 → glm-5.3-flash 有 1374 篇，占该真实类 21.94%。示例 ID ntlop_111811。
- glm-5.3-flash → claude-fable-5 有 1344 篇，占该真实类 20.90%。示例 ID ntlop_123661。

[混淆矩阵热图](figures/confusion.png) · [原始计数](confusion_counts.csv)

## 同篇对照

对照 E008-open-pool-style-lr。10251 篇改对，6652 篇改错。可按题号追踪变化。

[逐篇配对](paired_changes.csv)

[原文案例](cases.csv)

