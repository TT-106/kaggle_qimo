# E011-open-pool-char-style-lightgbm  字符 TF-IDF + 统计风格 + LightGBM

## 实验问题

保持 E010 的分折、原始 20 项统计特征和 LightGBM 参数，增加训练侧筛选的字符 TF-IDF 能提升多少？

## 实测结果

整体 BA **33.68%**。Macro-F1 **33.39%**。耗时 833.99 秒，包含每折特征处理、拟合与预测。 一次统计特征提取实测 14.85 秒。 一次逐篇字符计数实测 162.40 秒。

[各折](by_fold.csv) · [各类别](by_model.csv) · [逐篇预测](oof.parquet) · [实际配置](run.json)

## 常见误判

- gemini-3.6-flash → gemini-3.8-flash 有 2307 篇，占该真实类 48.00%。示例 ID ntlop_062154。
- gemini-3.7-flash → gemini-3.8-flash 有 2162 篇，占该真实类 69.50%。示例 ID ntlop_078299。
- glm-5.3-flash → claude-fable-5 有 1490 篇，占该真实类 23.17%。示例 ID ntlop_123655。
- glm-5.3 → glm-5.3-flash 有 1455 篇，占该真实类 23.23%。示例 ID ntlop_111808。

[混淆矩阵热图](figures/confusion.png) · [原始计数](confusion_counts.csv)

## 同篇对照

对照 E010-open-pool-style-lightgbm。16017 篇改对，7156 篇改错。可按题号追踪变化。

[逐篇配对](paired_changes.csv)

[原文案例](cases.csv)

