# E002-style-lr  统计风格 + 逻辑回归

## 实验问题

标点、长度和句子统计在未见题上有多少区分能力？

## 实测结果

整体 BA **28.26%**。Macro-F1 **26.72%**。耗时 2.36 秒，包含特征构建、拟合与预测。

[各折](by_fold.csv) · [各类别](by_model.csv) · [逐篇预测](oof.parquet) · [实际配置](run.json)

## 常见误判

- gemini-3.8-flash → gemini-3.7-flash 有 42 篇，占该真实类 28.00%。示例 ID 017623d68134。
- gemini-3.6-flash → gemini-3.7-flash 有 38 篇，占该真实类 25.33%。示例 ID 0433628fedae。
- gemini-3.6-flash → gemini-3.1-pro-preview 有 31 篇，占该真实类 20.67%。示例 ID 005a19083eae。
- gemini-3.7-flash → gemini-3.8-flash 有 31 篇，占该真实类 20.67%。示例 ID 05204897132f。

[混淆矩阵热图](figures/confusion.png) · [原始计数](confusion_counts.csv)

## 同篇对照

对照 E001-most-frequent-r3。715 篇改对，101 篇改错。可按题号追踪变化。

[逐篇配对](paired_changes.csv)

[原文案例](cases.csv)

[各折的高权重特征](top_features.csv)

