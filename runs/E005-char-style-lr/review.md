# E005-char-style-lr  字符 TF-IDF + 统计风格 + 逻辑回归

## 实验问题

在字符 TF-IDF 上增加 20 个显式统计特征，能否改善逐篇归因？

## 实测结果

整体 BA **32.11%**。Macro-F1 **30.50%**。耗时 90.75 秒，包含特征构建、拟合与预测。

[各折](by_fold.csv) · [各类别](by_model.csv) · [逐篇预测](oof.parquet) · [实际配置](run.json)

## 常见误判

- gemini-3.8-flash → gemini-3.7-flash 有 46 篇，占该真实类 30.67%。示例 ID 017623d68134。
- gemini-3.7-flash → gemini-3.8-flash 有 37 篇，占该真实类 24.67%。示例 ID 05204897132f。
- gemini-3.6-flash → gemini-3.7-flash 有 33 篇，占该真实类 22.00%。示例 ID 0433628fedae。
- gemini-3.6-flash → gemini-3.1-pro-preview 有 31 篇，占该真实类 20.67%。示例 ID 005a19083eae。

[混淆矩阵热图](figures/confusion.png) · [原始计数](confusion_counts.csv)

## 同篇对照

对照 E004-char-tfidf-lr。398 篇改对，362 篇改错。可按题号追踪变化。

[逐篇配对](paired_changes.csv)

[原文案例](cases.csv)

[各折的高权重特征](top_features.csv)

