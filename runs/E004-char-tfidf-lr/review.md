# E004-char-tfidf-lr  字符 TF-IDF + 逻辑回归

## 实验问题

字符片段能否提供比词和短语更有效的归因信号？

## 实测结果

整体 BA **30.78%**。Macro-F1 **29.69%**。耗时 66.29 秒，包含特征构建、拟合与预测。

[各折](by_fold.csv) · [各类别](by_model.csv) · [逐篇预测](oof.parquet) · [实际配置](run.json)

## 常见误判

- gemini-3.8-flash → gemini-3.7-flash 有 49 篇，占该真实类 32.67%。示例 ID 017623d68134。
- gemini-3.6-flash → gemini-3.7-flash 有 44 篇，占该真实类 29.33%。示例 ID 081237447f0c。
- gemini-3.7-flash → gemini-3.8-flash 有 38 篇，占该真实类 25.33%。示例 ID 05204897132f。
- glm-5.3 → claude-opus-5 有 36 篇，占该真实类 24.00%。示例 ID 0c620285bef2。

[混淆矩阵热图](figures/confusion.png) · [原始计数](confusion_counts.csv)

## 同篇对照

对照 E003-word-tfidf-lr。311 篇改对，229 篇改错。可按题号追踪变化。

[逐篇配对](paired_changes.csv)

[原文案例](cases.csv)

[各折的高权重特征](top_features.csv)

