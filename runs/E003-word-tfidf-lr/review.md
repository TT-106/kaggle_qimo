# E003-word-tfidf-lr  词级 TF-IDF + 逻辑回归

## 实验问题

同一分类器改用词与短语频率之后，整体效果和错误结构怎样变化？

## 实测结果

整体 BA **27.74%**。Macro-F1 **26.29%**。耗时 9.05 秒，包含特征构建、拟合与预测。

[各折](by_fold.csv) · [各类别](by_model.csv) · [逐篇预测](oof.parquet) · [实际配置](run.json)

## 常见误判

- gemini-3.8-flash → gemini-3.7-flash 有 46 篇，占该真实类 30.67%。示例 ID 03293c742476。
- glm-5.3 → claude-opus-5 有 46 篇，占该真实类 30.67%。示例 ID 094705a424ee。
- gemini-3.7-flash → gemini-3.6-flash 有 39 篇，占该真实类 26.00%。示例 ID 06606903cea3。
- gemini-3.7-flash → gemini-3.8-flash 有 37 篇，占该真实类 24.67%。示例 ID 05204897132f。

[混淆矩阵热图](figures/confusion.png) · [原始计数](confusion_counts.csv)

## 同篇对照

对照 E002-style-lr。438 篇改对，452 篇改错。可按题号追踪变化。

[逐篇配对](paired_changes.csv)

[原文案例](cases.csv)

[各折的高权重特征](top_features.csv)

