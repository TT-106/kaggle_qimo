# Name That LLM 期末论文项目

项目比较传统文本特征、Qwen 冻结向量和 DeBERTa 监督微调在未见 Prompt 上的归因表现。主指标是 18 类召回率的平均值，即 Balanced Accuracy（BA）。

## 目录

| 位置 | 用途 |
|---|---|
| [configs/](configs/) | 共享资源路径和当前评价口径。每次实际使用的参数快照随运行归档。 |
| [data/protocols/](data/protocols/) | 按协议保存官方训练文件指纹、标签顺序和分折清单。当前协议见 [manifest.json](data/protocols/prompt-group-5fold-v1/manifest.json)。 |
| [data/external/](data/external/) | 外部原始数据和相应审计。当前可查 [Open Pool 数据审计](data/external/ntl_open_pool_audit/REPORT.md)。 |
| [src/](src/) | 数据准备、实验入口和公共评价代码。 |
| [docs/](docs/) | 实验记录规范和单次回顾模板。 |
| [runs/](runs/) | 每次运行一个目录，从回顾和图表可以追到逐篇预测。 |
| [submissions/](submissions/) | Kaggle 提交产物、关联的实验编号和实际回执。 |
| [paper/](paper/) | 论文正文和选用的图表。引用实验编号与产物路径。 |
| [tests/](tests/) | 检查分折和归档契约的基础测试。 |

## 共享资源

[项目配置](configs/project.toml) 引用旧项目的 CPU 和 GPU 环境、官方数据，以及 Qwen3-Embedding-0.6B 权重。新项目的产物写入本目录。GPU 阶段启动前用选定环境实测 CUDA，DeBERTa-v3-base 的本地权重路径在使用前核实并登记。

## 从项目根目录运行

```powershell
$python = 'S:\coding_relative\kaggle\name-that-llm-local\.venv\Scripts\python.exe'
& $python -X utf8 -B .\src\prepare_data.py
& $python -X utf8 -B -m unittest discover -s tests -v
```

数据检查命令核验已冻结的文件。初次创建协议时，给同一命令加上 `--freeze`。已冻结目录按内容校验，新的评价口径使用新的协议编号。

## 第一阶段传统基线

[实测报告](paper/results/traditional_baselines/REPORT.md) · [固定参数](configs/traditional_baselines.toml) · [运行入口](src/run_traditional.py)

E001–E007 使用官方训练数据和同一份按题五折。每次运行使用独立编号，完整归档后由检查工具从逐篇结果复算。E001 的可核验结果在 `E001-most-frequent-r3`。

树模型依赖按 [版本清单](configs/traditional_tree_requirements.txt) 放入本项目的缓存。运行入口使用配置中的 CPU 环境，绘图调用旧 GPU 环境中的 Matplotlib。

```powershell
$python = 'S:\coding_relative\kaggle\name-that-llm-local\.venv\Scripts\python.exe'
uv pip install --python $python --target .\data\cache\tree_dependencies --no-deps --only-binary ':all:' --link-mode copy --index-url https://pypi.org/simple -r .\configs\traditional_tree_requirements.txt
```

## 外部 Open Pool 五折

[实测报告](paper/results/open_pool_style/REPORT.md) · [外部分折](data/protocols/open-pool-prompt-group-5fold-v1/manifest.json) · [实验配置](configs/open_pool_style.toml)

复现这组实验时，从 [Kaggle Open Pool 发布页](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool) 获取原始数据，放到 `data/external/ntl_open_pool.parquet`。使用 `Get-FileHash .\data\external\ntl_open_pool.parquet -Algorithm SHA256` 核对结果是否与外部分折清单的 `train_sha256` 一致。

E008–E010 在筛选后的 Open Pool 上训练与验证，沿用传统基线的 20 项统计特征和分类器参数。原始与归并题号合并后固定分折，按照发布者标签计算 BA。

[字符 TF-IDF 的 LightGBM 对照](paper/results/open_pool_char_style/REPORT.md) 记录 E011 和 E010 的五折差值、逐篇变化与耗时。两次使用同一份 Open Pool 分折与 LightGBM 参数。

## 实验审阅

运行前查 [实验规范](docs/experiment-protocol.md)，运行后从该次实验的 `review.md` 开始。审阅模板见 [单次实验回顾](docs/review-template.md)。

完成一次运行后，在项目根目录执行下列命令，将路径换成实际运行目录。

```powershell
$python = 'S:\coding_relative\kaggle\name-that-llm-local\.venv\Scripts\python.exe'
& $python -X utf8 -B .\src\check_run.py .\runs\E004-char-tfidf-lr
```
