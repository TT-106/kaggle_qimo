"""Compare the Open Pool character LightGBM run with its paired style baseline."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import tomllib

import pandas as pd

from summarize_traditional import ROOT, RESAMPLES, plot_overview, read_runs


BASE = "E010-open-pool-style-lightgbm"
CANDIDATE = "E011-open-pool-char-style-lightgbm"
OUTPUT = ROOT / "paper" / "results" / "open_pool_char_style"


def read_configuration():
    experiments = {}
    for file, run in (("open_pool_style.toml", BASE), ("open_pool_char_style.toml", CANDIDATE)):
        with (ROOT / "configs" / file).open("rb") as handle:
            config = tomllib.load(handle)
        experiments[run] = config["runs"][run]
    return {"runs": experiments}


def read_record(run):
    return json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8"))


def write_report(summary, fold_ba, base, added, path):
    old, new = summary.iloc[0], summary.iloc[1]
    char_features = added["config"]["experiment"]["chi2_char_features"]
    widths = sorted({part["feature_columns"] for part in added["fold_runtime"]})
    lines = [
        "# Open Pool 上给 LightGBM 增加字符 TF-IDF", "",
        f"E010 与 E011 都在同一份 {added['verified_oof_rows']:,} 篇、18 类的 Open Pool 按题五折上评价。"
        "原 20 个统计特征值和 LightGBM 训练参数一致。", "",
        f"E011 在训练折按 3–5 字符片段构建 TF-IDF，词表最多 80,000 列，"
        f"再用训练标签的卡方统计量保留 {char_features:,} 个片段。实际输入 {widths[0]:,} 列。"
        "逐篇计数共用，列选择、IDF 和卡方得分均从训练侧产生。", "",
        "| 对照 | 输入 | 合并 BA | Macro-F1 | 五折计算耗时 |",
        "|---|---|---:|---:|---:|",
        f"| [E010](../../../runs/{BASE}/review.md) | 20 项统计 | {old.balanced_accuracy:.2%} "
        f"| {old.macro_f1:.2%} | {old.compute_seconds:.2f} 秒 |",
        f"| [E011](../../../runs/{CANDIDATE}/review.md) | 统计 + 字符 TF-IDF | {new.balanced_accuracy:.2%} "
        f"| {new.macro_f1:.2%} | {new.compute_seconds:.2f} 秒 |",
        "",
        f"**E011 与 E010 的 BA 相差 {new.delta_ba_pp:+.2f} 个百分点。** "
        f"{int(new.corrected):,} 篇由错变对，{int(new.regressed):,} 篇由对变错。"
        f"按题组重抽 {RESAMPLES:,} 次的差值区间为 [{new.paired_95_lower_pp:+.2f}, "
        f"{new.paired_95_upper_pp:+.2f}] 个百分点。", "",
        "![特征对照的成绩与耗时](overview.png)", "",
        "| 折 | E010 BA | E011 BA | 差值，百分点 |",
        "|---:|---:|---:|---:|",
    ]
    for row in fold_ba.itertuples(index=False):
        lines.append(f"| {row.fold} | {row.base:.2%} | {row.char_style:.2%} | {row.delta_pp:+.2f} |")
    lines.extend([
        "",
        f"E011 另有一次统计特征提取 {added['shared_style_feature_seconds']:.2f} 秒和一次逐篇字符计数 "
        f"{added['shared_char_count_seconds']:.2f} 秒。E010 的统计特征提取耗时 "
        f"{base['shared_style_feature_seconds']:.2f} 秒。", "",
        "预先只在第一折训练侧测了轻量运行。80,020 列和 4,116 列分别跑了 4 轮，"
        "实测耗时 182.06 秒与 9.36 秒。完整五折采用 2,048 个字符特征的方案。", "",
        "在相同外部池分折和发布方标签下，这组对照衡量加入字符特征的效果。", "",
        f"[逐篇配对](../../../runs/{CANDIDATE}/paired_changes.csv) · "
        f"[E011 混淆矩阵](../../../runs/{CANDIDATE}/figures/confusion.png) · "
        "[各类召回差值](recall_comparison.csv) · [分折明细](fold_ba.csv)",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    base, added = read_record(BASE), read_record(CANDIDATE)
    if (base["config"]["estimator"] != added["config"]["estimator"]
            or base["config"]["style_columns"] != added["config"]["style_columns"]):
        raise ValueError("分类器参数或原始统计列不一致。")
    summary, recalls = read_runs(read_configuration())
    recalls["delta_pp"] = (recalls[CANDIDATE] - recalls[BASE]) * 100
    left = pd.read_csv(ROOT / "runs" / BASE / "by_fold.csv")[["fold", "balanced_accuracy"]]
    right = pd.read_csv(ROOT / "runs" / CANDIDATE / "by_fold.csv")[["fold", "balanced_accuracy"]]
    folds = left.rename(columns={"balanced_accuracy": "base"}).merge(
        right.rename(columns={"balanced_accuracy": "char_style"}), on="fold", validate="one_to_one"
    )
    folds["delta_pp"] = (folds.char_style - folds.base) * 100

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".open-pool-char-style-", dir=OUTPUT.parent) as temp:
        stage = Path(temp) / OUTPUT.name
        stage.mkdir()
        summary.to_csv(stage / "summary.csv", index=False, encoding="utf-8")
        folds.to_csv(stage / "fold_ba.csv", index=False, encoding="utf-8")
        recalls.to_csv(stage / "recall_comparison.csv", index_label="model", encoding="utf-8")
        plot_overview(summary, stage / "overview.png", title="Open Pool 上给 LightGBM 增加字符 TF-IDF", legend_location="upper right")
        write_report(summary, folds, base, added, stage / "REPORT.md")
        stage.rename(OUTPUT)
    print((OUTPUT / "REPORT.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
