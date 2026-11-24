"""Produce the Traditional Baselines paper table and its paired diagnostics."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import tomllib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESAMPLES = 2000
SEED = 42


def paired_summary(pairs, labels):
    aligned = pairs.assign(delta=pairs["outcome"].map({"corrected": 1, "regressed": -1}).fillna(0).astype(int))
    support = pd.crosstab(aligned.prompt_id, aligned.y_true).reindex(columns=labels, fill_value=0)
    change = aligned.pivot_table(index="prompt_id", columns="y_true", values="delta", aggfunc="sum", fill_value=0)
    change = change.reindex(index=support.index, columns=labels, fill_value=0)
    topics = len(support)
    draws = np.random.default_rng(SEED).multinomial(topics, np.full(topics, 1 / topics), size=RESAMPLES)
    denominators = draws @ support.to_numpy()
    numerators = draws @ change.to_numpy()
    recall_change = np.divide(numerators, denominators, out=np.zeros_like(numerators, dtype=float),
                              where=denominators != 0)
    delta_ba = recall_change.sum(axis=1) / (denominators != 0).sum(axis=1)
    lower, upper = np.quantile(delta_ba, [.025, .975]) * 100
    topics_with_net_change = change.sum(axis=1)
    return {
        "corrected": int(aligned.delta.eq(1).sum()),
        "regressed": int(aligned.delta.eq(-1).sum()),
        "prompts_net_better": int(topics_with_net_change.gt(0).sum()),
        "prompts_net_worse": int(topics_with_net_change.lt(0).sum()),
        "paired_95_lower_pp": lower,
        "paired_95_upper_pp": upper,
    }


def read_runs(config):
    results, recalls = [], {}
    for run_id, spec in config["runs"].items():
        folder = ROOT / "runs" / run_id
        info = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        if info["status"] != "completed" or info["exit_code"] != 0 or not info["verified_oof_rows"]:
            raise ValueError(f"实验 {run_id} 尚未完成归档。")
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        folds = pd.read_csv(folder / "by_fold.csv")
        classes = pd.read_csv(folder / "by_model.csv").set_index("model")
        recalls[run_id] = classes["recall"]
        record = {
            "run": run_id, "title": spec["title"], **metrics, "fold_min_ba": folds.balanced_accuracy.min(),
            "fold_max_ba": folds.balanced_accuracy.max(), "compute_seconds": info["compute_seconds"],
            **{key: sum(fold[key] for fold in info["fold_runtime"])
               for key in ("feature_seconds", "fit_seconds", "predict_seconds")},
            "baseline_run": spec.get("baseline"), "protocol_id": info["protocol_id"],
            "train_sha256": info["train_sha256"], "folds_sha256": info["folds_sha256"],
        }
        if spec.get("baseline"):
            parent = ROOT / "runs" / spec["baseline"]
            before = json.loads((parent / "metrics.json").read_text(encoding="utf-8"))
            paired = pd.read_csv(folder / "paired_changes.csv")
            record.update(delta_ba_pp=(metrics["balanced_accuracy"] - before["balanced_accuracy"]) * 100,
                          **paired_summary(paired, info["label_order"]))
        results.append(record)
    summary = pd.DataFrame(results)
    if not summary[["protocol_id", "train_sha256", "folds_sha256"]].nunique().eq(1).all():
        raise ValueError("所有实验须使用相同协议和数据才能生成这张对照表。")
    return summary, pd.DataFrame(recalls)


def plot_overview(summary, path, title="官方训练集 传统基线比较", *, legend_location="lower right"):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, (score_ax, time_ax) = plt.subplots(1, 2, figsize=(13.4, 5.8), sharey=True, layout="constrained")
    fig.suptitle(title, fontsize=17, fontweight="bold")
    positions = np.arange(len(summary))
    height = .54
    score_ax.barh(positions, summary.balanced_accuracy, height=height, color="#176b70")
    for iy, val in enumerate(summary.balanced_accuracy):
        score_ax.text(val + .006, iy, f"{val:.2%}", va="center", fontsize=9)
    score_ax.set_xlim(0, max(summary.balanced_accuracy) * 1.23)
    score_ax.xaxis.set_major_formatter(PercentFormatter(1))
    score_ax.set_yticks(positions, [f"{r[:4]}  {title}" for r, title in zip(summary.run, summary.title, strict=True)])
    score_ax.invert_yaxis()
    score_ax.set_title("五折合并 Balanced Accuracy", loc="left", fontweight="bold")
    score_ax.set_axisbelow(True)
    score_ax.grid(axis="x", alpha=.2)

    cumulative = np.zeros(len(summary))
    for column, label, color in [
        ("feature_seconds", "构建特征", "#6b9e9f"),
        ("fit_seconds", "拟合", "#ce7e53"),
        ("predict_seconds", "预测", "#5e6d8a"),
    ]:
        time_ax.barh(positions, summary[column], left=cumulative, height=height, label=label, color=color)
        cumulative += summary[column].to_numpy()
    order = max(cumulative)
    for iy, val in enumerate(cumulative):
        time_ax.text(val + order * .012, iy, f"{val:.2f} s", va="center", fontsize=9)
    time_ax.set_xlim(0, order * 1.25)
    time_ax.set_title("五折计算耗时", loc="left", fontweight="bold")
    time_ax.set_xlabel("秒")
    time_ax.tick_params(axis="y", left=False)
    time_ax.set_axisbelow(True)
    time_ax.grid(axis="x", alpha=.2)
    time_ax.legend(loc=legend_location, frameon=False)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def write_report(summary, destination):
    header = [
        "# 第一阶段传统基线实测", "",
        "七组实验均使用官方训练文本和冻结的按 Prompt 五折。BA、Macro-F1 由完整折外预测复算。"
        "耗时为五折的特征构建、拟合和预测之和。", "",
        "![传统基线的 BA 与耗时](overview.png)", "",
        "| 实验 | 方法 | 整体 BA | Macro-F1 | 各折 BA 范围 | 计算耗时 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        header.append(
            f"| [{row.run[:4]}](../../../runs/{row.run}/review.md) | {row.title} | {row.balanced_accuracy:.2%} "
            f"| {row.macro_f1:.2%} | {row.fold_min_ba:.2%}–{row.fold_max_ba:.2%} | {row.compute_seconds:.2f} 秒 |"
        )
    header.extend([
        "", "## 同篇比较", "",
        f"按同一 ID 配对改对和改错。BA 差值区间来自 {RESAMPLES:,} 次按 Prompt 成组重抽，"
        "用于观察在这批题目上的差值波动。", "",
        "| 实验与对照 | BA 差值 | 改对 | 改错 | 净改善题数 | 净退步题数 | 配对差值区间 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary.loc[summary.baseline_run.notna()].itertuples(index=False):
        header.append(
            f"| {row.run[:4]} 对 {row.baseline_run[:4]} | {row.delta_ba_pp:+.2f} 个百分点 "
            f"| {int(row.corrected)} | {int(row.regressed)} | {int(row.prompts_net_better)} "
            f"| {int(row.prompts_net_worse)} | [{row.paired_95_lower_pp:+.2f}, {row.paired_95_upper_pp:+.2f}] |"
        )
    best = summary.iloc[summary.balanced_accuracy.argmax()]
    header.extend([
        "", "## 本轮结论", "",
        f"本轮最高 BA 来自 **{best['run'][:4]} {best['title']}**，为 **{best['balanced_accuracy']:.2%}**。"
        f" [该次混淆矩阵](../../../runs/{best['run']}/figures/confusion.png)",
        "",
        "常量基线每折预测该训练侧的多数类，按固定标签顺序处理并列。"
        "各折理论 BA 为 1/18，合并分数按实际逐篇预测计算。重跑前的断言和绘图中止可查原日志。"
        " [首次记录](../../../runs/E001-most-frequent/run.log) · [第二次记录](../../../runs/E001-most-frequent-r2/run.log)",
        "", "[总表](summary.csv) · [按模型召回](recall_by_model.csv) · [实验参数](../../../configs/traditional_baselines.toml)",
    ])
    (destination / "REPORT.md").write_text("\n".join(header) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="从完整归档生成传统基线图和论文简表。")
    parser.add_argument("--output", type=Path, default=ROOT / "paper" / "results" / "traditional_baselines")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error(f"选择新的输出目录。{output} 已存在。")
    with (ROOT / "configs" / "traditional_baselines.toml").open("rb") as stream:
        config = tomllib.load(stream)
    summary, recalls = read_runs(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".traditional-", dir=output.parent) as scratch:
        staged = Path(scratch) / output.name
        staged.mkdir()
        summary.to_csv(staged / "summary.csv", index=False, encoding="utf-8")
        recalls.to_csv(staged / "recall_by_model.csv", index_label="model", encoding="utf-8")
        plot_overview(summary, staged / "overview.png")
        write_report(summary, staged)
        staged.rename(output)
    print(summary[["run", "balanced_accuracy", "macro_f1", "compute_seconds", "delta_ba_pp",
                   "corrected", "regressed"]].to_string(index=False))
    print(output / "REPORT.md")


if __name__ == "__main__":
    main()
