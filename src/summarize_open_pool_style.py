"""Summarize the shared external Open Pool style-classifier folds."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import tomllib

from summarize_traditional import ROOT, RESAMPLES, plot_overview, read_runs


SUITE = ROOT / "configs" / "open_pool_style.toml"
OUTPUT = ROOT / "paper" / "results" / "open_pool_style"


def build_report(summary, manifest, shared_seconds):
    cohort = manifest["cohort"]
    first, last = (min(item["validation_rows"] for item in manifest["fold_checks"]),
                   max(item["validation_rows"] for item in manifest["fold_checks"]))
    best = summary.iloc[summary.balanced_accuracy.argmax()]
    lines = [
        "# Open Pool 按题五折", "",
        f"使用 Open Pool 中 **{manifest['rows']:,} 篇、18 类、{cohort['selected_sources']} 个来源**。按两个题组编号"
        f"合并为 {manifest['prompts']:,} 组。每折有 {first:,}–{last:,} 篇验证文章，均覆盖 18 类。",
        "",
        f"原始文件有 {cohort['raw_rows']:,} 篇。先要求两个题组编号齐备，余 {cohort['retained_after_prompt_ids']:,} 篇。"
        f"排除带风险标记的记录后余 {cohort['retained_after_risk_flags']:,} 篇，再保留 own_bank 和 own_text。"
        "原始和归并题号跨折数均为 0。",
        "",
        f"标签来自 Open Pool 的 model 字段，其中 {cohort['selected_missing_served_model']:,} 篇缺少实际服务名记录。"
        "分数衡量这个外部池里对未见题组的归因表现。",
        "",
        "![外部数据上三个分类器的分数和耗时](overview.png)", "",
        "| 实验 | 方法 | BA | Macro-F1 | 各折 BA 范围 | 五折耗时 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(f"| [{row.run[:4]}](../../../runs/{row.run}/review.md) | {row.title} "
                     f"| {row.balanced_accuracy:.2%} | {row.macro_f1:.2%} "
                     f"| {row.fold_min_ba:.2%}–{row.fold_max_ba:.2%} | {row.compute_seconds:.2f} 秒 |")
    lines.extend([
        "",
        f"三组共享一次逐篇特征提取，耗时 {shared_seconds:.2f} 秒。表中耗时包括各折特征处理、拟合和预测。",
        "", "## 对照逻辑回归", "",
        f"同一篇文章配对，按合并题组重抽 {RESAMPLES:,} 次，观察 BA 差值在这批题组上的波动。",
        "",
        "| 实验 | BA 差值 | 改对 | 改错 | 配对差值区间 |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in summary.loc[summary.baseline_run.notna()].itertuples(index=False):
        lines.append(f"| {row.run[:4]} | {row.delta_ba_pp:+.2f} 个百分点 | {int(row.corrected):,} | "
                     f"{int(row.regressed):,} | [{row.paired_95_lower_pp:+.2f}, {row.paired_95_upper_pp:+.2f}] |")
    lines.extend([
        "",
        f"本轮最高为 **{best['title']}（{best['run'][:4]}），BA {best['balanced_accuracy']:.2%}**。"
        f" [混淆矩阵](../../../runs/{best['run']}/figures/confusion.png)",
        "",
        "三组共用外部语料、题组分折和 20 项风格特征。正式比赛的隐藏测试由 Kaggle 单独评分，"
        "外部语料对官方数据的迁移需要在官方数据上另测。",
        "",
        "[分折和语料范围](../../../data/protocols/open-pool-prompt-group-5fold-v1/manifest.json) · "
        "[总表](summary.csv) · [各类召回](recall_by_model.csv) · [本轮配置](../../../configs/open_pool_style.toml)",
    ])
    return "\n".join(lines) + "\n"


def main():
    with SUITE.open("rb") as source:
        suite = tomllib.load(source)
    manifest_path = ROOT / "data" / "protocols" / suite["evaluation"]["id"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary, recalls = read_runs(suite)
    records = [json.loads((ROOT / "runs" / run / "run.json").read_text(encoding="utf-8")) for run in summary.run]
    seconds = {item["shared_style_feature_seconds"] for item in records}
    if len(seconds) != 1 or None in seconds:
        raise ValueError("共用特征提取耗时缺失或不一致。")
    if any(item["verified_oof_rows"] != manifest["rows"] for item in records):
        raise ValueError("外部折外覆盖数与冻结语料不同。")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".open-pool-style-", dir=OUTPUT.parent) as temp:
        stage = Path(temp) / OUTPUT.name
        stage.mkdir()
        summary.to_csv(stage / "summary.csv", index=False, encoding="utf-8")
        recalls.to_csv(stage / "recall_by_model.csv", index_label="model", encoding="utf-8")
        plot_overview(summary, stage / "overview.png", title="Open Pool 外部数据的按题五折")
        (stage / "REPORT.md").write_text(build_report(summary, manifest, seconds.pop()), encoding="utf-8")
        stage.rename(OUTPUT)
    print(summary[["run", "balanced_accuracy", "macro_f1", "compute_seconds", "delta_ba_pp"]].to_string(index=False))
    print(OUTPUT / "REPORT.md")


if __name__ == "__main__":
    main()
