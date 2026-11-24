"""Render the class, length, and style comparisons from an Open Pool audit."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
POOL_COLOR = "#176B70"
OFFICIAL_COLOR = "#B46538"


def render(audit, output):
    by_model = pd.read_csv(audit / "by_model.csv").sort_values("pool_rows", ascending=True)
    words = pd.read_csv(audit / "word_count_histogram.csv")
    styles = pd.read_csv(audit / "style_by_model.csv")
    overall = styles.loc[styles["model"].eq("all")].set_index("corpus")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#D6E1E0",
        "grid.alpha": 0.75,
    })
    figure = plt.figure(figsize=(13.6, 8.1), layout="constrained")
    layout = figure.add_gridspec(2, 2, width_ratios=[1.14, 1])
    labels_axis = figure.add_subplot(layout[:, 0])
    length_axis = figure.add_subplot(layout[0, 1])
    style_axis = figure.add_subplot(layout[1, 1])
    figure.suptitle("Open Pool 与官方训练集", fontsize=20, fontweight="bold")

    labels = by_model["model"]
    pool = by_model["pool_rows"] / by_model["pool_rows"].sum()
    official = by_model["official_rows"] / by_model["official_rows"].sum()
    labels_axis.barh(labels, pool, color=POOL_COLOR, height=0.65)
    labels_axis.scatter(official, labels, color=OFFICIAL_COLOR, s=24, zorder=3)
    count_x = pool.max() * 1.21
    for label, count in zip(labels, by_model["pool_rows"], strict=True):
        labels_axis.text(count_x, label, f"{count:,}", ha="right", va="center", fontsize=9)
    labels_axis.text(count_x, 1.015, "Open Pool 篇数", transform=labels_axis.get_xaxis_transform(), ha="right", fontsize=9)
    labels_axis.set_xlim(0, pool.max() * 1.24)
    labels_axis.xaxis.set_major_formatter(PercentFormatter(1))
    labels_axis.set_xlabel("各语料内类别占比")
    labels_axis.set_title("18 类样本覆盖", loc="left")
    labels_axis.grid(axis="x")
    labels_axis.set_axisbelow(True)

    for corpus, name, color in (("open_pool", "Open Pool", POOL_COLOR), ("official_train", "官方训练集", OFFICIAL_COLOR)):
        group = words.loc[words["corpus"].eq(corpus)].sort_values("words")
        length_axis.step(group["words"], group["rows"].cumsum() / group["rows"].sum(), where="post", label=name, color=color, lw=2)
    length_axis.set_title("文本词数", loc="left")
    length_axis.set_xlabel("按空白分隔的词数")
    length_axis.set_ylabel("累积比例")
    length_axis.yaxis.set_major_formatter(PercentFormatter(1))
    length_axis.grid(axis="y")
    length_axis.legend(loc="lower right")

    feature_names = ["em_dash_share", "typographic_quotes_share", "newlines_share", "markdown_markers_share"]
    positions = range(len(feature_names))
    style_axis.barh([i - .18 for i in positions], overall.loc["open_pool", feature_names], .32, color=POOL_COLOR)
    style_axis.barh([i + .18 for i in positions], overall.loc["official_train", feature_names], .32, color=OFFICIAL_COLOR)
    style_axis.set_yticks(list(positions), ["长破折号", "弯引号", "换行", "Markdown 标记"])
    style_axis.invert_yaxis()
    style_axis.xaxis.set_major_formatter(PercentFormatter(1))
    style_axis.set_xlabel("含该标记的文本比例")
    style_axis.set_title("原始文本的格式特征", loc="left")
    style_axis.grid(axis="x")
    style_axis.set_axisbelow(True)
    for index, name in enumerate(feature_names):
        if overall[name].max() == 0:
            style_axis.text(0.005, index, "两组均为 0%", va="center", color="#515B5C", fontsize=9)
    with output.open("xb") as stream:
        figure.savefig(stream, format="png", dpi=160, facecolor="white")
    plt.close(figure)
    return output


def main():
    parser = argparse.ArgumentParser(description="从已有审计表绘制 Open Pool 对比图。")
    parser.add_argument("--audit", type=Path, default=ROOT / "data" / "external" / "ntl_open_pool_audit")
    parser.add_argument("--output", type=Path, help="为图指定新文件路径。")
    args = parser.parse_args()
    audit = args.audit.resolve()
    print(render(audit, args.output or audit / "overview.png"))


if __name__ == "__main__":
    main()
