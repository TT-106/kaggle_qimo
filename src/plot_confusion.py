"""Plot the validated 18-class confusion tables for a single experiment."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd


def plot_confusion(directory):
    counts = pd.read_csv(directory / "confusion_counts.csv", index_col="true_model")
    normalized = pd.read_csv(directory / "confusion_by_true.csv", index_col="true_model")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })

    fig, ax = plt.subplots(figsize=(12.8, 10.2), layout="constrained")
    colors = ax.imshow(normalized.to_numpy(), vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(counts.columns)), counts.columns, rotation=52, ha="right", fontsize=8)
    ax.set_yticks(range(len(counts.index)), counts.index, fontsize=8)
    ax.set_xlabel("预测模型", labelpad=10)
    ax.set_ylabel("真实模型", labelpad=10)
    ax.set_title(f"{directory.name}  混淆矩阵", loc="left", fontsize=15, fontweight="bold", pad=16)
    colorbar = fig.colorbar(colors, ax=ax, shrink=.80, pad=.025, format=PercentFormatter(1))
    colorbar.set_label("占该真实类别的比例", rotation=270, labelpad=17)

    for row, values in enumerate(counts.to_numpy()):
        for col, count in enumerate(values):
            if count:
                color = "white" if normalized.iat[row, col] > .55 else "#163347"
                ax.text(col, row, str(count), ha="center", va="center", color=color, fontsize=7)
    output = directory / "figures" / "confusion.png"
    output.parent.mkdir(exist_ok=True)
    with output.open("xb") as stream:
        fig.savefig(stream, format="png", dpi=165, facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绘制每次实验的混淆矩阵。")
    parser.add_argument("run_directory", type=Path)
    plot_confusion(parser.parse_args().run_directory.resolve())
