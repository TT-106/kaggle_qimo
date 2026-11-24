"""Verify that a run can be reviewed and that its metrics match its predictions."""

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import precision_recall_fscore_support

from prepare_data import ROOT, load_settings, protocol_path, read_protocol, read_training


BASE_FILES = ("run.json", "run.log", "review.md")
SCORED_FILES = (
    "oof.parquet",
    "metrics.json",
    "by_fold.csv",
    "by_model.csv",
    "confusion_counts.csv",
    "confusion_by_true.csv",
    "figures/confusion.png",
)
ROW_COLUMNS = ["id", "prompt_id", "fold", "y_true"]
PROTOCOL_KEYS = ("protocol_id", "train_sha256", "folds_sha256")


def require_files(directory, names):
    missing = [name for name in names if not (directory / name).is_file() or not (directory / name).stat().st_size]
    if missing:
        raise ValueError(f"{directory.name} 缺少有效产物：{', '.join(missing)}。")


def read_oof(path, expected, labels):
    oof = pd.read_parquet(path, columns=[*ROW_COLUMNS, "y_pred"]).sort_values("id").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(oof[ROW_COLUMNS], expected, check_dtype=False, check_exact=True)
    except AssertionError:
        raise ValueError(f"{path.name} 的 id、题号、折号或真值与冻结清单不同。") from None
    if not oof["y_pred"].isin(labels).all():
        raise ValueError(f"{path.name} 中发现空预测或超出标签清单的预测。")
    return oof


def metrics_for(oof, labels):
    truth, prediction = oof["y_true"], oof["y_pred"]
    return {
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(truth, prediction)),
    }


def diagnostic_tables(oof, labels):
    truth, prediction = oof["y_true"], oof["y_pred"]
    counts = pd.DataFrame(
        confusion_matrix(truth, prediction, labels=labels),
        index=pd.Index(labels, name="true_model"),
        columns=labels,
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=labels, zero_division=0
    )
    by_model = pd.DataFrame({
        "model": labels,
        "support": support,
        "predicted": counts.sum(axis=0).to_numpy(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })
    by_fold = pd.DataFrame([
        {"fold": int(fold), "rows": len(part), **metrics_for(part, labels)}
        for fold, part in oof.groupby("fold", sort=True)
    ])
    return {
        "by_fold.csv": by_fold,
        "by_model.csv": by_model,
        "confusion_counts.csv": counts.reset_index(),
        "confusion_by_true.csv": counts.div(counts.sum(axis=1), axis=0).reset_index(),
    }


def paired_changes(oof, baseline):
    pairs = oof.copy()
    pairs.insert(4, "baseline_pred", baseline["y_pred"])
    before = pairs["baseline_pred"] == pairs["y_true"]
    after = pairs["y_pred"] == pairs["y_true"]
    pairs["outcome"] = "both_wrong"
    pairs.loc[before & after, "outcome"] = "both_correct"
    pairs.loc[~before & after, "outcome"] = "corrected"
    pairs.loc[before & ~after, "outcome"] = "regressed"
    return pairs


def compare_table(path, expected):
    actual = pd.read_csv(path, dtype={name: "string" for name in ("id", "prompt_id")})
    if "id" in actual.columns:
        actual = actual.sort_values("id").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=0, atol=1e-8)
    except AssertionError as error:
        difference = str(error).splitlines()[0]
        raise ValueError(f"{path.name} 与逐篇预测复算结果不同。{difference}") from None


def check_scored_run(directory, metadata, manifest, data, folds):
    require_files(directory, SCORED_FILES)
    if any(metadata[key] != manifest[key] for key in PROTOCOL_KEYS):
        raise ValueError("运行记录的协议或数据指纹与冻结清单不同。")
    labels = manifest["label_order"]
    expected = folds.merge(data[["id", "model"]], on="id", validate="one_to_one")
    expected = expected.rename(columns={"model": "y_true"}).sort_values("id").reset_index(drop=True)
    oof = read_oof(directory / "oof.parquet", expected[ROW_COLUMNS], labels)
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    for name, value in metrics_for(oof, labels).items():
        if not math.isclose(metrics[name], value, rel_tol=0, abs_tol=1e-8):
            raise ValueError(f"metrics.json 的 {name} 与逐篇预测复算结果不同。")
    for name, table in diagnostic_tables(oof, labels).items():
        compare_table(directory / name, table)

    baseline_id = metadata["baseline_run"]
    if baseline_id is not None:
        if baseline_id in {"", ".", "..", directory.name} or Path(baseline_id).name != baseline_id:
            raise ValueError("baseline_run 应填写同一 runs 目录内另一次运行的编号。")
        baseline_dir = directory.parent / baseline_id
        baseline_meta = json.loads((baseline_dir / "run.json").read_text(encoding="utf-8"))
        if (
            baseline_meta["id"] != baseline_id
            or baseline_meta["kind"] != "grouped_oof"
            or baseline_meta["status"] != "completed"
            or baseline_meta["exit_code"] != 0
            or any(baseline_meta[key] != metadata[key] for key in PROTOCOL_KEYS)
        ):
            raise ValueError("对照运行必须是使用相同协议的完整折外实验。")
        baseline = read_oof(baseline_dir / "oof.parquet", expected[ROW_COLUMNS], labels)
        compare_table(directory / "paired_changes.csv", paired_changes(oof, baseline))
    return len(oof)


def check_run(directory):
    require_files(directory, BASE_FILES)
    metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    kind, status = metadata["kind"], metadata["status"]
    if metadata["id"] != directory.name or kind not in {"grouped_oof", "smoke"}:
        raise ValueError("运行编号须与目录一致，kind 须为 grouped_oof 或 smoke。")
    if status not in {"completed", "failed", "incomplete"}:
        raise ValueError("请在执行结束后检查归档，并记录 completed、failed 或 incomplete。")
    if status == "completed" and metadata["exit_code"] != 0:
        raise ValueError("已完成的运行须保存退出码 0。")
    if kind == "grouped_oof" and status == "completed":
        _, assets = load_settings()
        data, train_sha256 = read_training(assets["official_train"])
        frozen = protocol_path(ROOT, metadata["protocol_id"])
        manifest, folds = read_protocol(frozen, data, train_sha256)
        rows = check_scored_run(directory, metadata, manifest, data, folds)
        return f"{rows} 条折外预测及指标、混淆矩阵已核验"
    return "日志与回顾已归档"


def main():
    parser = argparse.ArgumentParser(description="核验一次运行的实验归档。")
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    try:
        directory = args.run_directory.resolve()
        print(f"{directory.name}：{check_run(directory)}。")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"实验归档核验失败。{error}") from None


if __name__ == "__main__":
    main()
