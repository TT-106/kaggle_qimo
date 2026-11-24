"""Freeze or verify the shared evaluation protocol for this project."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
import tomllib

import pandas as pd
import sklearn
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
TRAIN_COLUMNS = ("id", "prompt_id", "text", "model")
FOLD_COLUMNS = ("id", "prompt_id", "fold")


def load_settings(root=ROOT):
    with (root / "configs" / "project.toml").open("rb") as stream:
        config = tomllib.load(stream)
    shared = config["shared"]
    assets = {key: Path(shared["root"]) / value for key, value in shared.items() if key != "root"}
    return config["evaluation"], assets


def protocol_path(root, protocol_id):
    if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", protocol_id):
        raise ValueError(f"协议编号须由小写字母、数字和连字符组成。收到 {protocol_id!r}")
    return root / "data" / "protocols" / protocol_id


def read_training(path):
    raw = path.read_bytes()
    data = pd.read_csv(io.BytesIO(raw), dtype="string", keep_default_na=False)
    if tuple(data.columns) != TRAIN_COLUMNS or data.empty:
        raise ValueError(f"训练文件应含 {', '.join(TRAIN_COLUMNS)} 四列及数据行。检查 {path}")
    if data.isna().any().any() or any(data[col].str.strip().eq("").any() for col in TRAIN_COLUMNS):
        raise ValueError("训练文件存在缺失或空白字段，请核对官方数据副本。")
    if data["id"].duplicated().any():
        raise ValueError("训练文件有重复的 id，请核对官方数据副本。")
    return data, hashlib.sha256(raw).hexdigest()


def make_folds(data, n_splits):
    folds = data[["id", "prompt_id"]].copy()
    folds["fold"] = 0
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (_, valid_rows) in enumerate(splitter.split(data, groups=data["prompt_id"]), start=1):
        folds.iloc[valid_rows, folds.columns.get_loc("fold")] = fold
    return folds


def check_folds(data, folds, n_splits):
    if tuple(folds.columns) != FOLD_COLUMNS or folds["id"].duplicated().any():
        raise ValueError("folds.csv 应有唯一 id 和 id、prompt_id、fold 三列。")
    if set(data["id"]) != set(folds["id"]):
        raise ValueError("folds.csv 的 id 与官方训练文件不同。")
    if set(folds["fold"]) != set(range(1, n_splits + 1)):
        raise ValueError(f"folds.csv 的折号应为 1 至 {n_splits}。")
    aligned = data[["id", "prompt_id", "model"]].merge(
        folds, on=["id", "prompt_id"], validate="one_to_one"
    )
    if len(aligned) != len(data):
        raise ValueError("folds.csv 中的 prompt_id 与官方训练文件不同。")
    if not aligned.groupby("prompt_id")["fold"].nunique().eq(1).all():
        raise ValueError("发现一个 Prompt 被分到多折，请先修正分折。")

    labels = set(data["model"])
    stats = []
    for fold in range(1, n_splits + 1):
        train = aligned[aligned["fold"] != fold]
        valid = aligned[aligned["fold"] == fold]
        shared = set(train["prompt_id"]) & set(valid["prompt_id"])
        if shared or set(valid["model"]) != labels or set(train["model"]) != labels:
            raise ValueError(f"第 {fold} 折有 Prompt 重叠或类目缺失。请检查分折清单。")
        stats.append({
            "fold": fold,
            "train_rows": len(train),
            "validation_rows": len(valid),
            "train_prompts": int(train["prompt_id"].nunique()),
            "validation_prompts": int(valid["prompt_id"].nunique()),
            "validation_classes": int(valid["model"].nunique()),
            "shared_prompts": len(shared),
        })
    return stats


def read_protocol(directory, data, train_sha256):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    raw_folds = (directory / "folds.csv").read_bytes()
    if (
        manifest["protocol_id"] != directory.name
        or manifest["train_sha256"] != train_sha256
        or manifest["folds_sha256"] != hashlib.sha256(raw_folds).hexdigest()
        or manifest["label_order"] != sorted(data["model"].unique())
    ):
        raise ValueError(f"冻结协议与当前数据、标签或分折指纹不符。检查 {directory}")
    folds = pd.read_csv(
        io.BytesIO(raw_folds), dtype={"id": "string", "prompt_id": "string", "fold": "int64"}
    )
    if check_folds(data, folds, manifest["n_splits"]) != manifest["fold_checks"]:
        raise ValueError(f"冻结协议的分折统计与实际清单不符。检查 {directory}")
    return manifest, folds


def freeze_protocol(directory, source, data, train_sha256, n_splits):
    folds = make_folds(data, n_splits)
    checks = check_folds(data, folds, n_splits)
    raw_folds = folds.to_csv(index=False, lineterminator="\n").encode("utf-8")
    labels = sorted(data["model"].unique())
    manifest = {
        "protocol_id": directory.name,
        "splitter": "GroupKFold",
        "group_column": "prompt_id",
        "n_splits": n_splits,
        "primary_metric": "balanced_accuracy",
        "source_train": str(source),
        "train_sha256": train_sha256,
        "folds_sha256": hashlib.sha256(raw_folds).hexdigest(),
        "rows": len(data),
        "prompts": int(data["prompt_id"].nunique()),
        "label_order": labels,
        "class_counts": {label: int(count) for label, count in data["model"].value_counts().items()},
        "fold_checks": checks,
        "environment": {"python": sys.version.split()[0], "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
    }
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".prepare-", dir=directory.parent) as stage:
        staged = Path(stage) / directory.name
        staged.mkdir()
        (staged / "folds.csv").write_bytes(raw_folds)
        (staged / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        staged.rename(directory)


def main():
    parser = argparse.ArgumentParser(description="核验或首次冻结 Name That LLM 评价协议。")
    parser.add_argument("--freeze", action="store_true", help="首次创建协议文件。")
    args = parser.parse_args()
    try:
        settings, assets = load_settings()
        source = assets["official_train"]
        data, train_sha256 = read_training(source)
        directory = protocol_path(ROOT, settings["id"])
        if data["model"].nunique() != settings["classes"]:
            raise ValueError(f"训练数据的类别数应为 {settings['classes']}。")
        if not directory.exists():
            if not args.freeze:
                raise ValueError("协议尚待冻结，请用 --freeze 创建分折和数据指纹。")
            freeze_protocol(directory, source, data, train_sha256, settings["folds"])
        manifest, _ = read_protocol(directory, data, train_sha256)
        if settings["folds"] != manifest["n_splits"]:
            raise ValueError("配置折数与冻结协议不同。新划分请使用新的协议编号。")
        labels = manifest["label_order"]
        metric_check = balanced_accuracy_score(data["model"], [labels[0]] * len(data))
        if abs(metric_check - 1 / len(labels)) > 1e-12:
            raise ValueError("常量预测的 BA 自检失败。请核对标签和评价实现。")
        print(json.dumps({
            "status": "verified",
            "manifest": str(directory / "manifest.json"),
            "rows": manifest["rows"],
            "prompts": manifest["prompts"],
            "classes": len(labels),
            "fold_checks": manifest["fold_checks"],
            "constant_prediction_ba_self_check": metric_check,
            "shared_assets_present": {key: path.exists() for key, path in assets.items()},
        }, ensure_ascii=False, indent=2))
    except (OSError, KeyError, ValueError) as error:
        raise SystemExit(f"数据核验失败。{error}") from None


if __name__ == "__main__":
    main()
