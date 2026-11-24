"""Freeze the 18-label Open Pool cohort and its grouped validation folds."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import pandas as pd
import pyarrow.parquet as pq
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

from prepare_data import ROOT, TRAIN_COLUMNS, check_folds, load_settings, protocol_path, read_protocol


PROTOCOL_ID = "open-pool-prompt-group-5fold-v1"
POOL_FILE = Path("data/external/ntl_open_pool.parquet")
EVIDENCE = ("own_bank", "own_text")
FOLDS = 5
SEED = 42
GROUP_FIELDS = ("prompt_group", "prompt_canon")
POOL_COLUMNS = ["id", "text", "model", *GROUP_FIELDS, "prompt_evidence", "source_dataset", "gen_served_model"]


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def connected_prompts(groups):
    """Connect canonical IDs whenever the same original prompt ID links them."""
    parent = {key: key for key in groups["prompt_canon"].unique()}

    def root_of(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    links = groups[list(GROUP_FIELDS)].drop_duplicates()
    for _, canon in links.groupby("prompt_group", sort=False)["prompt_canon"]:
        keys = iter(canon)
        root = root_of(next(keys))
        for other in keys:
            other_root = root_of(other)
            first, last = sorted((root, other_root))
            parent[last] = first
            root = first
    return groups["prompt_canon"].map({key: root_of(key) for key in parent})


def select_open_pool(source):
    table = pq.ParquetFile(source)
    flags = [field.name for field in table.schema_arrow if field.name.startswith("flag_")]
    if not flags:
        raise ValueError("Open Pool 的风险标记列缺失。")
    pool = table.read(columns=POOL_COLUMNS + flags).to_pandas()
    grouped = pool[list(GROUP_FIELDS)].fillna("").apply(lambda series: series.str.strip().ne("")).all(axis=1)
    unflagged = ~pool[flags].fillna(True).any(axis=1)
    direct = pool["prompt_evidence"].isin(EVIDENCE)
    selected = pool.loc[grouped & unflagged & direct, POOL_COLUMNS].sort_values("id").reset_index(drop=True)
    for column in ["id", "text", "model", *GROUP_FIELDS]:
        selected[column] = selected[column].astype("string")
        if selected[column].isna().any() or selected[column].str.strip().eq("").any():
            raise ValueError(f"筛选后的 {column} 含有空值。")
    if selected["id"].duplicated().any():
        raise ValueError("筛选后的 Open Pool 含有重复文章 ID。")
    if selected.empty:
        raise ValueError("Open Pool 筛选结果为空。")
    selected["prompt_id"] = connected_prompts(selected).astype("string")
    scope = {
        "raw_rows": len(pool),
        "retained_after_prompt_ids": int(grouped.sum()),
        "retained_after_risk_flags": int((grouped & unflagged).sum()),
        "retained_after_prompt_evidence": len(selected),
        "required_prompt_evidence": list(EVIDENCE),
        "rejected_risk_flag_columns": flags,
        "selected_sources": int(selected["source_dataset"].nunique()),
        "selected_raw_prompt_ids": int(selected["prompt_group"].nunique()),
        "selected_canonical_prompt_ids": int(selected["prompt_canon"].nunique()),
        "selected_missing_served_model": int(selected["gen_served_model"].fillna("").str.strip().eq("").sum()),
    }
    return selected, scope


def check_original_prompt_groups(data, folds):
    pair = data[["id", *GROUP_FIELDS]].merge(folds[["id", "fold"]], on="id", validate="one_to_one")
    if len(pair) != len(data):
        raise ValueError("原始题组核验未覆盖全部外部样本。")
    result = {field: int(pair.groupby(field)["fold"].nunique().gt(1).sum()) for field in GROUP_FIELDS}
    if any(result.values()):
        raise ValueError(f"Open Pool 有题组跨折。{result}")
    return result


def freeze_open_pool(directory, source, selected, scope, labels):
    data = selected[list(TRAIN_COLUMNS)]
    if sorted(data["model"].unique()) != labels:
        raise ValueError("外部池筛选后的类别未覆盖官方 18 类。")
    folds = data[["id", "prompt_id"]].copy()
    folds["fold"] = 0
    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    for number, (_, validation) in enumerate(splitter.split(data, data["model"], data["prompt_id"]), start=1):
        folds.iloc[validation, folds.columns.get_loc("fold")] = number
    checks = check_folds(data, folds, FOLDS)
    original_checks = check_original_prompt_groups(selected, folds)
    raw = folds.to_csv(index=False, lineterminator="\n").encode("utf-8")
    fold_labels = data[["id", "model"]].merge(folds[["id", "fold"]], on="id", validate="one_to_one")
    by_fold_label = pd.crosstab(fold_labels.fold, fold_labels.model).reindex(columns=labels)
    manifest = {
        "protocol_id": directory.name, "input_kind": "open_pool", "source_relative_path": POOL_FILE.as_posix(),
        "source_train": str(source), "target_labels": "publisher model column",
        "train_sha256": file_sha256(source), "folds_sha256": hashlib.sha256(raw).hexdigest(),
        "n_splits": FOLDS, "splitter": "StratifiedGroupKFold", "seed": SEED, "shuffle": True,
        "group_column": "prompt_id", "group_method": "connected components of original and canonical prompt IDs",
        "primary_metric": "balanced_accuracy", "label_order": labels,
        "rows": len(data), "prompts": int(folds["prompt_id"].nunique()),
        "largest_prompt_component_rows": int(folds["prompt_id"].value_counts().max()),
        "class_counts": {label: int(count) for label, count in data["model"].value_counts().items()},
        "fold_checks": checks, "original_prompt_ids_in_multiple_folds": original_checks,
        "cohort": scope,
        "environment": {"python": sys.version.split()[0], "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
    }
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".prepare-open-pool-", dir=directory.parent) as temp:
        stage = Path(temp) / directory.name
        stage.mkdir()
        (stage / "folds.csv").write_bytes(raw)
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        by_fold_label.to_csv(stage / "validation_class_counts.csv", index_label="fold", encoding="utf-8")
        stage.rename(directory)


def load_open_pool_protocol(directory, source):
    selected, scope = select_open_pool(source)
    data = selected[list(TRAIN_COLUMNS)]
    manifest, folds = read_protocol(directory, data, file_sha256(source))
    if scope != manifest["cohort"]:
        raise ValueError("Open Pool 重新筛选的范围与冻结清单不同。")
    if check_original_prompt_groups(selected, folds) != manifest["original_prompt_ids_in_multiple_folds"]:
        raise ValueError("Open Pool 原始题号检查与冻结清单不同。")
    return data, manifest, folds


def main():
    parser = argparse.ArgumentParser(description="创建或复核 Open Pool 的 18 类按题五折。")
    parser.add_argument("--freeze", action="store_true", help="首次创建外部验证协议。")
    args = parser.parse_args()
    settings, _ = load_settings()
    directory = protocol_path(ROOT, PROTOCOL_ID)
    source = ROOT / POOL_FILE
    if not directory.exists():
        if not args.freeze:
            parser.error("协议尚待冻结，请加上 --freeze。")
        official = json.loads((protocol_path(ROOT, settings["id"]) / "manifest.json").read_text(encoding="utf-8"))
        selected, scope = select_open_pool(source)
        freeze_open_pool(directory, source, selected, scope, official["label_order"])
    _, manifest, _ = load_open_pool_protocol(directory, source)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
