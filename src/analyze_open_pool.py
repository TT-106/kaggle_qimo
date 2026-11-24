"""Profile the downloaded Open Pool against the frozen public training data."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unicodedata

import pandas as pd
import pyarrow
import pyarrow.parquet as pq

from prepare_data import ROOT, load_settings, protocol_path, read_protocol, read_training


WORD = re.compile(r"\w+")
FLAGS = (
    "flag_kb_proxy", "flag_kb_proxy_after_ruling", "flag_gemini37_routing_risk",
    "flag_near_train_prompt", "flag_train_prompt_regen",
)
SHINGLE_SIZE = 5
MIN_JACCARD = 0.80
MIN_CONTAINMENT = 0.90
MIN_SHARED = 30


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def shingles(text):
    tokens = WORD.findall(text)
    return set(zip(*(tokens[offset:] for offset in range(SHINGLE_SIZE))))


def official_matches(pool, official):
    by_id = {}
    raw_index, normalized_index, shingle_index = defaultdict(list), defaultdict(list), defaultdict(list)
    official_lengths = []
    for index, row in enumerate(official.itertuples(index=False)):
        canon = normalize(row.text)
        by_id[row.id] = index
        raw_index[text_hash(row.text)].append(index)
        normalized_index[text_hash(canon)].append(index)
        grams = shingles(canon)
        official_lengths.append(len(grams))
        for gram in grams:
            shingle_index[gram].append(index)

    raw_hashes, normalized_hashes = [], []
    matches = []
    for row_index, row in enumerate(pool[["id", "text", "model"]].itertuples(index=False)):
        canon = normalize(row.text)
        raw, normalized = text_hash(row.text), text_hash(canon)
        raw_hashes.append(raw)
        normalized_hashes.append(normalized)
        hits = {i: ("normalized_exact", None, None, None) for i in normalized_index.get(normalized, ())}
        hits.update({i: ("raw_exact", None, None, None) for i in raw_index.get(raw, ())})
        if not hits:
            grams = shingles(canon)
            candidates = Counter(hit for gram in grams for hit in shingle_index.get(gram, ()))
            for hit, shared in candidates.items():
                jaccard = shared / (len(grams) + official_lengths[hit] - shared)
                containment = shared / min(len(grams), official_lengths[hit])
                if jaccard >= MIN_JACCARD or (shared >= MIN_SHARED and containment >= MIN_CONTAINMENT):
                    hits[hit] = ("lexical_near", jaccard, containment, shared)
        if row.id in by_id:
            hits.setdefault(by_id[row.id], ("id_only", None, None, None))
        for index, (relation, jaccard, containment, common) in sorted(hits.items()):
            candidate = official.iloc[index]
            matches.append({
                "pool_row": row_index, "pool_id": row.id, "pool_model": row.model,
                "official_id": candidate.id, "official_model": candidate.model,
                "official_prompt_id": candidate.prompt_id, "official_fold": int(candidate.fold),
                "relation": relation, "same_id": row.id == candidate.id,
                "same_label": row.model == candidate.model,
                "jaccard": jaccard, "containment": containment, "common_shingles": common,
            })
    columns = [
        "pool_row", "pool_id", "pool_model", "official_id", "official_model",
        "official_prompt_id", "official_fold", "relation", "same_id", "same_label",
        "jaccard", "containment", "common_shingles",
    ]
    return pd.Series(raw_hashes), pd.Series(normalized_hashes), pd.DataFrame(matches, columns=columns)


def field_quality(pool, schema):
    return pd.DataFrame([
        {
            "column": field.name, "type": str(field.type), "nulls": int(pool[field.name].isna().sum()),
            "blanks": int(pool[field.name].str.strip().eq("").sum()) if "string" in str(field.type) else None,
        }
        for field in schema
    ])


def features(frame):
    text = frame["text"]
    return pd.DataFrame({
        "model": frame["model"],
        "words": text.map(lambda value: len(value.split())),
        "characters": text.str.len(),
        "newlines": text.str.contains(r"[\r\n]", regex=True),
        "em_dash": text.str.contains("—", regex=False),
        "typographic_quotes": text.str.contains("[‘’“”]", regex=True),
        "markdown_markers": text.str.contains(
            r"[*][*]|\x60{3}|(?m:^[ \t]*(?:#{1,6}[ \t]+|[-*+][ \t]+|\d+[.)][ \t]+))", regex=True
        ),
    })


def style_row(values, corpus, model):
    quantiles = values["words"].quantile([0.1, 0.5, 0.9])
    return {
        "corpus": corpus, "model": model, "rows": len(values), "words_p10": float(quantiles[0.1]),
        "words_p50": float(quantiles[0.5]), "words_p90": float(quantiles[0.9]),
        "words_lt_40": int(values["words"].lt(40).sum()),
        "words_gt_250": int(values["words"].gt(250).sum()),
        "characters_median": float(values["characters"].median()),
        **{name + "_share": float(values[name].mean()) for name in (
            "newlines", "em_dash", "typographic_quotes", "markdown_markers",
        )},
    }


def style_tables(pool_features, official_features):
    styles, histogram = [], []
    for name, values in (("open_pool", pool_features), ("official_train", official_features)):
        styles.append(style_row(values, name, "all"))
        styles.extend(style_row(part, name, label) for label, part in values.groupby("model", sort=True))
        histogram.extend(
            {"corpus": name, "words": int(words), "rows": int(count)}
            for words, count in values["words"].value_counts().sort_index().items()
        )
    return pd.DataFrame(styles), pd.DataFrame(histogram)


def labeled_groups(pool, normalized_hashes, label_set, overlap):
    with_canon = pool.loc[pool["prompt_canon"].str.strip().ne("")]
    groups = with_canon.groupby("prompt_canon").agg(
        rows=("id", "size"), labels=("model", "nunique"), sources=("source_dataset", "nunique"),
    )
    model_basenames = pool["gen_served_model"].str.strip().str.rsplit("/", n=1).str[-1]
    has_served = model_basenames.ne("")
    exact_served = model_basenames.eq(pool["model"])
    other_served = has_served & model_basenames.isin(label_set) & ~exact_served
    duplicates = normalized_hashes.duplicated(keep=False)
    conflicts = pd.DataFrame({"hash": normalized_hashes, "model": pool["model"]}).groupby("hash")["model"].nunique()
    conflict_rows = normalized_hashes.isin(conflicts[conflicts.gt(1)].index)
    for_review = (
        exact_served & ~pool[list(FLAGS)].fillna(True).any(axis=1)
        & pool["prompt_evidence"].isin({"own_bank", "own_text"})
        & ~pd.Series(pool.index.isin(overlap["pool_row"])) & ~conflict_rows
    )
    pool = pool.assign(
        has_served=has_served, exact_served=exact_served, other_official_served=other_served,
        has_route=pool["gen_route"].str.strip().ne(""), candidate_screen=for_review,
    )
    return pool, groups, duplicates, conflicts, model_basenames


def summarize(pool_file, settings, assets):
    train, train_hash = read_training(assets["official_train"])
    protocol, folds = read_protocol(protocol_path(ROOT, settings["id"]), train, train_hash)
    official = train.merge(folds, on=["id", "prompt_id"], validate="one_to_one")
    label_order = protocol["label_order"]
    source = pq.ParquetFile(pool_file)
    data = source.read().to_pandas()
    if len(data) != source.metadata.num_rows:
        raise ValueError("读取行数与 Parquet 元数据不符。")
    quality = field_quality(data, source.schema_arrow)
    data["text"] = data["text"].fillna("")
    for name in ("prompt_canon", "prompt_group", "gen_served_model", "gen_route", "gen_date"):
        data[name] = data[name].fillna("")
    raw_hashes, normalized_hashes, overlap = official_matches(data, official)
    audited, panels, dup_mask, conflicts, served = labeled_groups(data, normalized_hashes, set(label_order), overlap)

    pool_features = features(data)
    styles, hist = style_tables(pool_features, features(official))
    pool_counts = audited["model"].value_counts()
    by_model = []
    for model in label_order:
        part = audited.loc[audited["model"].eq(model)]
        sources = part["source_dataset"].value_counts()
        by_model.append({
            "model": model, "official_rows": int(official["model"].eq(model).sum()),
            "pool_rows": len(part), "primary_sources": len(sources),
            "prompt_groups": int(part["prompt_group"].nunique()),
            "top_source": str(sources.index[0]) if len(sources) else "",
            "top_source_rows": int(sources.iloc[0]) if len(sources) else 0,
            "has_served_model": int(part["has_served"].sum()),
            "served_name_matches_label": int(part["exact_served"].sum()),
            "served_name_other_official_label": int(part["other_official_served"].sum()),
            "kb_proxy_rows": int(part["flag_kb_proxy"].sum()),
            "kb_proxy_after_ruling_rows": int(part["flag_kb_proxy_after_ruling"].sum()),
            "metadata_screen_rows": int(part["candidate_screen"].sum()),
        })
    source_model = pd.crosstab(audited["source_dataset"], audited["model"]).reindex(columns=label_order, fill_value=0)
    by_source = audited.groupby("source_dataset").agg(
        rows=("id", "size"), models=("model", "nunique"), source_files=("source_file", "nunique"),
        license_values=("source_license", lambda values: "; ".join(sorted(set(values.dropna())))),
        has_served_model=("has_served", "sum"), has_route=("has_route", "sum"),
        kb_proxy_rows=("flag_kb_proxy", "sum"), kb_proxy_after_ruling_rows=("flag_kb_proxy_after_ruling", "sum"),
        metadata_screen_rows=("candidate_screen", "sum"),
    ).sort_values("rows", ascending=False).reset_index()
    served_table = (
        audited.loc[audited["has_served"], ["model", "gen_served_model", "source_dataset", "other_official_served"]]
        .groupby(["model", "gen_served_model", "source_dataset", "other_official_served"], dropna=False)
        .size().rename("rows").reset_index().sort_values("rows", ascending=False)
    )
    if by_source["rows"].sum() != len(data):
        raise ValueError("来源表行数与输入不一致。检查 source_dataset 缺失值。")
    if set(pool_counts.index) != set(label_order):
        raise ValueError("外部标签与官方标签清单不同，请检查 18 类映射。")
    report = {
        "pool_file": str(pool_file), "pool_sha256": sha256(pool_file), "bytes": pool_file.stat().st_size,
        "publisher": "https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool",
        "official_data": {"path": str(assets["official_train"]), "sha256": train_hash,
                          "folds_sha256": protocol["folds_sha256"], "protocol_id": protocol["protocol_id"]},
        "rows": len(data), "columns": len(source.schema_arrow), "primary_sources": len(by_source),
        "source_files_distinct_source_and_file": int(data[["source_dataset", "source_file"]].drop_duplicates().shape[0]),
        "labels": len(label_order), "label_order": label_order,
        "quality": {
            "id_duplicate_rows": int(data["id"].duplicated(keep=False).sum()),
            "blank_text_rows": int(data["text"].str.strip().eq("").sum()),
            "raw_duplicate_rows": int(raw_hashes.duplicated(keep=False).sum()),
            "normalized_duplicate_rows": int(dup_mask.sum()),
            "normalized_duplicate_extra_rows": int(normalized_hashes.duplicated().sum()),
            "normalized_cross_label_hashes": int(conflicts.gt(1).sum()),
            "claimed_multi_source_rows": int(data["n_sources"].gt(1).sum()),
            "stored_word_count_mismatch_rows": int(pool_features["words"].ne(data["n_words"]).sum()),
        },
        "prompt_groups": {
            "raw_count": int(data["prompt_group"].replace("", pd.NA).nunique()),
            "canonical_groups": len(panels), "unresolved_rows": int(data["prompt_canon"].str.strip().eq("").sum()),
            "canonical_multi_source_groups": int(panels["sources"].gt(1).sum()),
            "canonical_labels": {int(k): int(v) for k, v in panels["labels"].value_counts().sort_index().items()},
            "stored_panel_label_count_mismatches": int((
                data.loc[data["prompt_canon"].ne(""), "panel_labels"] !=
                data.loc[data["prompt_canon"].ne(""), "prompt_canon"].map(panels["labels"])
            ).sum()),
            "literal_official_id_overlap_rows": int(
                (data["prompt_group"].isin(official["prompt_id"]) |
                 data["prompt_canon"].isin(official["prompt_id"])).sum()
            ),
            "evidence": {key: int(n) for key, n in data["prompt_evidence"].value_counts().items()},
        },
        "provenance": {
            "source_license_rows": {str(key): int(n) for key, n in data["source_license"].value_counts().items()},
            "served_model_missing_rows": int((~audited["has_served"]).sum()),
            "route_missing_rows": int((~audited["has_route"]).sum()),
            "date_missing_rows": int(data["gen_date"].str.strip().eq("").sum()),
            "served_short_name_matches_rows": int(audited["exact_served"].sum()),
            "served_is_another_official_label_rows": int(audited["other_official_served"].sum()),
            "served_other_name_rows": int((audited["has_served"] & ~served.isin(label_order)).sum()),
            "flag_true_rows": {flag: int(data[flag].sum()) for flag in FLAGS},
            "proxy_flag_without_after_ruling_flag_rows": int(
                (data["flag_kb_proxy"] & ~data["flag_kb_proxy_after_ruling"]).sum()
            ),
            "reasoning_evidence_rows": {
                str(key): int(n) for key, n in data["gen_reasoning_evidence"].value_counts().items()
            },
            "metadata_screen_rows": int(audited["candidate_screen"].sum()),
        },
        "official_overlap": {
            "pairs": len(overlap), "matched_pool_rows": int(overlap["pool_row"].nunique()),
            "matched_official_articles": int(overlap["official_id"].nunique()),
            "id_pairs": int(overlap["same_id"].sum()),
            "by_relation_rows": {
                key: int(part["pool_row"].nunique()) for key, part in overlap.groupby("relation", sort=True)
            },
            "shingle_size": SHINGLE_SIZE, "min_jaccard": MIN_JACCARD,
            "min_containment": MIN_CONTAINMENT, "min_shared_for_containment": MIN_SHARED,
        },
        "environment": {"pandas": pd.__version__, "pyarrow": pyarrow.__version__},
    }
    tables = {
        "field_quality.csv": quality, "by_model.csv": pd.DataFrame(by_model), "by_source.csv": by_source,
        "source_model.csv": source_model.reset_index(), "served_model_labels.csv": served_table,
        "style_by_model.csv": styles, "word_count_histogram.csv": hist, "official_overlap.csv": overlap,
    }
    return report, tables


def main():
    parser = argparse.ArgumentParser(description="审计外部 Open Pool 和官方公开训练数据。")
    parser.add_argument("--pool", type=Path, default=ROOT / "data" / "external" / "ntl_open_pool.parquet")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "external" / "ntl_open_pool_audit")
    args = parser.parse_args()
    archive = args.output.resolve()
    if archive.exists():
        parser.error(f"审计目录已存在。复跑时用 --output 指向新目录：{archive}")
    settings, assets = load_settings()
    report, tables = summarize(args.pool.resolve(), settings, assets)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".open-pool-audit-", dir=archive.parent) as temp:
        staging = Path(temp) / archive.name
        staging.mkdir()
        for name, table in tables.items():
            table.to_csv(staging / name, index=False, encoding="utf-8")
        (staging / "summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(archive)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
