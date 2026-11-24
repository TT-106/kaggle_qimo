"""Run a fixed official-data baseline and produce a checked OOF archive."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import logging
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from time import perf_counter
import tomllib
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize
from threadpoolctl import threadpool_limits

from check_run import ROW_COLUMNS, check_scored_run, diagnostic_tables, metrics_for, paired_changes, read_oof
from prepare_data import ROOT, load_settings, protocol_path, read_protocol, read_training


CONFIG = ROOT / "configs" / "traditional_baselines.toml"
TREE_SITE = ROOT / "data" / "cache" / "tree_dependencies"
SENTENCES = re.compile(r"(?<=[.!?])\s+")
PUNCTUATION = {
    "comma": ",", "period": ".", "colon": ":", "semicolon": ";", "em_dash": "—",
    "hyphen": "-", "question": "?", "exclamation": "!", "curly_quotes": "‘’“”",
    "straight_quotes": "'\"", "parentheses": "()",
}
STYLE_COLUMNS = [
    "characters", "words", "mean_word_length", "unique_word_fraction",
    "sentences", "mean_words_per_sentence", "std_words_per_sentence",
    *(name + "_per_100_chars" for name in PUNCTUATION),
    "uppercase_per_100_chars", "digits_per_100_chars",
]


def style_values(text):
    words = text.split()
    lengths = [len(part.split()) for part in SENTENCES.split(text.strip()) if part]
    counts = Counter(text)
    per_100 = 100 / max(len(text), 1)
    return [
        len(text), len(words), sum(map(len, words)) / max(len(words), 1),
        len({word.casefold() for word in words}) / max(len(words), 1),
        len(lengths), float(np.mean(lengths or [0])), float(np.std(lengths or [0])),
        *(sum(counts[char] for char in chars) * per_100 for chars in PUNCTUATION.values()),
        sum(count for char, count in counts.items() if char.isupper()) * per_100,
        sum(count for char, count in counts.items() if char.isdigit()) * per_100,
    ]


def style_frame(texts):
    return pd.DataFrame([style_values(text) for text in texts], columns=STYLE_COLUMNS, dtype=float)


def resolved_parameters(config, spec):
    kind = spec["features"]
    vectorizer = {}
    if kind in {"char", "word", "char_style"}:
        vectorizer = config["tfidf"]["common"] | config["tfidf"]["word" if kind == "word" else "char"]
        vectorizer["ngram_range"] = tuple(vectorizer["ngram_range"])
    model = spec["classifier"]
    if model == "logistic":
        classifier = config["logistic"] | {"class_weight": None}
    elif model == "dummy":
        classifier = {"strategy": "most_frequent"}
    else:
        classifier = config["tree"]["common"] | config["tree"][model] | {"n_jobs": config["cpu"]["threads"]}
    return vectorizer, classifier


def feature_matrices(train, valid, spec, vectorizer_options):
    kind = spec["features"]
    if kind == "constant":
        return np.zeros((len(train), 1)), np.zeros((len(valid), 1)), ["constant"]
    if kind in {"style", "char_style"}:
        style_train, style_valid = style_frame(train), style_frame(valid)
        if spec["classifier"] == "logistic":
            scaler = StandardScaler()
            style_train = scaler.fit_transform(style_train)
            style_valid = scaler.transform(style_valid)
        if kind == "style":
            return style_train, style_valid, STYLE_COLUMNS

    vectorizer = TfidfVectorizer(**vectorizer_options)
    fit = vectorizer.fit_transform(train)
    hold = vectorizer.transform(valid)
    names = vectorizer.get_feature_names_out().tolist()
    if kind == "char_style":
        fit = sp.hstack([fit, sp.csr_matrix(normalize(style_train))], format="csr")
        hold = sp.hstack([hold, sp.csr_matrix(normalize(style_valid))], format="csr")
        names.extend("style:" + name for name in STYLE_COLUMNS)
    return fit, hold, names


def classifier_for(kind, params):
    if kind == "dummy":
        return DummyClassifier(**params)
    if kind == "logistic":
        return LogisticRegression(**params)
    if kind == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(**params)
    if kind == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params)
    raise ValueError(f"未知分类器 {kind}")


def layer_versions(model):
    framework = ["numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "threadpoolctl"]
    if model == "xgboost":
        import xgboost
        versions = {"xgboost": xgboost.__version__, "xgboost_path": xgboost.__file__}
    elif model == "lightgbm":
        import lightgbm
        versions = {"lightgbm": lightgbm.__version__, "lightgbm_path": lightgbm.__file__}
    else:
        versions = {}
    return {name: metadata.version(name) for name in framework} | versions


def scores_for_fold(model, fit, hold, labels, class_count, logger):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        before = perf_counter()
        model.fit(fit, labels)
        fit_seconds = perf_counter() - before
        before = perf_counter()
        probability = model.predict_proba(hold)
        predict_seconds = perf_counter() - before
    for event in caught:
        logger.warning("%s %s", event.category.__name__, event.message)
    if not np.array_equal(model.classes_, np.arange(class_count)):
        raise ValueError("分类器的概率列与冻结标签顺序不同。")
    if probability.shape != (hold.shape[0], class_count):
        raise ValueError("预测行数或类别数与验证数据不符。")
    if not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1), 1, atol=1e-6):
        raise ValueError("预测概率存在非有限值或行和异常。")
    return probability, fit_seconds, predict_seconds, [event.category.__name__ for event in caught]


def verify_dummy_folds(oof, labels):
    expected = 1 / len(labels)
    for fold, part in oof.groupby("fold"):
        ba = metrics_for(part, labels)["balanced_accuracy"]
        if part.y_pred.nunique() != 1 or not np.isclose(ba, expected, rtol=0, atol=1e-12):
            raise ValueError(f"第 {fold} 折未复现多数类常量预测的理论 BA。")


def top_coefficients(model, names, labels, fold):
    coefficients = getattr(model, "coef_", None)
    if coefficients is None:
        return []
    rows = []
    width = min(6, len(names))
    for label, weights in zip(labels, coefficients, strict=True):
        indices = np.argpartition(weights, -width)[-width:]
        for index in sorted(indices, key=lambda index: (-weights[index], names[index])):
            rows.append({"fold": fold, "model": label, "feature": names[index], "coefficient": float(weights[index])})
    return rows


def fit_grouped_folds(data, folds, labels, spec, vectorizer, params, threads, logger):
    dataset = data.merge(folds, on=["id", "prompt_id"], validate="one_to_one")
    label_to_index = {label: index for index, label in enumerate(labels)}
    blocks, details, weights = [], [], []
    label_array = np.asarray(labels)
    with threadpool_limits(limits=threads):
        for fold, valid in dataset.groupby("fold", sort=True):
            train = dataset[dataset.fold.ne(fold)]
            before = perf_counter()
            fit, hold, names = feature_matrices(train.text, valid.text, spec, vectorizer)
            features_seconds = perf_counter() - before
            clf = classifier_for(spec["classifier"], params)
            prob, fit_seconds, predict_seconds, warning_names = scores_for_fold(
                clf, fit, hold, train.model.map(label_to_index).to_numpy(), len(labels), logger
            )
            predictions = valid[["id", "prompt_id", "fold", "model"]].rename(columns={"model": "y_true"})
            predictions["y_pred"] = label_array[prob.argmax(axis=1)]
            probability = pd.DataFrame(prob, columns=["p_" + label for label in labels], index=predictions.index)
            blocks.append(pd.concat([predictions, probability], axis=1))
            runtime = {
                "fold": int(fold), "training_rows": len(train), "validation_rows": len(valid),
                "feature_columns": len(names), "feature_seconds": features_seconds, "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds, "warnings": warning_names,
                "sparse_nonzero_values": int(fit.nnz) if sp.issparse(fit) else None,
                "iterations": np.asarray(clf.n_iter_).tolist() if hasattr(clf, "n_iter_") else None,
            }
            if hasattr(clf, "booster_"):
                runtime["boosting_rounds"] = clf.booster_.current_iteration()
            if spec["classifier"] == "xgboost":
                runtime["boosting_rounds"] = clf.get_booster().num_boosted_rounds()
            details.append(runtime)
            weights.extend(top_coefficients(clf, names, labels, int(fold)))
            ba = metrics_for(predictions, labels)["balanced_accuracy"]
            logger.info("第 %d 折  BA %.6f  特征 %d 列  构建 %.2fs  拟合 %.2fs  预测 %.2fs",
                        fold, ba, len(names), features_seconds, fit_seconds, predict_seconds)
    oof = pd.concat(blocks, ignore_index=True).sort_values("id").reset_index(drop=True)
    return oof, details, weights


def json_write(path, value):
    staged = path.with_suffix(path.suffix + ".tmp")
    staged.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    staged.replace(path)


def iso_time():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def review_archive(directory, oof, data, spec, metrics, tables, pairs, metadata):
    counts = tables["confusion_counts.csv"].set_index("true_model")
    confusions = sorted(
        ((int(counts.at[truth, pred]), truth, pred) for truth in counts.index for pred in counts.columns
         if pred != truth and counts.at[truth, pred]),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    rows = [
        f"# {directory.name}  {spec['title']}", "", "## 实验问题", "", spec["question"], "",
        "## 实测结果", "",
        f"整体 BA **{metrics['balanced_accuracy']:.2%}**。Macro-F1 **{metrics['macro_f1']:.2%}**。"
        f"耗时 {metadata['compute_seconds']:.2f} 秒，包含特征构建、拟合与预测。",
        "",
        "[各折](by_fold.csv) · [各类别](by_model.csv) · [逐篇预测](oof.parquet) · [实际配置](run.json)",
        "", "## 常见误判", "",
    ]
    examples = []
    for count, truth, pred in confusions[:4]:
        support = int(counts.loc[truth].sum())
        article = oof[oof.y_true.eq(truth) & oof.y_pred.eq(pred)].iloc[0]
        rows.append(f"- {truth} → {pred} 有 {count} 篇，占该真实类 {count / support:.2%}。示例 ID {article.id}。")
        examples.append(article[ROW_COLUMNS + ["y_pred"]].to_dict() | {"case": "common_confusion"})
    rows.extend(["", "[混淆矩阵热图](figures/confusion.png) · [原始计数](confusion_counts.csv)", ""])
    if pairs is not None:
        outcomes = pairs["outcome"].value_counts()
        rows.extend([
            "## 同篇对照", "",
            f"对照 {metadata['baseline_run']}。{int(outcomes.get('corrected', 0))} 篇改对，"
            f"{int(outcomes.get('regressed', 0))} 篇改错。可按题号追踪变化。",
            "", "[逐篇配对](paired_changes.csv)", "",
        ])
        for outcome in ("corrected", "regressed"):
            for article in pairs[pairs.outcome.eq(outcome)].head(2).to_dict("records"):
                examples.append(article | {"case": outcome})
    if examples:
        sample = pd.DataFrame(examples)
        text_by_id = data.set_index("id")["text"]
        sample["text"] = sample.id.map(text_by_id)
        sample.to_csv(directory / "cases.csv", index=False, encoding="utf-8")
        rows.extend(["[原文案例](cases.csv)", ""])
    if (directory / "top_features.csv").exists():
        rows.extend(["[各折的高权重特征](top_features.csv)", ""])
    (directory / "review.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def render_confusion(directory, plot_python, logger):
    # Isolate the plotting interpreter from the CPU venv's Windows PYTHONHOME.
    plot = subprocess.run(
        [str(plot_python), "-I", "-X", "utf8", "-B", str(ROOT / "src" / "plot_confusion.py"), str(directory)],
        capture_output=True, encoding="utf-8", errors="replace",
    )
    if plot.stdout:
        logger.info("%s", plot.stdout.rstrip())
    if plot.stderr:
        logger.warning("%s", plot.stderr.rstrip())
    plot.check_returncode()


def run_experiment(run_id, config):
    settings, assets = load_settings()
    train_path = assets["official_train"]
    data, train_hash = read_training(train_path)
    manifest, folds = read_protocol(protocol_path(ROOT, settings["id"]), data, train_hash)
    spec = config["runs"][run_id]
    vectorizer, params = resolved_parameters(config, spec)
    threads = config["cpu"]["threads"]
    model = spec["classifier"]
    if model in {"xgboost", "lightgbm"} and TREE_SITE.is_dir():
        sys.path.insert(0, str(TREE_SITE))
    versions = layer_versions(model)
    labels = manifest["label_order"]

    directory = ROOT / "runs" / run_id
    directory.mkdir(exist_ok=False)
    logger = logging.getLogger(run_id)
    logger.setLevel(logging.INFO)
    handlers = [logging.FileHandler(directory / "run.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)]
    for handler in handlers:
        logger.addHandler(handler)
    logger.propagate = False

    source_files = [Path(__file__).resolve(), ROOT / "src" / "prepare_data.py", ROOT / "src" / "check_run.py",
                    ROOT / "src" / "plot_confusion.py", CONFIG]
    command = [sys.executable, *sys.orig_argv[1:]]
    info = {
        "id": run_id, "kind": "grouped_oof", "status": "running", "baseline_run": spec.get("baseline"),
        "protocol_id": manifest["protocol_id"], "train_sha256": train_hash, "folds_sha256": manifest["folds_sha256"],
        "train_file": str(train_path), "started_at_utc": iso_time(),
        "command": command, "working_directory": str(Path.cwd()), "question": spec["question"],
        "config": {"experiment": spec, "vectorizer": vectorizer, "estimator": params, "cpu_threads": threads,
                   "style_columns": STYLE_COLUMNS if spec["features"] in {"style", "char_style"} else []},
        "label_order": labels, "decision_rule": "probability argmax, first label in frozen order if tied",
        "environment": {"python": sys.version.split()[0], "executable": sys.executable, "packages": versions,
                        "plot_python": str(assets["gpu_python"]), "device": "cpu",
                        "processor": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor())},
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
    }

    start = perf_counter()
    try:
        json_write(directory / "run.json", info)
        logger.info("%s  %s  %s", run_id, spec["title"], settings["id"])
        logger.info("命令 %s", json.dumps(command, ensure_ascii=False))
        oof, fold_runtime, weights = fit_grouped_folds(data, folds, labels, spec, vectorizer, params, threads, logger)
        info["compute_seconds"] = sum(
            part[name] for part in fold_runtime for name in ("feature_seconds", "fit_seconds", "predict_seconds")
        )
        info["fold_runtime"] = fold_runtime
        pd.DataFrame(fold_runtime).to_csv(directory / "fold_runtime.csv", index=False, encoding="utf-8")
        oof.to_parquet(directory / "oof.parquet", index=False)
        if weights:
            pd.DataFrame(weights).to_csv(directory / "top_features.csv", index=False, encoding="utf-8")
        metrics = metrics_for(oof, labels)
        if model == "dummy":
            verify_dummy_folds(oof, labels)
        json_write(directory / "metrics.json", metrics)
        tables = diagnostic_tables(oof, labels)
        for name, table in tables.items():
            table.to_csv(directory / name, index=False, encoding="utf-8")

        pairs = None
        if spec.get("baseline"):
            baseline_dir = ROOT / "runs" / spec["baseline"]
            baseline = read_oof(baseline_dir / "oof.parquet", oof[ROW_COLUMNS], labels)
            pairs = paired_changes(oof[ROW_COLUMNS + ["y_pred"]], baseline)
            pairs.to_csv(directory / "paired_changes.csv", index=False, encoding="utf-8")

        render_confusion(directory, assets["gpu_python"], logger)
        review_archive(directory, oof, data, spec, metrics, tables, pairs, info)
        verified = check_scored_run(directory, info, manifest, data, folds)
        info.update(status="completed", exit_code=0, verified_oof_rows=verified,
                    ended_at_utc=iso_time(), elapsed_seconds=perf_counter() - start)
        json_write(directory / "run.json", info)
        logger.info("%s 已通过归档核验  BA %.6f  Macro-F1 %.6f  %d 条折外预测",
                    run_id, metrics["balanced_accuracy"], metrics["macro_f1"], verified)
    except BaseException as error:
        logger.exception("实验中止")
        info.update(status="failed", exit_code=1, error=str(error), ended_at_utc=iso_time(),
                    elapsed_seconds=perf_counter() - start)
        json_write(directory / "run.json", info)
        (directory / "review.md").write_text(
            f"# {run_id}\n\n实验中止。{type(error).__name__} {error}\n\n[运行日志](run.log)\n", encoding="utf-8"
        )
        raise
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()


def main():
    with CONFIG.open("rb") as stream:
        config = tomllib.load(stream)
    parser = argparse.ArgumentParser(description="运行并归档一组冻结的传统基线。")
    parser.add_argument("experiment", choices=config["runs"])
    args = parser.parse_args()
    run_experiment(args.experiment, config)


if __name__ == "__main__":
    main()
