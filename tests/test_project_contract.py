"""Small fixtures for the grouped evaluation and experiment archive contracts."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from check_run import check_run, check_scored_run, diagnostic_tables, metrics_for, paired_changes
from prepare_data import check_folds, freeze_protocol, make_folds, read_protocol, read_training


def demo_data():
    records = [
        {"id": f"{prompt}-{label}", "prompt_id": f"p{prompt}", "text": "test fixture", "model": label}
        for prompt in range(4)
        for label in ("a", "b")
    ]
    return pd.DataFrame(records)


class ProtocolTests(unittest.TestCase):
    def test_prompt_assigned_to_two_folds_is_rejected(self):
        data = demo_data()
        folds = make_folds(data, n_splits=2)
        self.assertEqual([part["shared_prompts"] for part in check_folds(data, folds, 2)], [0, 0])

        folds.loc[0, "fold"] = 3 - folds.loc[0, "fold"]
        with self.assertRaisesRegex(ValueError, "Prompt 被分到多折"):
            check_folds(data, folds, 2)

    def test_frozen_fold_file_change_is_detected(self):
        with TemporaryDirectory() as scratch:
            root = Path(scratch)
            source = root / "train.csv"
            demo_data().to_csv(source, index=False)
            data, fingerprint = read_training(source)
            directory = root / "test-protocol"
            freeze_protocol(directory, source, data, fingerprint, n_splits=2)
            manifest, folds = read_protocol(directory, data, fingerprint)
            self.assertEqual(len(folds), manifest["rows"])

            fold_file = directory / "folds.csv"
            fold_file.write_bytes(fold_file.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "分折指纹不符"):
                read_protocol(directory, data, fingerprint)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        scratch = TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.data = demo_data()
        self.folds = make_folds(self.data, n_splits=2)
        self.manifest = {
            "protocol_id": "test-protocol",
            "train_sha256": "test-data",
            "folds_sha256": "test-folds",
            "label_order": ["a", "b"],
        }
        self.oof = self.folds.merge(
            self.data[["id", "model"]], on="id", validate="one_to_one"
        ).rename(columns={"model": "y_true"}).sort_values("id").reset_index(drop=True)

    def create_archive(self, run_id, predictions, baseline_run=None):
        directory = self.root / run_id
        (directory / "figures").mkdir(parents=True)
        oof = self.oof.assign(y_pred=predictions)
        oof.to_parquet(directory / "oof.parquet", index=False)
        metadata = {
            "id": run_id,
            "kind": "grouped_oof",
            "status": "completed",
            "exit_code": 0,
            "baseline_run": baseline_run,
            **{key: value for key, value in self.manifest.items() if key != "label_order"},
        }
        (directory / "run.json").write_text(json.dumps(metadata), encoding="utf-8")
        (directory / "metrics.json").write_text(
            json.dumps(metrics_for(oof, self.manifest["label_order"])), encoding="utf-8"
        )
        for name, table in diagnostic_tables(oof, self.manifest["label_order"]).items():
            table.to_csv(directory / name, index=False)
        (directory / "figures" / "confusion.png").write_bytes(b"figure presence fixture")
        return directory, metadata, oof

    def test_count_matrix_is_recomputed(self):
        run, metadata, _ = self.create_archive("E000-a", ["a"] * len(self.oof))
        self.assertEqual(check_scored_run(run, metadata, self.manifest, self.data, self.folds), len(self.oof))
        counts = pd.read_csv(run / "confusion_counts.csv")
        self.assertEqual(counts["a"].tolist(), [4, 4])
        counts.loc[0, "a"] = 0
        counts.to_csv(run / "confusion_counts.csv", index=False)
        with self.assertRaisesRegex(ValueError, "confusion_counts.csv 与逐篇"):
            check_scored_run(run, metadata, self.manifest, self.data, self.folds)

    def test_same_id_comparison_is_checked(self):
        _, _, baseline = self.create_archive("E000-a", ["a"] * len(self.oof))
        predictions = ["a", "b", "b", "a", "a", "b", "a", "b"]
        run, metadata, oof = self.create_archive("E001-b", predictions, baseline_run="E000-a")
        pairs = paired_changes(oof, baseline)
        self.assertEqual(pairs["outcome"].value_counts().to_dict(), {
            "both_correct": 3, "corrected": 3, "both_wrong": 1, "regressed": 1,
        })
        pairs.to_csv(run / "paired_changes.csv", index=False)
        self.assertEqual(check_scored_run(run, metadata, self.manifest, self.data, self.folds), len(self.oof))

        pairs.loc[0, "outcome"] = "regressed"
        pairs.to_csv(run / "paired_changes.csv", index=False)
        with self.assertRaisesRegex(ValueError, "paired_changes.csv 与逐篇"):
            check_scored_run(run, metadata, self.manifest, self.data, self.folds)

    def test_failed_run_keeps_log_and_review(self):
        directory = self.root / "E000-failed"
        directory.mkdir()
        (directory / "run.json").write_text(
            '{"id":"E000-failed","kind":"grouped_oof","status":"failed"}', encoding="utf-8"
        )
        (directory / "run.log").write_text("fixture error", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "review.md"):
            check_run(directory)
        (directory / "review.md").write_text("fixture review", encoding="utf-8")
        self.assertEqual(check_run(directory), "日志与回顾已归档")


if __name__ == "__main__":
    unittest.main()
