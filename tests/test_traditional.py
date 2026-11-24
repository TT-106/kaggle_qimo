"""Focused checks for leakage and the traditional-baseline feature contract."""

import logging
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import tomllib
import unittest

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from run_traditional import CONFIG, feature_matrices, render_confusion, resolved_parameters, style_frame, verify_dummy_folds
from char_counts import CharacterCounts
from check_run import metrics_for
from prepare_data import load_settings


class TraditionalTests(unittest.TestCase):
    def test_majority_can_change_across_folds(self):
        outcomes = pd.DataFrame({
            "fold": [1, 1, 1, 2, 2, 2],
            "y_true": ["a", "a", "b", "a", "b", "b"],
            "y_pred": ["b", "b", "b", "a", "a", "a"],
        })
        verify_dummy_folds(outcomes, ["a", "b"])
        self.assertAlmostEqual(metrics_for(outcomes, ["a", "b"])["balanced_accuracy"], 1 / 3)
        outcomes.loc[0, "y_pred"] = "a"
        with self.assertRaisesRegex(ValueError, "第 1 折"):
            verify_dummy_folds(outcomes, ["a", "b"])

    def test_word_vocabulary_uses_training_text_only(self):
        with CONFIG.open("rb") as stream:
            config = tomllib.load(stream)
        spec = config["runs"]["E003-word-tfidf-lr"]
        options, _ = resolved_parameters(config, spec)
        train, validation, names = feature_matrices(["A fox", "A fox."], ["holdoutonly"], spec, options)
        self.assertEqual(set(names), {"A", "fox", "A fox"})
        self.assertEqual(train.shape, (2, 3))
        self.assertEqual(validation.nnz, 0)

    def test_twenty_style_features_preserve_unicode_punctuation(self):
        text = "A—b? It is “2”."
        features = style_frame([text])
        self.assertEqual(features.shape, (1, 20))
        self.assertEqual(features.at[0, "sentences"], 2)
        self.assertAlmostEqual(features.at[0, "em_dash_per_100_chars"], 100 / len(text))
        self.assertAlmostEqual(features.at[0, "curly_quotes_per_100_chars"], 200 / len(text))

    def test_shared_raw_counts_match_a_fold_only_vectorizer(self):
        text = ["aB ba", "aB Ca", "BC aB", "Qtest9 aB"]
        ids = ["t1", "t2", "t3", "v1"]
        options = {
            "analyzer": "char", "ngram_range": (1, 3), "min_df": 2, "max_features": 4,
            "lowercase": False, "use_idf": True, "smooth_idf": True, "sublinear_tf": True, "norm": "l2",
        }
        direct = TfidfVectorizer(**options)
        expected_train = direct.fit_transform(text[:3])
        expected_valid = direct.transform(text[3:])
        shared = CharacterCounts(text, ids, options)
        fit, hold, names = shared.fit_fold(ids[:3], ids[3:])
        self.assertEqual(names, direct.get_feature_names_out().tolist())
        np.testing.assert_allclose(fit.toarray(), expected_train.toarray(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(hold.toarray(), expected_valid.toarray(), rtol=1e-12, atol=1e-12)

    def test_tree_char_columns_keep_the_original_style_values(self):
        train = ["aaaa!", "aaaa.", "bbbb?", "bbbb."]
        valid = ["zzzz?"]
        train_style, valid_style = style_frame(train), style_frame(valid)
        spec = {"features": "char_style", "classifier": "lightgbm", "chi2_char_features": 2}
        options = {"analyzer": "char", "ngram_range": (1, 1)}
        fit, hold, names = feature_matrices(
            train, valid, spec, options, (train_style, valid_style), [0, 0, 1, 1]
        )
        self.assertEqual(len(names), 22)
        self.assertNotIn("z", names)
        self.assertEqual(fit[:, -20:].toarray().tolist(), train_style.values.tolist())
        self.assertEqual(hold[:, -20:].toarray().tolist(), valid_style.values.tolist())

    def test_plotting_uses_the_second_venv_in_isolation(self):
        _, assets = load_settings()
        with TemporaryDirectory() as scratch:
            directory = Path(scratch)
            counts = pd.DataFrame({"true_model": ["a", "b"], "a": [3, 1], "b": [1, 3]})
            shares = pd.DataFrame({"true_model": ["a", "b"], "a": [.75, .25], "b": [.25, .75]})
            counts.to_csv(directory / "confusion_counts.csv", index=False)
            shares.to_csv(directory / "confusion_by_true.csv", index=False)
            render_confusion(directory, assets["gpu_python"], logging.getLogger(__name__))
            png = directory / "figures" / "confusion.png"
            self.assertGreater(png.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
