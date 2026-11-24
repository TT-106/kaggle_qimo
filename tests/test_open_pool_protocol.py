"""Check that overlapping Open Pool prompt identifiers stay together."""

import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prepare_open_pool import check_original_prompt_groups, connected_prompts


class OpenPoolTests(unittest.TestCase):
    def setUp(self):
        self.rows = pd.DataFrame({
            "id": ["1", "2", "3", "4", "5"],
            "prompt_group": ["a", "a", "b", "b", "c"],
            "prompt_canon": ["p1", "p2", "p2", "p3", "p4"],
        })

    def test_prompt_links_are_transitive(self):
        self.assertEqual(connected_prompts(self.rows).tolist(), ["p1", "p1", "p1", "p1", "p4"])

    def test_split_original_or_canonical_id_is_rejected(self):
        folds = self.rows[["id"]].assign(fold=[1, 1, 1, 1, 2])
        self.assertEqual(check_original_prompt_groups(self.rows, folds), {"prompt_group": 0, "prompt_canon": 0})
        folds.loc[2, "fold"] = 2
        with self.assertRaisesRegex(ValueError, "跨折"):
            check_original_prompt_groups(self.rows, folds)


if __name__ == "__main__":
    unittest.main()
