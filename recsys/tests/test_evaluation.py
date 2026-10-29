"""Hand-calculated counterexamples and edge cases; no model/database needed."""

import json
from math import log2
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import sys
import unittest

from recsys.evaluation import (
    catalog_coverage, dominant_brand_share, evaluate_report,
    intralist_diversity, ranking_metrics,
)


class RankingTests(unittest.TestCase):
    def test_rank_order_changes_map_and_ndcg_not_recall(self):
        # Two hits among top 3: precisions 1 and 2/3; AP = 5/6.
        result = ranking_metrics({"u": ["a", "x", "b"]}, {"u": ["a", "b"]}, eligible_users=["u"], k=3)
        self.assertEqual(result["recall"], 1)
        self.assertAlmostEqual(result["map"], 5/6)
        self.assertAlmostEqual(result["ndcg"], 1.5 / (1 + 1/log2(3)))
        perfect = ranking_metrics({"u": ["a", "b", "x"]}, {"u": ["a", "b"]}, eligible_users=["u"], k=3)
        self.assertEqual(perfect["recall"], result["recall"])
        self.assertEqual(perfect["map"], 1)
        self.assertEqual(perfect["ndcg"], 1)

    def test_map12_uses_min_target_count_and_k(self):
        # 20 targets, 12 correct recommendations: AP@12=1, Recall@12=0.6.
        result = ranking_metrics({"u": list(range(12))}, {"u": list(range(20))}, eligible_users=["u"], k=12)
        self.assertEqual(result["map"], 1)
        self.assertEqual(result["recall"], .6)

    def test_short_list_does_not_shrink_ap_denominator(self):
        result = ranking_metrics({"u": [1]}, {"u": [1,2,3]}, eligible_users=["u"], k=12)
        self.assertAlmostEqual(result["map"], 1/3)

    def test_macro_mean_is_not_purchase_weighted(self):
        result = ranking_metrics({"a": [1], "b": []}, {"a": [1], "b": [2,3,4]}, eligible_users=["a","b"], k=12)
        self.assertEqual(result["recall"], .5)  # Purchase-weighted would be 1/4.

    def test_targets_deduplicated_but_recommendations_rejected(self):
        result = ranking_metrics({"u": [1]}, {"u": [1,1]}, eligible_users=["u"])
        self.assertEqual(result["recall"], 1)
        with self.assertRaises(ValueError):
            ranking_metrics({"u": [1,1]}, {"u": [1]}, eligible_users=["u"])

    def test_explicit_cohort_and_missing_recommendations(self):
        result = ranking_metrics({"extra": [1]}, {"u": [1], "extra": [1]}, eligible_users=["u", "no_target"])
        self.assertEqual(result["recall"], 0)
        self.assertEqual(result["n_evaluated_users"], 1)
        self.assertEqual(result["n_excluded_no_targets"], 1)
        self.assertEqual(result["n_evaluated_without_recommendations"], 1)

    def test_empty_cohort_has_undefined_accuracy(self):
        result = ranking_metrics({}, {}, eligible_users=[])
        self.assertIsNone(result["recall"])
        self.assertIsNone(result["ndcg"])
        self.assertIsNone(result["map"])

    def test_invalid_k_and_duplicate_users(self):
        for k in [0, -1, True, 1.5]:
            with self.subTest(k=k), self.assertRaises(ValueError):
                ranking_metrics({}, {}, eligible_users=[], k=k)
        with self.assertRaises(ValueError):
            ranking_metrics({}, {}, eligible_users=["u","u"])


class CatalogTests(unittest.TestCase):
    def test_unique_catalog_coverage_and_k(self):
        result = catalog_coverage({"u": ["a","b"], "v": ["a","c"]}, ["a","b","c","d"], k=1)
        self.assertEqual(result["value"], .25)
        self.assertEqual(result["n_recommended_items"], 1)

    def test_empty_catalog_and_unknown_item(self):
        self.assertIsNone(catalog_coverage({}, [])["value"])
        self.assertEqual(catalog_coverage({}, [1])["value"], 0)
        with self.assertRaises(ValueError):
            catalog_coverage({"u": [2]}, [1])

    def test_source_qualified_ids_do_not_collide(self):
        a, b = ("amazon", 1), ("hm", 1)
        self.assertEqual(catalog_coverage({"u": [a,b]}, [a,b])["value"], 1)


class DiversityTests(unittest.TestCase):
    def test_parallel_orthogonal_and_opposite_vectors(self):
        vectors = {"a": [2,0], "same": [1,0], "orth": [0,3], "opp": [-4,0]}
        for second, expected in [("same",0),("orth",1),("opp",2)]:
            with self.subTest(second=second):
                self.assertAlmostEqual(intralist_diversity({"u": ["a",second]}, vectors)["value"], expected)

    def test_user_mean_not_pair_weighted(self):
        # First list: 3 distances [1,2,1], mean 4/3; second: distance 0.
        result = intralist_diversity({"u": [1,2,3], "v": [1,4]}, {1:[1,0],2:[0,1],3:[-1,0],4:[1,0]})
        self.assertAlmostEqual(result["value"], 2/3)

    def test_short_lists_undefined_not_zero(self):
        result = intralist_diversity({"u": [1], "v": []}, {1:[1,0]})
        self.assertIsNone(result["value"])
        self.assertEqual(result["n_excluded_short_lists"], 2)

    def test_bad_vectors_fail_loudly(self):
        cases = [{1:[1,0]}, {1:[0,0],2:[1,0]}, {1:[1,0],2:[1]}, {1:[float("nan"),0],2:[1,0]}]
        for vectors in cases:
            with self.subTest(vectors=vectors), self.assertRaises(ValueError):
                intralist_diversity({"u": [1,2]}, vectors)


class BrandTests(unittest.TestCase):
    def test_customer_mean_not_event_weighted(self):
        result = dominant_brand_share({"u":["A","A","B"], "v":["B"]*10}, event_type="purchase")
        self.assertAlmostEqual(result["value"], 5/6)
        self.assertEqual(result["n_users"], 2)

    def test_unknown_brands_and_no_events(self):
        result = dominant_brand_share({"u":["A",None], "v":[None], "w":[]}, event_type="click")
        self.assertEqual(result["value"], 1)
        self.assertEqual(result["n_unknown_brand_events"], 2)
        self.assertEqual(result["n_excluded_no_known_brand_events"], 2)
        self.assertIsNone(dominant_brand_share({}, event_type="view")["value"])

    def test_invalid_event_type(self):
        with self.assertRaises(ValueError):
            dominant_brand_share({}, event_type="all")


class ReportTests(unittest.TestCase):
    def test_hand_calculated_example_and_json(self):
        source = Path(__file__).parents[1] / "examples/evaluation_toy.json"
        report = evaluate_report(**json.loads(source.read_text()))
        self.assertEqual(report["ranking"]["recall"], .5)
        self.assertAlmostEqual(report["ranking"]["map"], 5/12)
        self.assertEqual(report["catalog_coverage"]["value"], 1)
        self.assertAlmostEqual(report["intralist_diversity"]["value"], 2/3)
        self.assertAlmostEqual(report["dominant_brand_share"]["value"], 5/6)
        json.dumps(report, allow_nan=False)

    def test_unknown_targets_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_report(rankings={}, targets={"u":[2]}, eligible_users=["u"], catalog=[1])

    def test_cli_does_not_overwrite_results(self):
        root = Path(__file__).resolve().parents[2]
        source = root / "recsys/examples/evaluation_toy.json"
        with TemporaryDirectory() as folder:
            output = Path(folder) / "result.json"
            command = [sys.executable,"-m","recsys.evaluation",str(source),"--output",str(output)]
            subprocess.run(command,cwd=root,check=True,capture_output=True)
            original = output.read_bytes()
            second = subprocess.run(command,cwd=root,capture_output=True)
            self.assertNotEqual(second.returncode,0)
            self.assertEqual(original,output.read_bytes())

    def test_cli_rejects_json_customer_id_type_mismatch(self):
        root = Path(__file__).resolve().parents[2]
        with TemporaryDirectory() as folder:
            source = Path(folder) / "input.json"
            source.write_text(json.dumps({
                "rankings":{"1":[1]}, "targets":{"1":[1]},
                "eligible_users":[1], "catalog":[1],
            }))
            result = subprocess.run([sys.executable,"-m","recsys.evaluation",str(source)],cwd=root,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("string IDs", result.stderr)


if __name__ == "__main__":
    unittest.main()
