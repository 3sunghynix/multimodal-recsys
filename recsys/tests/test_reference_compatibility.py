"""Cross-check production metrics against independent readable formulas.

Use the same valid cohort, binary targets, catalogue and fixed normalized
vectors. Boundary-policy differences are tested explicitly, not hidden.
"""
import unittest

import numpy as np
import pandas as pd

from recsys.eval import metrics as fast
from recsys.eval import reference
from recsys.eval.online import dominant_brand_share


class ReferenceCompatibilityTests(unittest.TestCase):
    def test_accuracy_and_per_customer_values_match_for_multiple_k(self):
        rng = np.random.default_rng(42)
        users = np.array([107, 101, 103, 109, 102])  # Intentionally unsorted.
        catalog = np.arange(40)
        rec = np.stack([rng.permutation(catalog)[:20] for _ in users])
        targets = {int(u): set(rng.choice(catalog, size=n, replace=False).tolist())
                   for u, n in zip(users, [1, 3, 12, 20, 30])}
        truth = pd.DataFrame([(u, item) for u in users for item in targets[int(u)]], columns=['user_id', 'item_id'])
        rankings = {int(u): row.tolist() for u, row in zip(users, rec)}
        for k in (1, 3, 12, 20):
            actual, per = fast.accuracy_at_k(rec, users, truth, (k,))
            expected = reference.ranking_metrics(rankings, targets, eligible_users=users.tolist(), k=k)
            for name, column in [('recall', 'recall'), ('ndcg', 'ndcg'), ('map', 'ap')]:
                with self.subTest(k=k, metric=name):
                    self.assertAlmostEqual(actual[f'{name}@{k}'], expected[name], places=12)
                    np.testing.assert_allclose(per[f'{name}@{k}'], [u[column] for u in expected['per_user']], atol=1e-12, rtol=0)

    def test_coverage_matches_fixed_catalogue_denominator(self):
        catalog = [10, 20, 30, 40, 50, 60]
        rec = np.array([[10, 30, 20], [20, 10, 40]])
        for k in (1, 2, 3):
            actual = fast.coverage_at_k(rec, k, len(catalog))
            expected = reference.catalog_coverage({i: row.tolist() for i, row in enumerate(rec)}, catalog, k=k)
            self.assertAlmostEqual(actual, expected['value'], places=12)

    def test_diversity_matches_for_same_normalized_content_vectors(self):
        vectors = np.array([[1., 0.], [0., 1.], [-1., 0.], [1., 1.], [1., -1.]])
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        rec = np.array([[0, 1, 2], [3, 0, 4]])
        for k in (2, 3):
            actual = fast.intra_list_diversity_at_k(rec, k, vectors)
            expected = reference.intralist_diversity({i: row.tolist() for i, row in enumerate(rec)},
                                                     {i: v.tolist() for i, v in enumerate(vectors)}, k=k)
            self.assertAlmostEqual(actual, expected['value'], places=12)

    def test_online_mean_matches_offline_per_customer_loyalty_on_known_brands(self):
        brand_of = np.array([0, 0, 1, 2])
        history = pd.DataFrame({'user_id': [7, 7, 7, 9, 9, 9, 9], 'item_id': [0, 1, 2, 3, 3, 3, 2]})
        offline = fast.user_brand_loyalty(history, brand_of)
        events = {int(u): brand_of[group.item_id.to_numpy()].tolist() for u, group in history.groupby('user_id')}
        online = dominant_brand_share(events, event_type='purchase')
        for result in online['per_user']:
            self.assertAlmostEqual(result['value'], offline.loc[result['user_id']], places=12)
        self.assertAlmostEqual(online['value'], float(offline.mean()), places=12)

    def test_zero_vector_policy_difference_is_explicit(self):
        rec = np.array([[0, 1]])
        vectors = np.array([[0., 0.], [1., 0.]])
        self.assertEqual(fast.intra_list_diversity_at_k(rec, 2, vectors), 1.)
        with self.assertRaises(ValueError):
            reference.intralist_diversity({1: [0, 1]}, {i: row.tolist() for i, row in enumerate(vectors)}, k=2)


if __name__ == '__main__':
    unittest.main()
