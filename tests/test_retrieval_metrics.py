"""Vérifier les conventions des métriques avec des classements connus."""

import unittest

from contextiq.evaluation.retrieval_metrics import compute_retrieval_metrics


class TestRetrievalMetrics(unittest.TestCase):
    def test_multiple_sources_and_first_relevant_rank(self):
        scores = compute_retrieval_metrics(["X", "A", "B"], ["A", "B"], 3)
        self.assertAlmostEqual(scores.precision, 2 / 3)
        self.assertEqual(scores.recall, 1)
        self.assertEqual(scores.hit_rate, 1)
        self.assertEqual(scores.mrr, 0.5)

    def test_k_truncates_ranking(self):
        scores = compute_retrieval_metrics(["X", "A"], ["A"], 1)
        self.assertTrue(all(value == 0 for value in scores.model_dump().values()))

    def test_precision_denominator_remains_k(self):
        scores = compute_retrieval_metrics(["A"], ["A", "B"], 3)
        self.assertAlmostEqual(scores.precision, 1 / 3)
        self.assertEqual(scores.recall, 0.5)

    def test_no_results(self):
        scores = compute_retrieval_metrics([], ["A"], 3)
        self.assertTrue(all(value == 0 for value in scores.model_dump().values()))

    def test_out_of_scope_and_duplicates_are_rejected(self):
        for retrieved, gold in ((["A"], []), (["A", "A"], ["A"]), (["A"], ["A", "A"])):
            with self.subTest(retrieved=retrieved, gold=gold), self.assertRaises(ValueError):
                compute_retrieval_metrics(retrieved, gold, 3)

    def test_invalid_k(self):
        for k in (0, -1, 1.5, True):
            with self.subTest(k=k), self.assertRaises(ValueError):
                compute_retrieval_metrics(["A"], ["A"], k)
