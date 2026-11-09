"""Vérifier les décisions aux limites, sans données confidentielles."""

import unittest

from pydantic import ValidationError

from contextiq.safety.abstention import AbstentionPolicy
from contextiq.schemas.models import Document, RetrievalResult


def result(score: float, identifier: str = "A") -> RetrievalResult:
    """Créer une source fictive associée à un score connu."""
    return RetrievalResult(document=Document(id=identifier, text="Texte fictif"), score=score)


class TestAbstentionPolicy(unittest.TestCase):
    def test_no_results(self):
        decision = AbstentionPolicy(threshold=0.3).decide([])
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "no_results")
        self.assertIsNone(decision.best_score)

    def test_below_threshold_preserves_sources(self):
        results = [result(0.2)]
        decision = AbstentionPolicy(threshold=0.3).decide(results)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "below_threshold")
        self.assertEqual(decision.results, results)
        self.assertEqual(decision.best_score, 0.2)
        self.assertEqual(decision.threshold, 0.3)

    def test_equal_and_above_threshold(self):
        for score in (0.3, 0.4):
            with self.subTest(score=score):
                decision = AbstentionPolicy(threshold=0.3).decide([result(score)])
                self.assertTrue(decision.accepted)
                self.assertEqual(decision.reason, "threshold_met")

    def test_zero_score_refused_even_with_zero_threshold(self):
        decision = AbstentionPolicy(threshold=0).decide([result(0.0)])
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "no_positive_score")

    def test_best_score_does_not_depend_on_order(self):
        decision = AbstentionPolicy(threshold=0.3).decide([result(0.1), result(0.5, "B")])
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.best_score, 0.5)

    def test_bm25_scores_can_exceed_one(self):
        self.assertTrue(AbstentionPolicy(threshold=2).decide([result(3.0)]).accepted)

    def test_threshold_is_required_and_finite(self):
        with self.assertRaises(ValidationError):
            AbstentionPolicy()
        for threshold in (-1, float("nan"), float("inf")):
            with self.subTest(threshold=threshold), self.assertRaises(ValidationError):
                AbstentionPolicy(threshold=threshold)
