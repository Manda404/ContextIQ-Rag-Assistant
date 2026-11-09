"""Vérifier les dénominateurs et conserver les groupes absents comme None."""

import unittest

from contextiq.evaluation.abstention_metrics import AbstentionEvaluator
from contextiq.safety.abstention import AbstentionPolicy
from contextiq.schemas.models import Document, EvaluationQuestion, RetrievalResult


class FixedScoreRetriever:
    """Fournir des scores connus sans exécuter TF-IDF ou BM25."""

    def retrieve(self, question: str, top_k: int = 3) -> list[RetrievalResult]:
        scores = {
            "pertinente acceptée": 0.8,
            "pertinente refusée": 0.1,
            "hors périmètre accepté": 0.7,
            "hors périmètre refusé": 0.0,
        }
        score = scores[question]
        if not score:
            return []
        return [RetrievalResult(document=Document(id="A", text="Texte fictif"), score=score)]


class TestAbstentionEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = AbstentionEvaluator()
        self.policy = AbstentionPolicy(threshold=0.3)
        self.retriever = FixedScoreRetriever()

    def test_error_rates_and_acceptance_rate(self):
        questions = [
            EvaluationQuestion(q="pertinente acceptée", gold_docs=["A"]),
            EvaluationQuestion(q="pertinente refusée", gold_docs=["A"]),
            EvaluationQuestion(q="hors périmètre accepté", gold_docs=[]),
            EvaluationQuestion(q="hors périmètre refusé", gold_docs=[]),
        ]
        report = self.evaluator.evaluate(self.retriever, questions, self.policy)
        self.assertEqual(report.in_scope_count, 2)
        self.assertEqual(report.out_of_scope_count, 2)
        self.assertEqual(report.false_refusals, 1)
        self.assertEqual(report.false_acceptances, 1)
        self.assertEqual(report.false_refusal_rate, 0.5)
        self.assertEqual(report.false_acceptance_rate, 0.5)
        self.assertEqual(report.acceptance_rate, 0.5)
        self.assertEqual(len(report.cases), 4)

    def test_missing_group_has_no_rate(self):
        for question, missing_field in (
            (EvaluationQuestion(q="pertinente acceptée", gold_docs=["A"]), "false_acceptance_rate"),
            (EvaluationQuestion(q="hors périmètre refusé", gold_docs=[]), "false_refusal_rate"),
        ):
            with self.subTest(missing_field=missing_field):
                report = self.evaluator.evaluate(self.retriever, [question], self.policy)
                self.assertIsNone(getattr(report, missing_field))

    def test_acceptance_does_not_mean_correct_source(self):
        question = EvaluationQuestion(q="pertinente acceptée", gold_docs=["B"])
        report = self.evaluator.evaluate(self.retriever, [question], self.policy)
        self.assertEqual(report.acceptance_rate, 1)
        self.assertEqual(report.false_refusal_rate, 0)
        self.assertNotIn("B", [r.document.id for r in report.cases[0].decision.results])

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            self.evaluator.evaluate(self.retriever, [], self.policy)
        for k in (0, -1, True, 1.5):
            with self.subTest(k=k), self.assertRaises(ValueError):
                self.evaluator.evaluate(
                    self.retriever,
                    [EvaluationQuestion(q="pertinente acceptée", gold_docs=["A"])],
                    self.policy,
                    top_k=k,
                )
