"""Vérifier l'agrégation indépendamment des algorithmes de recherche."""

import unittest

from contextiq.evaluation.evaluator import RetrievalEvaluator
from contextiq.schemas.models import Document, EvaluationQuestion, RetrievalResult


class FixedRetriever:
    """Un classement connu pour isoler les calculs de l'évaluateur."""

    def retrieve(self, question: str, top_k: int = 3) -> list[RetrievalResult]:
        return [
            RetrievalResult(document=Document(id=identifier, text="Texte fictif"), score=1.0)
            for identifier in ["A", "B"][:top_k]
        ]


class TestRetrievalEvaluator(unittest.TestCase):
    def test_macro_average_excludes_out_of_scope(self):
        questions = [
            EvaluationQuestion(q="Question A", gold_docs=["A"]),
            EvaluationQuestion(q="Question B", gold_docs=["B"]),
            EvaluationQuestion(q="Hors périmètre", gold_docs=[]),
        ]
        report = RetrievalEvaluator().evaluate(FixedRetriever(), questions, top_k=2)
        self.assertEqual(report.in_scope_count, 2)
        self.assertEqual(report.out_of_scope_count, 1)
        self.assertEqual(report.summary.precision, 0.5)
        self.assertEqual(report.summary.recall, 1)
        self.assertEqual(report.summary.mrr, 0.75)
        self.assertIsNone(report.questions[-1].metrics)
        self.assertEqual(len(report.questions[-1].results), 2)

    def test_k_is_forwarded_to_retriever(self):
        report = RetrievalEvaluator().evaluate(
            FixedRetriever(),
            [EvaluationQuestion(q="Question", gold_docs=["B"])],
            top_k=1,
        )
        self.assertEqual(len(report.questions[0].results), 1)
        self.assertEqual(report.summary.recall, 0)

    def test_only_out_of_scope_has_no_summary(self):
        report = RetrievalEvaluator().evaluate(
            FixedRetriever(),
            [EvaluationQuestion(q="Hors périmètre", gold_docs=[])],
        )
        self.assertIsNone(report.summary)
        self.assertEqual(report.in_scope_count, 0)
        self.assertEqual(report.out_of_scope_count, 1)

    def test_empty_dataset_and_invalid_k(self):
        evaluator = RetrievalEvaluator()
        with self.assertRaises(ValueError):
            evaluator.evaluate(FixedRetriever(), [])
        with self.assertRaises(ValueError):
            evaluator.evaluate(
                FixedRetriever(), [EvaluationQuestion(q="Question", gold_docs=[])], top_k=0
            )
