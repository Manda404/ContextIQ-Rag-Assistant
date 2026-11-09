"""Tester indépendamment les sources, les citations, la fidélité et les refus."""

import unittest

from contextiq.evaluation.answer_metrics import ExtractiveAnswerEvaluator
from contextiq.generation.prompt_builder import PromptBuilder
from contextiq.schemas.models import Document, EvaluationQuestion, ExtractiveAnswer


class TestExtractiveAnswerEvaluator(unittest.TestCase):
    def setUp(self):
        self.documents = [
            Document(id="A", text="Préavis de 10 jours, sans pénalité."),
            Document(id="B", text="Les frais sont plafonnés à 150 euros."),
        ]
        self.evaluator = ExtractiveAnswerEvaluator(self.documents)
        self.question = EvaluationQuestion(q="Quel préavis ?", gold_docs=["A"])

    def answer(self, source_id="A", text=None, question=None):
        """Construire une réponse fictive, sans appeler un retriever ou un modèle."""
        source = next(doc for doc in self.documents if doc.id == source_id)
        return ExtractiveAnswer(
            question=question or self.question.q,
            answered=True,
            reason="extractive",
            text=text if text is not None else f"{source.text} [{source.id}]",
            sources=[source],
        )

    def test_correct_excerpt(self):
        case = self.evaluator.evaluate(self.answer(), self.question)
        self.assertTrue(case.reference_hit)
        self.assertTrue(case.verbatim_faithful)
        self.assertTrue(case.citation_valid)
        self.assertTrue(case.successful)

    def test_faithful_excerpt_can_be_irrelevant(self):
        case = self.evaluator.evaluate(self.answer(source_id="B"), self.question)
        self.assertTrue(case.verbatim_faithful)
        self.assertTrue(case.citation_valid)
        self.assertFalse(case.reference_hit)
        self.assertFalse(case.successful)

    def test_missing_and_wrong_citation_do_not_change_body_faithfulness(self):
        for suffix in ("", " [UNKNOWN]"):
            answer = self.answer(text=self.documents[0].text + suffix)
            case = self.evaluator.evaluate(answer, self.question)
            with self.subTest(suffix=suffix):
                self.assertTrue(case.verbatim_faithful)
                self.assertFalse(case.citation_valid)
                self.assertFalse(case.successful)

    def test_known_citation_does_not_make_invented_text_faithful(self):
        case = self.evaluator.evaluate(self.answer(text="Préavis de 50 jours. [A]"), self.question)
        self.assertTrue(case.citation_valid)
        self.assertFalse(case.verbatim_faithful)
        self.assertFalse(case.successful)

    def test_declared_source_is_checked_against_independent_original(self):
        answer = self.answer(text="Texte inventé. [A]")
        answer.sources = [Document(id="A", text="Texte inventé.")]
        case = self.evaluator.evaluate(answer, self.question)
        self.assertFalse(case.verbatim_faithful)
        self.assertFalse(case.successful)

    def test_unknown_or_missing_source_is_invalid(self):
        for sources in ([], [Document(id="UNKNOWN", text="Texte")]):
            answer = self.answer()
            answer.sources = sources
            case = self.evaluator.evaluate(answer, self.question)
            with self.subTest(sources=sources):
                self.assertFalse(case.citation_valid)
                self.assertFalse(case.verbatim_faithful)
                self.assertFalse(case.successful)

    def test_refusal_for_out_of_scope_is_successful(self):
        question = EvaluationQuestion(q="Quelle météo ?", gold_docs=[])
        answer = ExtractiveAnswer(
            question=question.q,
            answered=False,
            reason="abstention",
            text=PromptBuilder.REFUSAL_MESSAGE,
        )
        case = self.evaluator.evaluate(answer, question)
        self.assertTrue(case.refusal_correct)
        self.assertTrue(case.successful)
        self.assertIsNone(case.verbatim_faithful)
        self.assertIsNone(case.citation_valid)

    def test_false_refusal_is_not_successful(self):
        answer = ExtractiveAnswer(
            question=self.question.q,
            answered=False,
            reason="abstention",
            text=PromptBuilder.REFUSAL_MESSAGE,
        )
        case = self.evaluator.evaluate(answer, self.question)
        self.assertFalse(case.refusal_correct)
        self.assertFalse(case.scope_decision_correct)
        self.assertFalse(case.successful)

    def test_answering_out_of_scope_is_not_successful(self):
        question = EvaluationQuestion(q="Quelle météo ?", gold_docs=[])
        case = self.evaluator.evaluate(self.answer(question=question.q), question)
        self.assertTrue(case.verbatim_faithful)
        self.assertFalse(case.reference_hit)
        self.assertFalse(case.successful)

    def test_refusal_with_sources_or_wrong_message_is_invalid(self):
        question = EvaluationQuestion(q="Quelle météo ?", gold_docs=[])
        for sources, text in (
            (self.documents, PromptBuilder.REFUSAL_MESSAGE),
            ([], "Réponse inventée"),
        ):
            answer = ExtractiveAnswer(
                question=question.q,
                answered=False,
                reason="abstention",
                text=text,
                sources=sources,
            )
            with self.subTest(sources=sources, text=text):
                self.assertFalse(self.evaluator.evaluate(answer, question).successful)

    def test_summary_denominators_and_joint_success(self):
        correct = self.evaluator.evaluate(self.answer(), self.question)
        wrong_source = self.evaluator.evaluate(self.answer(source_id="B"), self.question)
        out = EvaluationQuestion(q="Quelle météo ?", gold_docs=[])
        refused = ExtractiveAnswer(
            question=out.q,
            answered=False,
            reason="abstention",
            text=PromptBuilder.REFUSAL_MESSAGE,
        )
        summary = self.evaluator.summarize(
            [correct, wrong_source, self.evaluator.evaluate(refused, out)]
        )
        self.assertEqual(summary.question_count, 3)
        self.assertEqual(summary.answered_count, 2)
        self.assertAlmostEqual(summary.success_rate, 2 / 3)
        self.assertEqual(summary.faithfulness_rate, 1)
        self.assertEqual(summary.citation_validity_rate, 1)
        self.assertEqual(summary.relevant_source_rate, 0.5)
        self.assertEqual(summary.correct_refusal_rate, 1)

    def test_missing_groups_are_not_scored_as_zero(self):
        summary = self.evaluator.summarize([self.evaluator.evaluate(self.answer(), self.question)])
        self.assertIsNone(summary.correct_refusal_rate)
        out = EvaluationQuestion(q="Quelle météo ?", gold_docs=[])
        refused = ExtractiveAnswer(
            question=out.q,
            answered=False,
            reason="abstention",
            text=PromptBuilder.REFUSAL_MESSAGE,
        )
        summary = self.evaluator.summarize([self.evaluator.evaluate(refused, out)])
        self.assertIsNone(summary.faithfulness_rate)
        self.assertIsNone(summary.citation_validity_rate)
        self.assertIsNone(summary.relevant_source_rate)
        self.assertIsNone(summary.false_refusal_rate)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            self.evaluator.evaluate(
                self.answer(), EvaluationQuestion(q="Autre question", gold_docs=["A"])
            )
        with self.assertRaises(ValueError):
            self.evaluator.evaluate(
                self.answer(), EvaluationQuestion(q=self.question.q, gold_docs=["UNKNOWN"])
            )
        with self.assertRaises(ValueError):
            self.evaluator.summarize([])
        with self.assertRaises(ValueError):
            ExtractiveAnswerEvaluator([])
        with self.assertRaises(ValueError):
            ExtractiveAnswerEvaluator(self.documents * 2)
