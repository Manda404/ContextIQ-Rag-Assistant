"""Évaluer les extraits contre la base originale et les annotations de référence."""

import re

from pydantic import BaseModel, ConfigDict

from contextiq.generation.prompt_builder import PromptBuilder
from contextiq.logging import log_operation
from contextiq.schemas.models import Document, EvaluationQuestion, ExtractiveAnswer


class AnswerEvaluation(BaseModel):
    """Contrôles par réponse ; les critères d'extraction valent None pour un refus."""

    model_config = ConfigDict(extra="forbid")

    question: EvaluationQuestion
    answer: ExtractiveAnswer
    reference_hit: bool | None
    verbatim_faithful: bool | None
    citation_valid: bool | None
    refusal_correct: bool | None
    scope_decision_correct: bool
    successful: bool


class AnswerEvaluationSummary(BaseModel):
    """Taux descriptifs ; None signifie que le dénominateur est nul."""

    model_config = ConfigDict(extra="forbid")

    question_count: int
    answered_count: int
    in_scope_count: int
    out_of_scope_count: int
    answer_rate: float
    success_rate: float
    faithfulness_rate: float | None
    citation_validity_rate: float | None
    relevant_source_rate: float | None
    false_refusal_rate: float | None
    correct_refusal_rate: float | None


class ExtractiveAnswerEvaluator:
    """Contrôler des extraits exacts, sans prétendre juger le sens d'une réponse libre.

    La pertinence est approchée par la présence d'une source annotée. La fidélité
    exige le passage original complet ; la citation est évaluée séparément. Ces contrôles ne constituent
    pas un évaluateur sémantique applicable aux réponses reformulées par un LLM.
    """

    def __init__(self, documents: list[Document]) -> None:
        """Garder une référence indépendante des sources déclarées par la réponse."""
        if not documents or len({doc.id for doc in documents}) != len(documents):
            raise ValueError("Fournir une base non vide avec des identifiants uniques.")
        self.documents = {doc.id: doc.model_copy(deep=True) for doc in documents}

    @log_operation
    def evaluate(self, answer: ExtractiveAnswer, question: EvaluationQuestion) -> AnswerEvaluation:
        """Comparer une réponse à sa question et à la base de référence.

        Une citation valide désigne un document connu et apparaît en fin d'extrait.
        Elle n'implique pas que ce document réponde à la question.
        Un refus est correct si aucune source n'est attendue, aucune n'est citée
        et le message standard de refus est restitué.
        """
        if answer.question != question.q:
            raise ValueError("La réponse et l'annotation doivent concerner la même question.")
        if set(question.gold_docs) - self.documents.keys():
            raise ValueError("Une source attendue est absente de la base de référence.")
        if len(question.gold_docs) != len(set(question.gold_docs)):
            raise ValueError("Les sources attendues doivent être uniques.")

        reference_hit = verbatim_faithful = citation_valid = refusal_correct = None
        scope_correct = answer.answered == bool(question.gold_docs)
        if answer.answered:
            source = answer.sources[0] if len(answer.sources) == 1 else None
            reference = self.documents.get(source.id) if source is not None else None
            citation = re.search(r" \[([^\[\]\n]+)\]$", answer.text)
            reference_hit = bool({doc.id for doc in answer.sources} & set(question.gold_docs))
            citation_valid = bool(
                reference is not None
                and citation is not None
                and citation.group(1) == reference.id
                and answer.text != reference.text
            )
            verbatim_faithful = bool(
                reference is not None
                and source.text == reference.text
                and (
                    answer.text == reference.text
                    or (citation is not None and answer.text[: citation.start()] == reference.text)
                )
            )
            successful = scope_correct and reference_hit and verbatim_faithful and citation_valid
        else:
            refusal_correct = (
                not question.gold_docs
                and not answer.sources
                and answer.text == PromptBuilder.REFUSAL_MESSAGE
            )
            successful = refusal_correct

        return AnswerEvaluation(
            question=question,
            answer=answer,
            reference_hit=reference_hit,
            verbatim_faithful=verbatim_faithful,
            citation_valid=citation_valid,
            refusal_correct=refusal_correct,
            scope_decision_correct=scope_correct,
            successful=successful,
        )

    @staticmethod
    def _rate(values: list[bool]) -> float | None:
        """Calculer une proportion sans inventer de score pour un groupe absent."""
        return sum(values) / len(values) if values else None

    @log_operation
    def summarize(self, cases: list[AnswerEvaluation]) -> AnswerEvaluationSummary:
        """Agréger en conservant un dénominateur explicite pour chaque critère.

        Fidélité et citations : toutes les réponses produites.
        Source pertinente : réponses produites aux questions avec source.
        Faux refus : toutes les questions avec source.
        Refus correct : toutes les questions hors périmètre.
        Succès : toutes les questions, avec critères conjoints par réponse.
        """
        if not cases:
            raise ValueError("Fournir au moins un résultat d'évaluation.")
        answered = [case for case in cases if case.answer.answered]
        in_scope = [case for case in cases if case.question.gold_docs]
        out_of_scope = [case for case in cases if not case.question.gold_docs]
        return AnswerEvaluationSummary(
            question_count=len(cases),
            answered_count=len(answered),
            in_scope_count=len(in_scope),
            out_of_scope_count=len(out_of_scope),
            answer_rate=len(answered) / len(cases),
            success_rate=sum(case.successful for case in cases) / len(cases),
            faithfulness_rate=self._rate([bool(case.verbatim_faithful) for case in answered]),
            citation_validity_rate=self._rate([bool(case.citation_valid) for case in answered]),
            relevant_source_rate=self._rate(
                [bool(case.reference_hit) for case in in_scope if case.answer.answered]
            ),
            false_refusal_rate=self._rate([not case.answer.answered for case in in_scope]),
            correct_refusal_rate=self._rate([bool(case.refusal_correct) for case in out_of_scope]),
        )
