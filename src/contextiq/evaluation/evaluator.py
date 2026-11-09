"""Évaluer TF-IDF et BM25 avec le même protocole et garder le détail."""

from statistics import mean

from pydantic import BaseModel, ConfigDict

from contextiq.evaluation.retrieval_metrics import RetrievalScores, compute_retrieval_metrics
from contextiq.logging import log_operation
from contextiq.retrieval.base import Retriever
from contextiq.schemas.models import EvaluationQuestion, RetrievalResult


class QuestionEvaluation(BaseModel):
    """Résultats par question ; metrics=None pour le hors périmètre."""

    model_config = ConfigDict(extra="forbid")

    question: EvaluationQuestion
    results: list[RetrievalResult]
    metrics: RetrievalScores | None


class RetrievalEvaluation(BaseModel):
    """Moyennes sur les seules questions avec source et détails inspectables."""

    model_config = ConfigDict(extra="forbid")

    top_k: int
    in_scope_count: int
    out_of_scope_count: int
    summary: RetrievalScores | None
    questions: list[QuestionEvaluation]


class RetrievalEvaluator:
    """Appeler un retriever sans le réentraîner ni régler ses paramètres."""

    @log_operation
    def evaluate(
        self,
        retriever: Retriever,
        questions: list[EvaluationQuestion],
        top_k: int = 3,
    ) -> RetrievalEvaluation:
        """Calculer les métriques par question, puis leur moyenne arithmétique."""
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k doit être un entier strictement positif.")
        if not questions:
            raise ValueError("Le jeu de questions ne doit pas être vide.")

        evaluations = []
        for question in questions:
            results = retriever.retrieve(question.q, top_k=top_k)
            metrics = None
            if question.gold_docs:
                metrics = compute_retrieval_metrics(
                    [result.document.id for result in results],
                    question.gold_docs,
                    top_k,
                )
            evaluations.append(
                QuestionEvaluation(question=question, results=results, metrics=metrics)
            )

        # Ne jamais transformer une question hors périmètre en un score zéro.
        in_scope_scores = [item.metrics for item in evaluations if item.metrics is not None]
        summary = None
        if in_scope_scores:
            summary = RetrievalScores(
                precision=mean(score.precision for score in in_scope_scores),
                recall=mean(score.recall for score in in_scope_scores),
                hit_rate=mean(score.hit_rate for score in in_scope_scores),
                mrr=mean(score.mrr for score in in_scope_scores),
            )
        return RetrievalEvaluation(
            top_k=top_k,
            in_scope_count=len(in_scope_scores),
            out_of_scope_count=len(evaluations) - len(in_scope_scores),
            summary=summary,
            questions=evaluations,
        )
