"""Mesurer les refus sans les confondre avec la qualité des réponses."""

from pydantic import BaseModel, ConfigDict

from contextiq.logging import log_operation
from contextiq.retrieval.base import Retriever
from contextiq.safety.abstention import AbstentionPolicy
from contextiq.schemas.models import AbstentionDecision, EvaluationQuestion


class AbstentionCase(BaseModel):
    """Une annotation et la décision prise sans consulter cette annotation."""

    model_config = ConfigDict(extra="forbid")

    question: EvaluationQuestion
    decision: AbstentionDecision


class AbstentionReport(BaseModel):
    """Compteurs et taux ; None signale qu'un groupe est absent du jeu."""

    model_config = ConfigDict(extra="forbid")

    threshold: float
    in_scope_count: int
    out_of_scope_count: int
    false_refusals: int
    false_acceptances: int
    false_refusal_rate: float | None
    false_acceptance_rate: float | None
    acceptance_rate: float
    cases: list[AbstentionCase]


class AbstentionEvaluator:
    """Évaluer une politique fixée à l'avance, sans choisir son seuil."""

    @log_operation
    def evaluate(
        self,
        retriever: Retriever,
        questions: list[EvaluationQuestion],
        policy: AbstentionPolicy,
        top_k: int = 3,
    ) -> AbstentionReport:
        """Comparer les décisions aux annotations de périmètre.

        Faux refus : question avec source refusée / questions avec source.
        Fausse acceptation : hors périmètre accepté / questions hors périmètre.
        Le taux d'acceptation porte sur toutes les questions. Il ne mesure pas
        l'exactitude : une question acceptée peut avoir de mauvaises sources.
        """
        if not questions:
            raise ValueError("Le jeu de questions ne doit pas être vide.")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k doit être un entier strictement positif.")

        cases = [
            AbstentionCase(
                question=question,
                decision=policy.decide(retriever.retrieve(question.q, top_k=top_k)),
            )
            for question in questions
        ]
        in_scope_count = sum(bool(case.question.gold_docs) for case in cases)
        out_of_scope_count = len(cases) - in_scope_count
        false_refusals = sum(
            bool(case.question.gold_docs) and not case.decision.accepted for case in cases
        )
        false_acceptances = sum(
            not case.question.gold_docs and case.decision.accepted for case in cases
        )
        return AbstentionReport(
            threshold=policy.threshold,
            in_scope_count=in_scope_count,
            out_of_scope_count=out_of_scope_count,
            false_refusals=false_refusals,
            false_acceptances=false_acceptances,
            false_refusal_rate=false_refusals / in_scope_count if in_scope_count else None,
            false_acceptance_rate=false_acceptances / out_of_scope_count
            if out_of_scope_count
            else None,
            acceptance_rate=sum(case.decision.accepted for case in cases) / len(cases),
            cases=cases,
        )
