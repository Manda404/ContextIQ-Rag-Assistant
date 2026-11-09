"""Première règle d'abstention fondée sur le meilleur score lexical."""

from pydantic import BaseModel, ConfigDict, Field

from contextiq.logging import log_operation
from contextiq.schemas.models import AbstentionDecision, RetrievalResult


class AbstentionPolicy(BaseModel):
    """Appliquer un seuil explicite, à définir séparément pour chaque retriever.

    Aucun seuil par défaut : les scores TF-IDF et BM25 ont des échelles différentes.
    Cette règle est une heuristique ; accepter ne prouve pas que le contexte suffit.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    threshold: float = Field(ge=0, allow_inf_nan=False)

    @log_operation
    def decide(self, results: list[RetrievalResult]) -> AbstentionDecision:
        """Refuser sans résultat positif ou sous le seuil ; accepter à égalité.

        Les résultats restent disponibles pour l'analyse, y compris lors d'un refus.
        Le maximum est calculé explicitement sans dépendre de l'ordre de la liste.
        """
        best_score = max((result.score for result in results), default=None)
        if best_score is None:
            reason = "no_results"
        elif best_score == 0:
            reason = "no_positive_score"
        elif best_score < self.threshold:
            reason = "below_threshold"
        else:
            reason = "threshold_met"

        return AbstentionDecision(
            accepted=reason == "threshold_met",
            reason=reason,
            threshold=self.threshold,
            best_score=best_score,
            results=results,
        )
