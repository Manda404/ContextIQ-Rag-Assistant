"""Restituer un passage documentaire cité, sans reformulation ni appel LLM."""

from contextiq.generation.prompt_builder import PromptBuilder
from contextiq.logging import log_operation
from contextiq.schemas.models import AbstentionDecision, ExtractiveAnswer


class ExtractiveAnswerGenerator:
    """Transformer une recherche acceptée en un extrait accompagné de sa source."""

    def __init__(self) -> None:
        """Réutiliser les mêmes contrôles d'abstention et de sécurité que le prompt."""
        self.prompt_builder = PromptBuilder()

    @log_operation
    def generate(self, question: str, decision: AbstentionDecision) -> ExtractiveAnswer:
        """Retourner le meilleur document autorisé ou transmettre le refus.

        Le document est conservé entier pour préserver ses conditions et plafonds.
        Aucun raisonnement métier, résumé ou contrôle de suffisance n'est réalisé :
        une recherche acceptée peut encore sélectionner un passage non pertinent.
        """
        checked = self.prompt_builder.build(question, decision)
        if not checked.ready:
            return ExtractiveAnswer(
                question=question,
                answered=False,
                reason=checked.reason,
                text=checked.refusal_message or self.prompt_builder.REFUSAL_MESSAGE,
                blocked_source_ids=checked.blocked_source_ids,
            )

        # Se limiter aux sources qui ont franchi les contrôles précédents.
        allowed_results = [
            result for result in decision.results if result.document.id in checked.source_ids
        ]
        # Le score décroissant et l'identifiant départagent les résultats comme les retrievers.
        best = min(allowed_results, key=lambda result: (-result.score, result.document.id))
        source = best.document.model_copy(deep=True)
        return ExtractiveAnswer(
            question=question,
            answered=True,
            reason="extractive",
            text=f"{source.text} [{source.id}]",
            sources=[source],
        )
