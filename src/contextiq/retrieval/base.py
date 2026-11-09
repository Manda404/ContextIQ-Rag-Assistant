"""Interface commune utilisée pour comparer les retrievers."""

from typing import Protocol

from contextiq.schemas.models import RetrievalResult


class Retriever(Protocol):
    """Tout retriever offrant retrieve peut être évalué, sans héritage imposé."""

    def retrieve(self, question: str, top_k: int = 3) -> list[RetrievalResult]:
        """Retourner au plus K résultats dans l'ordre de pertinence."""
        ...
