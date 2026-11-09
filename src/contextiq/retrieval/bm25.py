"""BM25 avec saturation des fréquences et normalisation des longueurs."""

from collections import Counter
from math import log1p

from pydantic import BaseModel, ConfigDict, Field

from contextiq.ingestion.preprocessing import TextPreprocessor
from contextiq.logging import log_operation
from contextiq.schemas.models import Document, RetrievalResult


class BM25Parameters(BaseModel):
    """Paramètres fixes de départ, sans optimisation sur le jeu d'évaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    k1: float = Field(default=1.5, gt=0, allow_inf_nan=False)
    b: float = Field(default=0.75, ge=0, le=1, allow_inf_nan=False)


class BM25Retriever:
    """Utiliser le même prétraitement et la même interface que TF-IDF."""

    @log_operation
    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75) -> None:
        """Calculer les fréquences, longueurs et IDF sur les documents seuls."""
        self.parameters = BM25Parameters(k1=k1, b=b)
        if not documents:
            raise ValueError("La base documentaire ne doit pas être vide.")
        if len({document.id for document in documents}) != len(documents):
            raise ValueError("Les identifiants des documents doivent être uniques.")

        self.documents = [document.model_copy(deep=True) for document in documents]
        self.preprocessor = TextPreprocessor()
        tokens = [self.preprocessor.tokenize(document.text) for document in self.documents]
        self.term_frequencies = [Counter(words) for words in tokens]
        self.document_lengths = [len(words) for words in tokens]
        self.average_length = sum(self.document_lengths) / len(documents)
        if self.average_length == 0:
            raise ValueError("La base documentaire doit contenir des termes indexables.")

        # Chaque terme est compté une seule fois par document pour calculer l'IDF.
        document_frequencies = Counter(term for words in tokens for term in set(words))
        count = len(documents)
        self.idf = {
            term: log1p((count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequencies.items()
        }

    def _score(self, terms: set[str], index: int) -> float:
        """Additionner les contributions BM25 des termes de la question."""
        k1 = self.parameters.k1
        b = self.parameters.b
        length_ratio = self.document_lengths[index] / self.average_length
        normalization = k1 * (1 - b + b * length_ratio)
        score = 0.0
        # Un tri stabilise également l'ordre des additions entre exécutions.
        for term in sorted(terms):
            frequency = self.term_frequencies[index].get(term, 0)
            if frequency:
                score += self.idf[term] * frequency * (k1 + 1) / (frequency + normalization)
        return score

    @log_operation
    def retrieve(self, question: str, top_k: int = 3) -> list[RetrievalResult]:
        """Classer les documents par score BM25, sans retourner les scores nuls."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("La question doit être un texte non vide.")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k doit être un entier strictement positif.")

        # Répéter un mot dans la question ne change pas son poids dans cette variante.
        terms = set(self.preprocessor.tokenize(question))
        scores = [self._score(terms, index) for index in range(len(self.documents))]
        indices = sorted(
            range(len(self.documents)),
            key=lambda index: (-scores[index], self.documents[index].id),
        )
        return [
            RetrievalResult(document=self.documents[index], score=scores[index])
            for index in indices[:top_k]
            if scores[index] > 0
        ]
