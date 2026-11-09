"""Première baseline lexicale : TF-IDF et similarité cosinus."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from contextiq.ingestion.preprocessing import TextPreprocessor
from contextiq.logging import log_operation
from contextiq.schemas.models import Document, RetrievalResult


class TfidfRetriever:
    """Indexer uniquement les documents, puis rechercher des questions inédites."""

    @log_operation
    def __init__(self, documents: list[Document]) -> None:
        """Construire l'index sans utiliser les questions ni leurs annotations."""
        if not documents:
            raise ValueError("La base documentaire ne doit pas être vide.")
        if len({document.id for document in documents}) != len(documents):
            raise ValueError("Les identifiants des documents doivent être uniques.")

        # Une copie garde l'index cohérent si les objets d'entrée sont modifiés.
        self.documents = [document.model_copy(deep=True) for document in documents]
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(
            tokenizer=self.preprocessor.tokenize,
            token_pattern=None,
            lowercase=False,  # Déjà géré par notre prétraitement commun.
        )
        self.document_vectors = self.vectorizer.fit_transform(
            [document.text for document in self.documents]
        )

    @log_operation
    def retrieve(self, question: str, top_k: int = 3) -> list[RetrievalResult]:
        """Retourner au plus K documents, classés par similarité décroissante.

        Les scores nuls sont exclus. Un score positif indique un recouvrement
        lexical, pas que le document suffit à répondre à la question.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("La question doit être un texte non vide.")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k doit être un entier strictement positif.")

        # transform réutilise le vocabulaire et l'IDF appris sur les documents.
        question_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(question_vector, self.document_vectors)[0]

        # L'identifiant départage les égalités pour rendre l'ordre reproductible.
        indices = sorted(
            range(len(self.documents)),
            key=lambda index: (-float(scores[index]), self.documents[index].id),
        )
        return [
            RetrievalResult(document=self.documents[index], score=float(scores[index]))
            for index in indices[:top_k]
            if scores[index] > 0
        ]
