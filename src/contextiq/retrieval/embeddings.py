"""Recherche sémantique locale avec un modèle multilingue préentraîné."""

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from contextiq.logging import log_operation
from contextiq.schemas.models import Document, RetrievalResult


class EmbeddingRetriever:
    """Comparer les embeddings des questions et des documents par cosinus."""

    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"

    @log_operation
    def __init__(
        self,
        documents: list[Document],
        *,
        local_files_only: bool = True,
        cache_dir: str | Path | None = None,
        model_name: str = MODEL_NAME,
        revision: str = MODEL_REVISION,
    ) -> None:
        """Charger le modèle sur CPU et encoder uniquement les documents.

        Par défaut, le modèle doit déjà être en cache : aucun réseau n'est requis.
        Au premier usage, local_files_only=False autorise le téléchargement des
        poids publics. L'encodage des textes s'effectue toujours sur la machine.
        """
        if not documents:
            raise ValueError("La base documentaire ne doit pas être vide.")
        if len({document.id for document in documents}) != len(documents):
            raise ValueError("Les identifiants des documents doivent être uniques.")

        self.documents = [document.model_copy(deep=True) for document in documents]
        self.model_name = model_name
        self.revision = revision
        self.model = SentenceTransformer(
            self.model_name,
            revision=self.revision,
            device="cpu",
            local_files_only=local_files_only,
            cache_folder=str(cache_dir) if cache_dir is not None else None,
            trust_remote_code=False,
            token=False,
        )
        texts = [document.text for document in self.documents]
        self._check_length(texts)
        # Les textes originaux sont confiés au tokenizer du modèle, sans nettoyage lexical.
        self.document_vectors = self.model.encode_document(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def _check_length(self, texts: list[str]) -> None:
        """Refuser un texte trop long plutôt que tronquer silencieusement sa fin."""
        token_ids = self.model.tokenizer(texts, truncation=False, padding=False)["input_ids"]
        if any(len(tokens) > self.model.max_seq_length for tokens in token_ids):
            raise ValueError(
                f"Texte trop long pour le modèle ({self.model.max_seq_length} tokens maximum). "
                "Découper les documents ou raccourcir la question avant la recherche."
            )

    @log_operation
    def retrieve(self, question: str, top_k: int = 3) -> list[RetrievalResult]:
        """Retourner les meilleurs cosinus positifs, sans seuil d'abstention implicite."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("La question doit être un texte non vide.")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k doit être un entier strictement positif.")
        self._check_length([question])
        question_vector = self.model.encode_query(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # Le produit scalaire de vecteurs normalisés est leur similarité cosinus.
        # clip corrige les petits dépassements numériques au voisinage de +/-1.
        scores = np.clip(self.document_vectors @ question_vector[0], -1.0, 1.0)
        indices = sorted(
            range(len(self.documents)),
            key=lambda index: (-float(scores[index]), self.documents[index].id),
        )
        return [
            RetrievalResult(document=self.documents[index], score=float(scores[index]))
            for index in indices[:top_k]
            if scores[index] > 0
        ]
