"""Relier l'interface aux classes métier, sans dépendre de Streamlit."""

from time import perf_counter
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from contextiq.generation.extractive_answer import ExtractiveAnswerGenerator
from contextiq.generation.prompt_builder import PromptBuilder
from contextiq.logging import configure_logging, log_operation, logger
from contextiq.monitoring.conversations import ConversationRecord, ConversationStore
from contextiq.monitoring.mlflow_tracking import MLflowTracker, TrackedEvaluation, fingerprint
from contextiq.retrieval.base import Retriever
from contextiq.retrieval.bm25 import BM25Retriever
from contextiq.retrieval.tfidf import TfidfRetriever
from contextiq.safety.abstention import AbstentionPolicy
from contextiq.schemas.models import (
    AbstentionDecision,
    Dataset,
    ExtractiveAnswer,
    PromptBuildResult,
)
from contextiq.settings import Settings

Method = Literal["TF-IDF", "BM25", "Embeddings"]


class SimulationTurn(BaseModel):
    """Conserver le résultat d'une question avec les paramètres réellement utilisés."""

    model_config = ConfigDict(extra="forbid")

    method: Method
    top_k: int
    decision: AbstentionDecision
    answer: ExtractiveAnswer
    prompt: PromptBuildResult
    elapsed_ms: float
    interaction_id: str | None = None


class AssistantSession:
    """Charger chaque retriever une fois par session, à la première utilisation."""

    def __init__(self, dataset: Dataset, settings: Settings) -> None:
        """Garder une copie indépendante des données et des paramètres de l'interface."""
        self.dataset = dataset.model_copy(deep=True)
        self.settings = settings.model_copy(deep=True)
        self.retrievers: dict[str, Retriever] = {}
        self.session_id = uuid4().hex
        self.conversations = (
            ConversationStore(
                configure_logging(config=settings.logging).parent / "conversations.db"
            )
            if settings.monitoring.log_conversations
            else None
        )
        self.corpus_sha256 = fingerprint([doc.model_dump() for doc in self.dataset.documents])

    def get_retriever(self, method: Method) -> Retriever:
        """Initialiser uniquement la méthode demandée, sans télécharger de poids par défaut."""
        if method not in ("TF-IDF", "BM25", "Embeddings"):
            raise ValueError("Choisir TF-IDF, BM25 ou Embeddings.")
        if method not in self.retrievers:
            documents = self.dataset.documents
            if method == "TF-IDF":
                retriever = TfidfRetriever(documents)
            elif method == "BM25":
                retriever = BM25Retriever(documents, **self.settings.retrieval.bm25.model_dump())
            else:
                # Le modèle sémantique n'est importé et chargé que lorsqu'il est choisi.
                from contextiq.retrieval.embeddings import EmbeddingRetriever

                retriever = EmbeddingRetriever(
                    documents, **self.settings.retrieval.embeddings.model_dump()
                )
            self.retrievers[method] = retriever
        return self.retrievers[method]

    @log_operation
    def ask(self, question: str, method: Method, top_k: int, threshold: float) -> SimulationTurn:
        """Traiter une question indépendante ; l'historique ne devient pas un contexte LLM."""
        start = perf_counter()
        metadata = dict(
            session_id=self.session_id,
            question=question,
            method=method,
            top_k=top_k,
            threshold=threshold,
            corpus_sha256=self.corpus_sha256,
            configuration_sha256=fingerprint(self.settings.retrieval.model_dump()),
            prompt_sha256=fingerprint(PromptBuilder.SYSTEM_PROMPT),
        )
        try:
            retriever = self.get_retriever(method)
            results = retriever.retrieve(question, top_k=top_k)
            decision = AbstentionPolicy(threshold=threshold).decide(results)
            answer = ExtractiveAnswerGenerator().generate(question, decision)
            prompt = PromptBuilder().build(question, decision)
        except Exception as error:
            if self.conversations is not None:
                self.conversations.append(
                    ConversationRecord(
                        **metadata,
                        status="error",
                        error_type=type(error).__name__,
                        elapsed_ms=1000 * (perf_counter() - start),
                    )
                )
            raise
        turn = SimulationTurn(
            method=method,
            top_k=top_k,
            decision=decision,
            answer=answer,
            prompt=prompt,
            elapsed_ms=1000 * (perf_counter() - start),
        )
        if self.conversations is not None:
            record = ConversationRecord(
                **metadata,
                status="answered" if answer.answered else "refused",
                answer=answer.text,
                reason=answer.reason,
                best_score=decision.best_score,
                retrieved_sources=results,
                cited_source_ids=[source.id for source in answer.sources],
                elapsed_ms=turn.elapsed_ms,
            )
            # Enregistrer avant d'afficher : une réponse non journalisée n'est pas présentée.
            self.conversations.append(record)
            turn.interaction_id = record.id
            logger.bind(interaction_id=record.id, conversation_session_id=self.session_id).info(
                "Échange enregistré dans le journal local"
            )
        return turn

    def evaluate(self, method: Method, top_k: int) -> TrackedEvaluation:
        """Créer un run MLflow à la demande, avec les seuils YAML de l'évaluation."""
        retriever = self.get_retriever(method)
        return MLflowTracker(self.settings).evaluate(method, retriever, self.dataset, top_k=top_k)
