"""Schémas des fichiers fournis et contrôles avant les expériences."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Refuser les chaînes vides sans modifier le texte original des sources.
NonEmptyText = Annotated[str, Field(strict=True, pattern=r"\S")]


class Document(BaseModel):
    """Un document de référence, identifié par son identifiant de source."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: NonEmptyText
    text: NonEmptyText


class EvaluationQuestion(BaseModel):
    """Une question annotée ; gold_docs=[] signifie hors périmètre."""

    model_config = ConfigDict(extra="forbid", strict=True)

    q: NonEmptyText
    # Champ obligatoire : une annotation absente n'est pas un refus attendu.
    gold_docs: list[NonEmptyText]


class RetrievalResult(BaseModel):
    """Un document retrouvé et son score, qui n'est pas une probabilité."""

    model_config = ConfigDict(extra="forbid", strict=True)

    document: Document
    score: float = Field(ge=0, allow_inf_nan=False)


class AbstentionDecision(BaseModel):
    """Décision du filtre lexical, sans garantie de réponse correcte."""

    model_config = ConfigDict(extra="forbid", strict=True)

    accepted: bool
    reason: Literal["no_results", "no_positive_score", "below_threshold", "threshold_met"]
    threshold: float = Field(ge=0, allow_inf_nan=False)
    best_score: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    results: list[RetrievalResult]


class PromptMessage(BaseModel):
    """Un message structuré : les documents ne peuvent pas créer un rôle système."""

    model_config = ConfigDict(extra="forbid", strict=True)

    role: Literal["system", "user"]
    content: NonEmptyText


class PromptBuildResult(BaseModel):
    """Prompt préparé ou refus explicite ; aucun LLM n'est appelé."""

    model_config = ConfigDict(extra="forbid", strict=True)

    ready: bool
    reason: Literal[
        "ready", "abstention", "empty_context", "suspicious_question", "suspicious_context"
    ]
    messages: list[PromptMessage] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    blocked_source_ids: list[str] = Field(default_factory=list)
    refusal_message: str | None = None


class ExtractiveAnswer(BaseModel):
    """Extrait sourcé ou refus ; answered ne garantit pas la pertinence du passage."""

    model_config = ConfigDict(extra="forbid", strict=True)

    question: NonEmptyText
    answered: bool
    reason: Literal[
        "extractive", "abstention", "empty_context", "suspicious_question", "suspicious_context"
    ]
    text: NonEmptyText
    sources: list[Document] = Field(default_factory=list)
    blocked_source_ids: list[str] = Field(default_factory=list)


class Dataset(BaseModel):
    """Documents et questions cohérents, prêts pour les futurs retrievers."""

    model_config = ConfigDict(extra="forbid", strict=True)

    documents: list[Document] = Field(min_length=1)
    questions: list[EvaluationQuestion] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_sources(self) -> "Dataset":
        """Refuser les identifiants ambigus et les annotations introuvables."""
        document_ids = [document.id for document in self.documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("Les identifiants des documents doivent être uniques.")

        known_ids = set(document_ids)
        for question in self.questions:
            if len(question.gold_docs) != len(set(question.gold_docs)):
                raise ValueError(f"Sources répétées pour la question : {question.q}")
            unknown_ids = set(question.gold_docs) - known_ids
            if unknown_ids:
                raise ValueError(f"Sources absentes de la base : {sorted(unknown_ids)}")
        return self
