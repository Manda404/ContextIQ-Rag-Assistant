"""Journal local des questions, réponses et revues humaines, avec écritures atomiques."""

import sqlite3
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from contextiq.schemas.models import RetrievalResult


class ConversationRecord(BaseModel):
    """Une requête réellement traitée, distincte des métriques du jeu d'évaluation."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    question: str
    method: str
    top_k: int
    threshold: float
    status: Literal["answered", "refused", "error"]
    answer: str | None = None
    reason: str | None = None
    error_type: str | None = None
    best_score: float | None = None
    retrieved_sources: list[RetrievalResult] = Field(default_factory=list)
    cited_source_ids: list[str] = Field(default_factory=list)
    elapsed_ms: float = Field(ge=0)
    corpus_sha256: str
    configuration_sha256: str
    prompt_sha256: str


class ConversationFeedback(BaseModel):
    """Une appréciation humaine révisable, sans inférer une vérité depuis la similarité."""

    model_config = ConfigDict(extra="forbid")

    interaction_id: str
    verdict: Literal["helpful", "irrelevant", "incorrect", "incomplete"]
    comment: str = Field(default="", max_length=2000)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ConversationSummary(BaseModel):
    """Effectifs sur tout le journal ; les taux humains portent seulement sur les cas revus."""

    total: int
    answered: int
    refused: int
    errors: int
    reviewed: int
    flagged: int
    refusal_rate: float | None
    error_rate: float | None
    flagged_review_rate: float | None


class ConversationStore:
    """SQLite conserve les textes localement et supporte plusieurs sessions simultanées."""

    def __init__(self, path: str | Path) -> None:
        """Créer le journal et les tables si nécessaire ; ne jamais publier leur contenu."""
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                    status TEXT NOT NULL, payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    interaction_id TEXT PRIMARY KEY REFERENCES interactions(id),
                    verdict TEXT NOT NULL, payload TEXT NOT NULL
                );
            """)
        self.path.chmod(0o600)

    @contextmanager
    def _connect(self):
        """Fermer chaque connexion après transaction et préserver l'intégrité des retours."""
        with closing(sqlite3.connect(self.path, timeout=10)) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            with connection:
                yield connection

    def append(self, record: ConversationRecord) -> None:
        """Enregistrer une seule fois une requête avec ses sources et sa version de corpus."""
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO interactions VALUES (?, ?, ?, ?)",
                (record.id, record.created_at.isoformat(), record.status, record.model_dump_json()),
            )

    def recent(self, limit: int = 200) -> list[ConversationRecord]:
        """Relire les derniers échanges, y compris ceux des sessions précédentes."""
        if type(limit) is not int or limit < 1:
            raise ValueError("limit doit être un entier positif.")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM interactions ORDER BY created_at DESC, id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [ConversationRecord.model_validate_json(row[0]) for row in rows]

    def review(self, feedback: ConversationFeedback) -> None:
        """Créer ou corriger la revue d'un échange existant sans compter deux fois cet échange."""
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO feedback VALUES (?, ?, ?) ON CONFLICT(interaction_id) DO UPDATE SET "
                "verdict=excluded.verdict, payload=excluded.payload",
                (feedback.interaction_id, feedback.verdict, feedback.model_dump_json()),
            )

    def get_feedback(self, interaction_id: str) -> ConversationFeedback | None:
        """Retrouver le dernier avis humain associé à une requête."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM feedback WHERE interaction_id=?", (interaction_id,)
            ).fetchone()
        return ConversationFeedback.model_validate_json(row[0]) if row else None

    def summarize(self) -> ConversationSummary:
        """Distinguer les incidents techniques et les problèmes identifiés par une revue humaine."""
        with self._connect() as connection:
            counts = dict(
                connection.execute(
                    "SELECT status, count(*) FROM interactions GROUP BY status"
                ).fetchall()
            )
            reviewed, flagged = connection.execute(
                "SELECT count(*), coalesce(sum(verdict != 'helpful'), 0) FROM feedback"
            ).fetchone()
        total = sum(counts.values())
        return ConversationSummary(
            total=total,
            answered=counts.get("answered", 0),
            refused=counts.get("refused", 0),
            errors=counts.get("error", 0),
            reviewed=reviewed,
            flagged=flagged,
            refusal_rate=counts.get("refused", 0) / total if total else None,
            error_rate=counts.get("error", 0) / total if total else None,
            flagged_review_rate=flagged / reviewed if reviewed else None,
        )
