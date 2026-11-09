"""Chargement local des JSON, sans chemin propre à une machine."""

import os
from pathlib import Path

from pydantic import TypeAdapter

from contextiq.logging import log_operation
from contextiq.schemas.models import Dataset, Document, EvaluationQuestion


class DataLoader:
    """Charger les données depuis un dossier configurable.

    Priorité : argument data_dir, variable CONTEXTIQ_DATA_DIR, puis data/raw
    du projet détecté depuis le dossier courant (racine ou notebook).
    Les chemins relatifs explicites sont relatifs au dossier courant.
    """

    @log_operation
    def __init__(self, data_dir: str | Path | None = None) -> None:
        """Résoudre le dossier une seule fois pour les deux fichiers."""
        configured_dir = data_dir if data_dir is not None else os.getenv("CONTEXTIQ_DATA_DIR")
        if configured_dir is not None:
            self.data_dir = Path(configured_dir).expanduser().resolve()
        else:
            self.data_dir = self._find_data_dir()

    @staticmethod
    def _find_data_dir() -> Path:
        """Remonter depuis le notebook jusqu'à la racine de ContextIQ."""
        current = Path.cwd().resolve()
        for directory in (current, *current.parents):
            if (directory / "pyproject.toml").is_file() and (
                directory / "src" / "contextiq"
            ).is_dir():
                return directory / "data" / "raw"
        raise FileNotFoundError(
            "Racine du projet introuvable. Fournir data_dir ou CONTEXTIQ_DATA_DIR."
        )

    @log_operation
    def load_documents(self) -> list[Document]:
        """Lire la base UTF-8 et valider chaque document avec Pydantic."""
        path = self.data_dir / "knowledge_base.json"
        return TypeAdapter(list[Document]).validate_json(path.read_text(encoding="utf-8"))

    @log_operation
    def load_questions(self) -> list[EvaluationQuestion]:
        """Lire les annotations en conservant les sources vides hors périmètre."""
        path = self.data_dir / "eval_set.json"
        return TypeAdapter(list[EvaluationQuestion]).validate_json(path.read_text(encoding="utf-8"))

    @log_operation
    def load(self) -> Dataset:
        """Charger les deux fichiers et vérifier leurs références croisées."""
        return Dataset(
            documents=self.load_documents(),
            questions=self.load_questions(),
        )
