"""Charger les paramètres YAML et les valider avant les expériences."""

import os
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from contextiq.logging import LoggingConfig, configure_logging, load_logging_config, log_operation
from contextiq.retrieval.bm25 import BM25Parameters

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
Threshold = Annotated[float, Field(ge=0, allow_inf_nan=False)]
METHODS = {"TF-IDF", "BM25", "Embeddings"}


class ConfigModel(BaseModel):
    """Refuser les clés inconnues pour détecter les fautes dans les YAML."""

    model_config = ConfigDict(extra="forbid", strict=True)


class EmbeddingSettings(ConfigModel):
    """Choisir les poids et leur cache, sans chemin propre à une machine."""

    model_name: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    local_files_only: bool
    cache_dir: str | None


class RetrievalSettings(ConfigModel):
    """Paramètres des retrievers et seuils de la politique d'abstention."""

    top_k: PositiveInt
    bm25: BM25Parameters
    embeddings: EmbeddingSettings
    abstention_thresholds: dict[str, Threshold]

    @field_validator("abstention_thresholds")
    @classmethod
    def validate_methods(cls, values: dict[str, float]) -> dict[str, float]:
        """Exiger un seuil pour chacune des trois méthodes comparées."""
        if set(values) != METHODS:
            raise ValueError("Fournir les méthodes TF-IDF, BM25 et Embeddings exactement.")
        return values


class EvaluationSettings(ConfigModel):
    """Valeurs de K à comparer et grilles de seuils exploratoires."""

    k_values: list[PositiveInt] = Field(min_length=1)
    inspection_k: PositiveInt
    threshold_grids: dict[str, list[Threshold]]

    @model_validator(mode="after")
    def validate_experiments(self) -> "EvaluationSettings":
        """Vérifier que les expériences peuvent être parcourues sans ambiguïté."""
        if len(self.k_values) != len(set(self.k_values)):
            raise ValueError("k_values ne doit pas contenir de doublons.")
        if self.inspection_k not in self.k_values:
            raise ValueError("inspection_k doit appartenir à k_values.")
        if set(self.threshold_grids) != METHODS:
            raise ValueError("Fournir les grilles TF-IDF, BM25 et Embeddings exactement.")
        if any(not values for values in self.threshold_grids.values()):
            raise ValueError("Chaque grille de seuils doit contenir au moins une valeur.")
        return self


class MonitoringSettings(ConfigModel):
    """Conserver les expériences MLflow sur la machine, sans serveur externe."""

    experiment_name: str = Field(default="contextiq-rag-assistant", pattern=r"\S")
    storage_dir: str = Field(default="mlflow", pattern=r"\S")
    port: int = Field(default=5000, ge=1024, le=65535)


class Settings(ConfigModel):
    """Point d'entrée unique pour lire les configurations du projet."""

    retrieval: RetrievalSettings
    evaluation: EvaluationSettings
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)

    @classmethod
    @log_operation
    def load(cls, config_dir: str | Path | None = None) -> "Settings":
        """Priorité : argument, CONTEXTIQ_CONFIG_DIR, puis configs/ du projet.

        Les chemins relatifs explicitement fournis sont relatifs au dossier courant.
        Aucun fichier .env n'est chargé automatiquement. Les fichiers manquants
        ou invalides provoquent une erreur plutôt qu'un remplacement silencieux.
        """
        configured = config_dir if config_dir is not None else os.getenv("CONTEXTIQ_CONFIG_DIR")
        if configured is not None:
            directory = Path(configured).expanduser().resolve()
        else:
            directory = cls._find_config_dir()
        settings = cls(
            retrieval=yaml.safe_load((directory / "retrieval.yaml").read_text(encoding="utf-8")),
            evaluation=yaml.safe_load((directory / "evaluation.yaml").read_text(encoding="utf-8")),
            logging=load_logging_config(directory),
            monitoring=yaml.safe_load((directory / "monitoring.yaml").read_text(encoding="utf-8"))
            if (directory / "monitoring.yaml").exists()
            else MonitoringSettings(),
        )
        # Un dossier relatif est ancré au parent de configs/, pas au notebook courant.
        storage = Path(settings.monitoring.storage_dir).expanduser()
        if not storage.is_absolute():
            storage = directory.parent / storage
        settings.monitoring.storage_dir = str(storage.resolve())
        # Un cache relatif au YAML garde le même emplacement depuis les notebooks.
        cache_dir = settings.retrieval.embeddings.cache_dir
        if cache_dir is not None:
            cache_path = Path(cache_dir).expanduser()
            if not cache_path.is_absolute():
                cache_path = directory / cache_path
            settings.retrieval.embeddings.cache_dir = str(cache_path.resolve())
        configure_logging(config=settings.logging)
        return settings

    @staticmethod
    def _find_config_dir() -> Path:
        """Retrouver la configuration après déplacement du projet ou depuis un notebook."""
        current = Path.cwd().resolve()
        for directory in (current, *current.parents):
            if (directory / "pyproject.toml").is_file() and (
                directory / "src" / "contextiq"
            ).is_dir():
                return directory / "configs"
        raise FileNotFoundError(
            "Racine du projet introuvable. Fournir config_dir ou CONTEXTIQ_CONFIG_DIR."
        )
