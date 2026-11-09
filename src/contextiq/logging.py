"""Journalisation centralisée du pipeline, sans enregistrer les textes métier."""

import os
import sys
from collections.abc import Callable
from contextlib import suppress
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Literal, ParamSpec, TypeVar
from uuid import uuid4

import yaml
from loguru import logger as base_logger
from pydantic import BaseModel, ConfigDict, Field

P = ParamSpec("P")
R = TypeVar("R")
LOG_FILENAME = "contextiq-rag-assistant.log"
logger = base_logger.bind(
    application="contextiq",
    session_id=uuid4().hex[:12],
    context=os.getenv("CONTEXTIQ_LOG_CONTEXT", "application"),
)
_lock = RLock()
_sink_id: int | None = None
_console_sink_id: int | None = None
_log_path: Path | None = None
_level: str | None = None
_rotation: str | None = None
_retention: str | None = None
_trace_id: ContextVar[str | None] = ContextVar("contextiq_trace_id", default=None)


class LoggingConfig(BaseModel):
    """Paramètres Loguru ; les sorties des notebooks sont désactivées par défaut."""

    model_config = ConfigDict(extra="forbid", strict=True)

    console: bool = False
    log_dir: str = Field(default="logs", min_length=1)
    level: Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    rotation: str = Field(default="10 MB", min_length=1)
    retention: str = Field(default="7 days", min_length=1)


def _project_root() -> Path:
    """Retrouver le projet depuis le package ou le dossier courant."""
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for directory in (start, *start.parents):
            if (directory / "pyproject.toml").is_file() and (
                directory / "src" / "contextiq"
            ).is_dir():
                return directory
    return Path.cwd().resolve()


def load_logging_config(config_dir: str | Path | None = None) -> LoggingConfig:
    """Lire logging.yaml sans déclencher de log ni de dépendance vers Settings.

    Un dossier personnalisé peut omettre ce fichier : les valeurs par défaut
    s'appliquent alors, avec console=False. Un fichier présent mais invalide échoue.
    """
    configured = config_dir if config_dir is not None else os.getenv("CONTEXTIQ_CONFIG_DIR")
    directory = (
        Path(configured).expanduser().resolve()
        if configured is not None
        else _project_root() / "configs"
    )
    path = directory / "logging.yaml"
    if not path.is_file():
        return LoggingConfig()
    return LoggingConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def configure_logging(
    log_dir: str | Path | None = None,
    level: str | None = None,
    console: bool | None = None,
    *,
    config: LoggingConfig | None = None,
) -> Path:
    """Partager un sink fichier et une sortie console entre classes et notebooks.

    Les valeurs initiales viennent de configs/logging.yaml (console=False).
    Le dossier par défaut est logs/ à la racine du projet. CONTEXTIQ_LOG_DIR
    et CONTEXTIQ_LOG_LEVEL permettent de le configurer sans modifier le code.
    Les chemins relatifs sont résolus depuis la racine. Les appels sans argument
    conservent la configuration active ; un argument explicite la reconfigure.
    """
    global _sink_id, _console_sink_id, _log_path, _level, _rotation, _retention
    with _lock:
        if (
            _sink_id is not None
            and log_dir is None
            and level is None
            and console is None
            and config is None
        ):
            return _log_path
        configuration = config if config is not None else load_logging_config()
        directory = Path(
            log_dir or os.getenv("CONTEXTIQ_LOG_DIR", configuration.log_dir)
        ).expanduser()
        if not directory.is_absolute():
            directory = _project_root() / directory
        path = directory.resolve() / LOG_FILENAME
        selected_level = (level or os.getenv("CONTEXTIQ_LOG_LEVEL", configuration.level)).upper()
        selected_rotation, selected_retention = configuration.rotation, configuration.retention
        if (
            _sink_id is not None
            and console is not None
            and log_dir is None
            and level is None
            and config is None
        ):
            # Changer l'affichage seul conserve le fichier et son niveau actifs.
            path, selected_level = _log_path, _level
            selected_rotation, selected_retention = _rotation, _retention
        console_env = os.getenv("CONTEXTIQ_LOG_CONSOLE")
        if console is not None:
            selected_console = console
        elif console_env is not None:
            selected_console = console_env.lower() not in {"0", "false", "no"}
        else:
            selected_console = configuration.console
        base_logger.level(selected_level)  # Valider avant de remplacer le sink actif.
        if (
            _sink_id is not None
            and path == _log_path
            and selected_level == _level
            and selected_console == (_console_sink_id is not None)
            and selected_rotation == _rotation
            and selected_retention == _retention
        ):
            return path

        path.parent.mkdir(parents=True, exist_ok=True)
        new_sink = base_logger.add(
            path,
            level=selected_level,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS Z} | {level:<8} | {extra} | {message}",
            filter=lambda record: record["extra"].get("application") == "contextiq",
            encoding="utf-8",
            rotation=selected_rotation,
            retention=selected_retention,
            backtrace=False,
            diagnose=False,
            enqueue=False,
            watch=True,
        )
        if _sink_id is not None:
            base_logger.remove(_sink_id)
        # Retirer uniquement le sink console par défaut, pas ceux d'autres utilisateurs.
        with suppress(ValueError):
            base_logger.remove(0)
        if _console_sink_id is not None:
            base_logger.remove(_console_sink_id)
            _console_sink_id = None
        if selected_console:
            _console_sink_id = base_logger.add(
                # Jupyter remplace stderr entre les cellules : utiliser le flux courant.
                lambda message: sys.stderr.write(message),
                level=selected_level,
                format="{time:HH:mm:ss} | {level:<8} | {message} | {extra}",
                filter=lambda record: record["extra"].get("application") == "contextiq",
                colorize=False,
                backtrace=False,
                diagnose=False,
            )
        _sink_id, _log_path, _level = new_sink, path, selected_level
        _rotation, _retention = selected_rotation, selected_retention
        return path


def _result_metadata(result: object) -> dict:
    """Conserver uniquement des compteurs, scores et décisions connus du pipeline."""
    details = {}
    if isinstance(result, (list, tuple, dict)):
        details["result_count"] = len(result)
    for field in (
        "accepted",
        "answered",
        "ready",
        "threshold",
        "best_score",
        "top_k",
        "in_scope_count",
        "out_of_scope_count",
        "question_count",
        "answered_count",
        "success_rate",
        "precision",
        "recall",
        "mrr",
        "hit_rate",
        "false_refusals",
        "false_acceptances",
        "faithfulness_rate",
        "citation_validity_rate",
        "relevant_source_rate",
        "false_refusal_rate",
        "correct_refusal_rate",
        "answer_rate",
        "acceptance_rate",
        "false_acceptance_rate",
    ):
        value = getattr(result, field, None)
        if isinstance(value, (bool, int, float)):
            details[field] = value
    for field in (
        "documents",
        "questions",
        "sources",
        "results",
        "source_ids",
        "blocked_source_ids",
    ):
        value = getattr(result, field, None)
        if isinstance(value, (list, dict)):
            details[f"{field}_count"] = len(value)
    reason = getattr(result, "reason", None)
    if isinstance(reason, str) and reason in {
        "no_results",
        "no_positive_score",
        "below_threshold",
        "threshold_met",
        "ready",
        "abstention",
        "empty_context",
        "suspicious_question",
        "suspicious_context",
        "extractive",
    }:
        details["reason"] = reason
    return details


def log_operation(function: Callable[P, R]) -> Callable[P, R]:
    """Tracer début, fin, durée et échec sans journaliser les arguments ou les textes.

    Les exceptions sont relancées sans changer leur comportement. Seul leur type
    est écrit : leur message et leurs variables pourraient contenir des données.
    Une trace corrèle les opérations imbriquées ; la session identifie le processus.
    """

    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        configure_logging()
        token = None
        if _trace_id.get() is None:
            token = _trace_id.set(uuid4().hex[:12])
        start = perf_counter()
        try:
            with logger.contextualize(
                operation=f"{function.__module__}.{function.__qualname__}",
                trace_id=_trace_id.get(),
            ):
                parameters = {
                    name: value
                    for name, value in kwargs.items()
                    if name in {"top_k", "k1", "b", "local_files_only"}
                    and type(value) in (bool, int, float)
                }
                logger.bind(**parameters).info("Début de l’opération")
                try:
                    result = function(*args, **kwargs)
                except Exception as error:
                    logger.bind(
                        error_type=type(error).__name__,
                        duration_ms=round((perf_counter() - start) * 1000, 3),
                    ).error("Échec de l’opération")
                    raise
                details = parameters | _result_metadata(result)
                if function.__name__ == "__init__" and args:
                    documents = getattr(args[0], "documents", None)
                    if isinstance(documents, (list, dict)):
                        details["document_count"] = len(documents)
                suspicious = details.get("reason") in {"suspicious_question", "suspicious_context"}
                if function.__name__ == "is_suspicious":
                    details["suspicious"] = result
                    suspicious = result is True
                event_logger = logger.bind(
                    duration_ms=round((perf_counter() - start) * 1000, 3),
                    **details,
                )
                if suspicious:
                    event_logger.warning("Contenu suspect détecté ; contrôle de sécurité appliqué")
                else:
                    event_logger.info("Opération terminée")
                return result
        finally:
            if token is not None:
                _trace_id.reset(token)

    return wrapped
