"""Enregistrer une évaluation dans MLflow, sans exporter les textes métier."""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from contextiq.evaluation.answer_metrics import ExtractiveAnswerEvaluator
from contextiq.evaluation.evaluator import RetrievalEvaluator
from contextiq.generation.extractive_answer import ExtractiveAnswerGenerator
from contextiq.logging import log_operation
from contextiq.retrieval.base import Retriever
from contextiq.safety.abstention import AbstentionPolicy
from contextiq.schemas.models import Dataset
from contextiq.settings import Settings


class TrackedEvaluation(BaseModel):
    """Identifiant consultable dans MLflow et métriques agrégées enregistrées."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    metrics: dict[str, float]


def fingerprint(value: object) -> str:
    """Identifier un contenu sans enregistrer le texte qui a servi au calcul."""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class MLflowTracker:
    """Une exécution par méthode et K ; SQLite et artefacts restent locaux.

    Le client explicite évite de modifier le run actif ou l'URI globale d'un notebook.
    Aucun autolog ni enregistrement des questions, sources ou réponses n'est activé.
    """

    def __init__(self, settings: Settings) -> None:
        """Ouvrir le stockage défini par Settings.load() et retrouver l'expérience."""
        # Désactiver la télémétrie avant le premier import de MLflow.
        os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
        os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
        from mlflow import MlflowClient

        self.settings = settings.model_copy(deep=True)
        directory = Path(settings.monitoring.storage_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        self.tracking_uri = f"sqlite:///{directory / 'tracking.db'}"
        self.artifact_dir = directory / "artifacts"
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        experiment = self.client.get_experiment_by_name(settings.monitoring.experiment_name)
        if experiment is None:
            self.experiment_id = self.client.create_experiment(
                settings.monitoring.experiment_name,
                artifact_location=self.artifact_dir.as_uri(),
            )
        elif experiment.lifecycle_stage == "deleted":
            raise ValueError("L'expérience MLflow est supprimée : la restaurer ou changer son nom.")
        else:
            # Ne pas réutiliser par inadvertance une expérience pointant vers le réseau.
            if experiment.artifact_location != self.artifact_dir.as_uri():
                raise ValueError(
                    "L'expérience existante doit utiliser le dossier d'artefacts local."
                )
            self.experiment_id = experiment.experiment_id

    @log_operation
    def evaluate(
        self,
        method: Literal["TF-IDF", "BM25", "Embeddings"],
        retriever: Retriever,
        dataset: Dataset,
        *,
        top_k: int | None = None,
    ) -> TrackedEvaluation:
        """Évaluer les classements et les réponses, puis conserver leurs agrégats.

        Les annotations ne sont lues qu'après la recherche. Le temps mesuré couvre
        la boucle d'évaluation hors initialisation ; ce n'est pas un SLA de recherche.
        Les taux sans dénominateur restent absents, plutôt que remplacés par zéro.
        """
        from mlflow.entities import Metric, Param

        if method not in self.settings.retrieval.abstention_thresholds:
            raise ValueError("Méthode attendue : TF-IDF, BM25 ou Embeddings.")
        k = self.settings.retrieval.top_k if top_k is None else top_k
        if type(k) is not int or k < 1:
            raise ValueError("top_k doit être un entier strictement positif.")
        run = self.client.create_run(
            self.experiment_id,
            tags={
                "mlflow.runName": f"{method}-k{k}",
                "method": method,
                "study_status": "exploratory",
                "generator": "extractive",
                "python_version": platform.python_version(),
                "mlflow_version": version("mlflow"),
            },
        )
        run_id = run.info.run_id
        try:
            start = time.perf_counter()
            report = RetrievalEvaluator().evaluate(retriever, dataset.questions, top_k=k)
            policy = AbstentionPolicy(
                threshold=self.settings.retrieval.abstention_thresholds[method]
            )
            generator = ExtractiveAnswerGenerator()
            evaluator = ExtractiveAnswerEvaluator(dataset.documents)
            cases = []
            false_acceptances = 0
            # Réutiliser les recherches du rapport pour ne pas interroger deux fois le modèle.
            for item in report.questions:
                decision = policy.decide(item.results)
                false_acceptances += int(not item.question.gold_docs and decision.accepted)
                answer = generator.generate(item.question.q, decision)
                cases.append(evaluator.evaluate(answer, item.question))
            summary = evaluator.summarize(cases)
            metrics = {"evaluation_seconds": time.perf_counter() - start}
            if report.summary is not None:
                metrics.update(
                    {
                        f"retrieval_{key}": value
                        for key, value in report.summary.model_dump().items()
                    }
                )
            metrics.update(
                {
                    f"answer_{key}": float(value)
                    for key, value in summary.model_dump().items()
                    if value is not None
                }
            )
            metrics["abstention_false_acceptances"] = float(false_acceptances)
            if report.out_of_scope_count:
                metrics["abstention_false_acceptance_rate"] = (
                    false_acceptances / report.out_of_scope_count
                )
            configuration = self.settings.retrieval.model_dump(
                exclude={"embeddings": {"cache_dir"}}
            )
            parameters = {
                "method": method,
                "top_k": k,
                "threshold": policy.threshold,
                "documents_sha256": fingerprint([d.model_dump() for d in dataset.documents]),
                "questions_sha256": fingerprint([q.model_dump() for q in dataset.questions]),
                "configuration_sha256": fingerprint(configuration),
            }
            source = Path(__file__).resolve().parents[1]
            parameters["source_code_sha256"] = fingerprint(
                {
                    str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(source.rglob("*.py"))
                }
            )
            lock = source.parents[1] / "poetry.lock"
            if lock.exists():
                parameters["poetry_lock_sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest()
            if method == "BM25":
                parameters.update(self.settings.retrieval.bm25.model_dump())
            elif method == "Embeddings":
                parameters.update(
                    {
                        "model_name": getattr(retriever, "model_name", "unknown"),
                        "revision": getattr(retriever, "revision", "unknown"),
                    }
                )
            timestamp = int(time.time() * 1000)
            self.client.log_batch(
                run_id,
                metrics=[Metric(key, value, timestamp, 0) for key, value in metrics.items()],
                params=[Param(key, str(value)) for key, value in parameters.items()],
            )
            self.client.log_dict(run_id, configuration, "retrieval_config.json")
            self.client.log_dict(run_id, metrics, "metrics.json")
            self.client.log_dict(
                run_id,
                {
                    name: version(name)
                    for name in (
                        "mlflow",
                        "pydantic",
                        "numpy",
                        "scikit-learn",
                        "sentence-transformers",
                        "torch",
                    )
                },
                "versions.json",
            )
            self.client.set_terminated(run_id, status="FINISHED")
        except Exception:
            self.client.set_terminated(run_id, status="FAILED")
            raise
        return TrackedEvaluation(run_id=run_id, metrics=metrics)


def main() -> None:
    """Ouvrir l'interface sur la boucle locale, avec la même base que les notebooks."""
    settings = Settings.load()
    tracker = MLflowTracker(settings)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--backend-store-uri",
            tracker.tracking_uri,
            "--default-artifact-root",
            tracker.artifact_dir.as_uri(),
            "--no-serve-artifacts",
            "--host",
            "127.0.0.1",
            "--port",
            str(settings.monitoring.port),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
