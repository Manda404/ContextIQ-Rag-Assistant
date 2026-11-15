"""Contrôler le stockage réel local, ses métriques et l'absence de textes métier."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from contextiq.monitoring.mlflow_tracking import MLflowTracker
from contextiq.retrieval.bm25 import BM25Retriever
from contextiq.schemas.models import Dataset, Document, EvaluationQuestion
from contextiq.settings import Settings


class TestMLflowTracker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.settings = Settings.load()
        cls.settings.monitoring.storage_dir = cls.directory.name
        cls.settings.monitoring.experiment_name = "unit-tests"
        cls.settings.retrieval.abstention_thresholds["BM25"] = 0.0
        cls.tracker = MLflowTracker(cls.settings)
        cls.dataset = Dataset(
            documents=[Document(id="PRIVATE-ID", text="PRIVATE-TEXT remboursement exceptionnel")],
            questions=[
                EvaluationQuestion(q="PRIVATE-QUESTION remboursement", gold_docs=["PRIVATE-ID"]),
                EvaluationQuestion(q="astronomie", gold_docs=[]),
            ],
        )

    def test_real_run_metrics_artifacts_and_privacy(self):
        tracker = self.tracker
        result = tracker.evaluate(
            "BM25", BM25Retriever(self.dataset.documents), self.dataset, top_k=1
        )
        run = tracker.client.get_run(result.run_id)
        self.assertEqual(run.info.status, "FINISHED")
        self.assertEqual(run.data.metrics["answer_success_rate"], 1.0)
        self.assertEqual(run.data.metrics["retrieval_recall"], 1.0)
        self.assertEqual(run.data.metrics["abstention_false_acceptance_rate"], 0.0)
        self.assertEqual(run.data.params["top_k"], "1")
        self.assertIn("source_code_sha256", run.data.params)
        self.assertIn("documents_sha256", run.data.params)
        combined = json.dumps({"params": run.data.params, "tags": run.data.tags})
        for artifact in tracker.client.list_artifacts(result.run_id):
            path = tracker.client.download_artifacts(result.run_id, artifact.path)
            combined += Path(path).read_text()
        for secret in ("PRIVATE-ID", "PRIVATE-TEXT", "PRIVATE-QUESTION"):
            self.assertNotIn(secret, combined)

    def test_failed_evaluation_marks_run_failed(self):
        with patch.object(self.tracker.client, "log_batch", side_effect=RuntimeError("test")):
            with self.assertRaises(RuntimeError):
                self.tracker.evaluate("BM25", BM25Retriever(self.dataset.documents), self.dataset)
        runs = self.tracker.client.search_runs([self.tracker.experiment_id])
        self.assertTrue(any(run.info.status == "FAILED" for run in runs))

    def test_missing_group_metrics_are_not_replaced_with_zero(self):
        dataset = self.dataset.model_copy(deep=True)
        dataset.questions = dataset.questions[:1]
        result = self.tracker.evaluate("BM25", BM25Retriever(dataset.documents), dataset)
        self.assertNotIn("abstention_false_acceptance_rate", result.metrics)
        self.assertNotIn("answer_correct_refusal_rate", result.metrics)

    def test_reopening_tracker_keeps_experiment(self):
        other = MLflowTracker(self.settings)
        self.assertEqual(self.tracker.experiment_id, other.experiment_id)

    def test_invalid_method_and_k(self):
        engine = BM25Retriever(self.dataset.documents)
        with self.assertRaises(ValueError):
            self.tracker.evaluate("unknown", engine, self.dataset)
        with self.assertRaises(ValueError):
            self.tracker.evaluate("BM25", engine, self.dataset, top_k=0)

    def test_monitoring_yaml_is_loaded_and_portable(self):
        settings = Settings.load()
        self.assertTrue(Path(settings.monitoring.storage_dir).is_absolute())
        self.assertEqual(settings.monitoring.experiment_name, "contextiq-rag-assistant")
        self.assertEqual(settings.monitoring.port, 5000)
