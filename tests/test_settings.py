"""Vérifier les YAML, leur portabilité et la propagation des paramètres."""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import yaml
from pydantic import ValidationError

from contextiq.retrieval.bm25 import BM25Retriever
from contextiq.safety.abstention import AbstentionPolicy
from contextiq.schemas.models import Document
from contextiq.settings import Settings


class TestSettings(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.config_dir = self.root / "configs"
        self.config_dir.mkdir()
        # Utiliser les fichiers du projet comme point de départ, sans lire les données.
        project_configs = Path(__file__).resolve().parents[1] / "configs"
        self.retrieval = yaml.safe_load((project_configs / "retrieval.yaml").read_text())
        self.evaluation = yaml.safe_load((project_configs / "evaluation.yaml").read_text())
        self.write_configs()

    def write_configs(self):
        """Enregistrer un jeu de configuration modifiable dans un dossier temporaire."""
        for filename, value in (
            ("retrieval.yaml", self.retrieval),
            ("evaluation.yaml", self.evaluation),
        ):
            (self.config_dir / filename).write_text(yaml.safe_dump(value), encoding="utf-8")

    def test_yaml_values_are_used_by_components(self):
        self.retrieval["bm25"] = {"k1": 0.9, "b": 0.2}
        self.retrieval["abstention_thresholds"]["BM25"] = 4.0
        self.evaluation["k_values"] = [1, 2]
        self.evaluation["inspection_k"] = 2
        self.write_configs()
        settings = Settings.load(self.config_dir)
        retriever = BM25Retriever(
            [Document(id="A", text="préavis")],
            **settings.retrieval.bm25.model_dump(),
        )
        policy = AbstentionPolicy(threshold=settings.retrieval.abstention_thresholds["BM25"])
        self.assertEqual(retriever.parameters.k1, 0.9)
        self.assertEqual(retriever.parameters.b, 0.2)
        self.assertEqual(policy.threshold, 4)
        self.assertEqual(settings.evaluation.k_values, [1, 2])

    def test_environment_and_explicit_directory_priority(self):
        with patch.dict(os.environ, {"CONTEXTIQ_CONFIG_DIR": str(self.config_dir)}):
            self.assertEqual(Settings.load().retrieval.top_k, self.retrieval["top_k"])
            with self.assertRaises(FileNotFoundError):
                Settings.load(self.root / "missing")

    def test_discovery_from_moved_project_and_notebooks(self):
        (self.root / "pyproject.toml").touch()
        (self.root / "src/contextiq").mkdir(parents=True)
        (self.root / "notebooks").mkdir()
        with patch.dict(os.environ):
            os.environ.pop("CONTEXTIQ_CONFIG_DIR", None)
            for cwd in (self.root, self.root / "notebooks"):
                with patch("contextiq.settings.Path.cwd", return_value=cwd):
                    self.assertEqual(Settings.load().retrieval.top_k, self.retrieval["top_k"])

    def test_cache_is_resolved_relative_to_yaml(self):
        self.retrieval["embeddings"]["cache_dir"] = "../model-cache"
        self.write_configs()
        self.assertEqual(
            Settings.load(self.config_dir).retrieval.embeddings.cache_dir,
            str(self.root / "model-cache"),
        )

    def test_invalid_retrieval_values_are_rejected(self):
        for key, value in (("top_k", 0), ("top_k", True), ("typo", 3)):
            original = self.retrieval.copy()
            self.retrieval[key] = value
            self.write_configs()
            with self.subTest(key=key, value=value), self.assertRaises(ValidationError):
                Settings.load(self.config_dir)
            self.retrieval = original

    def test_missing_or_unknown_method_is_rejected(self):
        self.retrieval["abstention_thresholds"] = {"BM25": 2.0}
        self.write_configs()
        with self.assertRaises(ValidationError):
            Settings.load(self.config_dir)

    def test_logging_yaml_is_available_in_settings(self):
        (self.config_dir / "logging.yaml").write_text(
            "console: false\nlevel: DEBUG\n", encoding="utf-8"
        )
        settings = Settings.load(self.config_dir)
        self.assertFalse(settings.logging.console)
        self.assertEqual(settings.logging.level, "DEBUG")

    def test_invalid_evaluation_values_are_rejected(self):
        for updates in (
            {"k_values": []},
            {"k_values": [1, 1]},
            {"inspection_k": 99},
            {"threshold_grids": {"BM25": [1.0]}},
            {"threshold_grids": {"BM25": [], "TF-IDF": [0.1], "Embeddings": [0.1]}},
            {"threshold_grids": {"BM25": [-1.0], "TF-IDF": [0.1], "Embeddings": [0.1]}},
        ):
            original = self.evaluation.copy()
            self.evaluation.update(updates)
            self.write_configs()
            with self.subTest(updates=updates), self.assertRaises(ValidationError):
                Settings.load(self.config_dir)
            self.evaluation = original

    def test_missing_file_and_malformed_yaml_are_explicit(self):
        path = self.config_dir / "retrieval.yaml"
        path.write_text("[invalid", encoding="utf-8")
        with self.assertRaises(yaml.YAMLError):
            Settings.load(self.config_dir)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            Settings.load(self.config_dir)
