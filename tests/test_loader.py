"""Vérifier l'ingestion avec des données fictives dans un dossier temporaire."""

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pydantic import ValidationError

from contextiq.ingestion.loader import DataLoader
from contextiq.schemas.models import Dataset, Document, EvaluationQuestion


class TestDataLoader(unittest.TestCase):
    """Protéger les annotations et la portabilité des chemins."""

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data_dir = Path(self.temp.name).resolve()
        self.documents = [{"id": "D1", "text": "Préavis de 10 jours, sans pénalité."}]
        self.questions = [
            {"q": "Quel préavis ?", "gold_docs": ["D1"]},
            {"q": "Quelle météo ?", "gold_docs": []},
        ]
        self.write_json("knowledge_base.json", self.documents)
        self.write_json("eval_set.json", self.questions)

    def write_json(self, name, data):
        """Créer un fichier de test sans dépendre des données confidentielles."""
        (self.data_dir / name).write_text(json.dumps(data), encoding="utf-8")

    def test_load_preserves_sources_and_out_of_scope(self):
        dataset = DataLoader(self.data_dir).load()
        self.assertEqual(dataset.documents[0].text, self.documents[0]["text"])
        self.assertEqual(dataset.questions[1].gold_docs, [])

    def test_environment_and_explicit_path_priority(self):
        with patch.dict(os.environ, {"CONTEXTIQ_DATA_DIR": str(self.data_dir)}):
            self.assertEqual(len(DataLoader().load().documents), 1)
            self.assertEqual(DataLoader(self.data_dir / "other").data_dir, self.data_dir / "other")

    def test_discovery_after_project_is_moved(self):
        (self.data_dir / "pyproject.toml").touch()
        (self.data_dir / "src/contextiq").mkdir(parents=True)
        notebook_dir = self.data_dir / "notebooks"
        notebook_dir.mkdir()
        with patch.dict(os.environ):
            os.environ.pop("CONTEXTIQ_DATA_DIR", None)
            for current in (self.data_dir, notebook_dir):
                with patch("contextiq.ingestion.loader.Path.cwd", return_value=current):
                    self.assertEqual(DataLoader().data_dir, self.data_dir / "data/raw")

    def test_invalid_document_fields_are_rejected(self):
        for item in (
            {"id": "D1"},
            {"id": 1, "text": "Texte"},
            {"id": "D1", "text": "  "},
            {"id": "D1", "text": "Texte", "unexpected": True},
        ):
            with self.subTest(item=item), self.assertRaises(ValidationError):
                Document.model_validate(item)

    def test_missing_annotation_is_not_out_of_scope(self):
        with self.assertRaises(ValidationError):
            EvaluationQuestion(q="Quel préavis ?")

    def test_duplicate_documents_are_rejected(self):
        self.write_json("knowledge_base.json", self.documents * 2)
        with self.assertRaisesRegex(ValidationError, "uniques"):
            DataLoader(self.data_dir).load()

    def test_unknown_and_duplicate_sources_are_rejected(self):
        for sources in (["D404"], ["D1", "D1"]):
            with self.subTest(sources=sources):
                self.write_json("eval_set.json", [{"q": "Question", "gold_docs": sources}])
                with self.assertRaises(ValidationError):
                    DataLoader(self.data_dir).load()

    def test_empty_datasets_are_rejected(self):
        with self.assertRaises(ValidationError):
            Dataset(documents=[], questions=[])

    def test_invalid_json_and_wrong_root_are_rejected(self):
        for content in ("not json", "{}"):
            (self.data_dir / "knowledge_base.json").write_text(content, encoding="utf-8")
            with self.subTest(content=content), self.assertRaises(ValidationError):
                DataLoader(self.data_dir).load()

    def test_missing_file_is_explicit(self):
        (self.data_dir / "eval_set.json").unlink()
        with self.assertRaises(FileNotFoundError):
            DataLoader(self.data_dir).load()
