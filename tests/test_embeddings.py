"""Vérifier le classement avec des vecteurs connus, sans télécharger de modèle."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from contextiq.retrieval.embeddings import EmbeddingRetriever
from contextiq.schemas.models import Document


class TestEmbeddingRetriever(unittest.TestCase):
    def setUp(self):
        loader = patch("contextiq.retrieval.embeddings.SentenceTransformer")
        self.load_model = loader.start()
        self.addCleanup(loader.stop)
        self.model = MagicMock()
        self.load_model.return_value = self.model
        self.model.max_seq_length = 128
        self.model.tokenizer.side_effect = lambda texts, **kwargs: {
            "input_ids": [list(range(len(text.split()) + 2)) for text in texts]
        }
        self.model.encode_document.return_value = np.eye(3)
        self.model.encode_query.return_value = np.array([[0.6, 0.8, 0.0]])
        self.documents = [
            Document(id="A", text="Résiliation annuelle."),
            Document(id="B", text="Ligne suspendue."),
            Document(id="C", text="Préavis mensuel."),
        ]

    def test_cosine_ranking_positive_scores_and_top_k(self):
        retriever = EmbeddingRetriever(self.documents)
        results = retriever.retrieve("Ma ligne est coupée", top_k=10)
        self.assertEqual([r.document.id for r in results], ["B", "A"])
        self.assertAlmostEqual(results[0].score, 0.8)
        self.assertAlmostEqual(results[1].score, 0.6)
        self.assertEqual(len(retriever.retrieve("Question", top_k=1)), 1)

    def test_nonpositive_cosines_are_excluded(self):
        self.model.encode_query.return_value = np.array([[-1.0, 0.0, 0.0]])
        self.assertEqual(EmbeddingRetriever(self.documents).retrieve("Question"), [])

    def test_ties_use_identifiers(self):
        self.model.encode_document.return_value = np.array([[1.0, 0.0], [1.0, 0.0]])
        self.model.encode_query.return_value = np.array([[1.0, 0.0]])
        retriever = EmbeddingRetriever([self.documents[1], self.documents[0]])
        self.assertEqual([r.document.id for r in retriever.retrieve("Question")], ["A", "B"])

    def test_original_texts_and_normalized_vectors(self):
        retriever = EmbeddingRetriever(self.documents)
        self.model.encode_document.assert_called_once_with(
            [document.text for document in self.documents],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        retriever.retrieve("Ma ligne est coupée ?")
        self.model.encode_query.assert_called_once_with(
            ["Ma ligne est coupée ?"],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def test_query_does_not_reencode_documents(self):
        retriever = EmbeddingRetriever(self.documents)
        retriever.retrieve("Question 1")
        retriever.retrieve("Question 2")
        self.assertEqual(self.model.encode_document.call_count, 1)
        self.assertEqual(self.model.encode_query.call_count, 2)

    def test_offline_pinned_model_and_configurable_cache(self):
        EmbeddingRetriever(self.documents, cache_dir=Path("cache"))
        self.load_model.assert_called_once_with(
            EmbeddingRetriever.MODEL_NAME,
            revision=EmbeddingRetriever.MODEL_REVISION,
            device="cpu",
            local_files_only=True,
            cache_folder="cache",
            trust_remote_code=False,
            token=False,
        )

    def test_sources_are_snapshot_copies(self):
        retriever = EmbeddingRetriever(self.documents)
        self.documents[1].text = "Modifié"
        self.assertEqual(retriever.retrieve("Question")[0].document.text, "Ligne suspendue.")

    def test_configured_model_and_revision_are_forwarded(self):
        retriever = EmbeddingRetriever(
            self.documents,
            model_name="example/model",
            revision="fixed-revision",
        )
        self.assertEqual(retriever.model_name, "example/model")
        self.assertEqual(retriever.revision, "fixed-revision")
        self.assertEqual(self.load_model.call_args.args[0], "example/model")
        self.assertEqual(self.load_model.call_args.kwargs["revision"], "fixed-revision")

    def test_invalid_corpus_is_rejected_before_loading_model(self):
        for documents in ([], [self.documents[0], self.documents[0]]):
            with self.subTest(documents=documents), self.assertRaises(ValueError):
                EmbeddingRetriever(documents)
        self.load_model.assert_not_called()

    def test_invalid_query_and_k(self):
        retriever = EmbeddingRetriever(self.documents)
        for k in (0, -1, 1.5, True):
            with self.subTest(k=k), self.assertRaises(ValueError):
                retriever.retrieve("Question", top_k=k)
        with self.assertRaises(ValueError):
            retriever.retrieve(" ")

    def test_long_documents_and_queries_are_not_silently_truncated(self):
        long_text = "mot " * 130
        with self.assertRaisesRegex(ValueError, "Texte trop long"):
            EmbeddingRetriever([Document(id="A", text=long_text)])
        retriever = EmbeddingRetriever(self.documents)
        with self.assertRaisesRegex(ValueError, "Texte trop long"):
            retriever.retrieve(long_text)
