"""Tests BM25 sur des textes fictifs et des scores calculables à la main."""

import unittest
from math import log

from pydantic import ValidationError

from contextiq.retrieval.bm25 import BM25Retriever
from contextiq.schemas.models import Document


class TestBM25Retriever(unittest.TestCase):
    def test_formula_and_term_frequency_saturation(self):
        retriever = BM25Retriever(
            [
                Document(id="A", text="préavis préavis"),
                Document(id="B", text="préavis"),
                Document(id="C", text="remise"),
            ],
            k1=1.5,
            b=0,
        )
        results = retriever.retrieve("PREAVIS")
        idf = log(1 + (3 - 2 + 0.5) / (2 + 0.5))
        self.assertEqual([r.document.id for r in results], ["A", "B"])
        self.assertAlmostEqual(results[0].score, idf * 2 * 2.5 / (2 + 1.5))
        self.assertAlmostEqual(results[1].score, idf)
        self.assertLess(results[0].score, 2 * results[1].score)

    def test_length_normalization_favors_shorter_document(self):
        results = BM25Retriever(
            [
                Document(id="A", text="préavis"),
                Document(id="B", text="préavis contrat mensuel conditions"),
            ]
        ).retrieve("préavis")
        self.assertEqual(results[0].document.id, "A")
        self.assertGreater(results[0].score, results[1].score)

    def test_unknown_query_and_empty_document(self):
        retriever = BM25Retriever(
            [
                Document(id="A", text="..."),
                Document(id="B", text="préavis"),
            ]
        )
        self.assertEqual(retriever.retrieve("galaxie"), [])
        self.assertEqual(retriever.retrieve("!!!"), [])
        self.assertEqual(len(retriever.retrieve("préavis", top_k=10)), 1)

    def test_ties_and_repeated_query_terms(self):
        retriever = BM25Retriever(
            [
                Document(id="B", text="préavis"),
                Document(id="A", text="préavis"),
            ]
        )
        self.assertEqual([r.document.id for r in retriever.retrieve("préavis")], ["A", "B"])
        self.assertEqual(retriever.retrieve("préavis"), retriever.retrieve("préavis préavis"))
        self.assertEqual(len(retriever.retrieve("préavis", top_k=1)), 1)

    def test_original_sources_are_preserved(self):
        document = Document(id="A", text="PRÉAVIS : 10 jours !")
        retriever = BM25Retriever([document])
        document.text = "Modifié"
        self.assertEqual(retriever.retrieve("preavis")[0].document.text, "PRÉAVIS : 10 jours !")

    def test_invalid_parameters(self):
        for parameters in ({"k1": 0}, {"k1": float("nan")}, {"b": -1}, {"b": 1.1}):
            with self.subTest(parameters=parameters), self.assertRaises(ValidationError):
                BM25Retriever([Document(id="A", text="préavis")], **parameters)

    def test_invalid_corpus(self):
        document = Document(id="A", text="préavis")
        for documents in ([], [document, document], [Document(id="A", text="...")]):
            with self.subTest(documents=documents), self.assertRaises(ValueError):
                BM25Retriever(documents)

    def test_invalid_query_or_k(self):
        retriever = BM25Retriever([Document(id="A", text="préavis")])
        for k in (0, -1, 1.5, True):
            with self.subTest(k=k), self.assertRaises(ValueError):
                retriever.retrieve("préavis", top_k=k)
        with self.assertRaises(ValueError):
            retriever.retrieve(" ")
