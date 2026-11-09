"""Vérifier le comportement du retriever avec une base fictive."""

import unittest

from contextiq.retrieval.tfidf import TfidfRetriever
from contextiq.schemas.models import Document


class TestTfidfRetriever(unittest.TestCase):
    def setUp(self):
        self.documents = [
            Document(id="A", text="Résiliation annuelle : frais plafonnés à 150 euros."),
            Document(id="B", text="Préavis mensuel de 10 jours."),
            Document(id="C", text="Une remise est possible sans retard de paiement."),
        ]
        self.retriever = TfidfRetriever(self.documents)

    def test_matching_document_and_original_source(self):
        result = self.retriever.retrieve("RESILIATION annuelle", top_k=1)[0]
        self.assertEqual(result.document.id, "A")
        self.assertEqual(result.document.text, self.documents[0].text)
        self.assertGreater(result.score, 0)

    def test_unknown_words_and_punctuation_return_no_result(self):
        for question in ("astronomie galaxie", "... !"):
            with self.subTest(question=question):
                self.assertEqual(self.retriever.retrieve(question), [])

    def test_query_does_not_modify_vocabulary(self):
        before = dict(self.retriever.vectorizer.vocabulary_)
        self.retriever.retrieve("astronomie annuelle")
        self.assertEqual(self.retriever.vectorizer.vocabulary_, before)

    def test_positive_scores_sorted_and_top_k_bounded(self):
        retriever = TfidfRetriever(
            [
                Document(id="A", text="préavis mensuel"),
                Document(id="B", text="préavis annuel frais"),
            ]
        )
        results = retriever.retrieve("préavis mensuel", top_k=10)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].document.id, "A")
        self.assertGreater(results[0].score, results[1].score)
        self.assertEqual(len(retriever.retrieve("préavis", top_k=1)), 1)

    def test_ties_use_document_id(self):
        retriever = TfidfRetriever(
            [
                Document(id="B", text="même texte"),
                Document(id="A", text="même texte"),
            ]
        )
        self.assertEqual([r.document.id for r in retriever.retrieve("texte")], ["A", "B"])

    def test_input_changes_do_not_desynchronize_index(self):
        self.documents[0].text = "Texte remplacé"
        self.assertIn("Résiliation", self.retriever.retrieve("résiliation")[0].document.text)

    def test_invalid_arguments_are_rejected(self):
        for top_k in (0, -1, 1.5, True):
            with self.subTest(top_k=top_k), self.assertRaises(ValueError):
                self.retriever.retrieve("préavis", top_k=top_k)
        with self.assertRaises(ValueError):
            self.retriever.retrieve("   ")

    def test_empty_or_duplicate_corpus_is_rejected(self):
        for documents in ([], [self.documents[0], self.documents[0]]):
            with self.subTest(documents=documents), self.assertRaises(ValueError):
                TfidfRetriever(documents)
