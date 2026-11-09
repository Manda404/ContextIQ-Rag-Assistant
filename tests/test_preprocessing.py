"""Protéger les éléments importants des textes contractuels."""

import unittest

from contextiq.ingestion.preprocessing import TextPreprocessor


class TestTextPreprocessor(unittest.TestCase):
    def test_accents_case_and_apostrophes(self):
        tokens = TextPreprocessor().tokenize("RÉSILIATION d’un contrat, préavis !")
        self.assertEqual(tokens, ["resiliation", "d", "un", "contrat", "preavis"])

    def test_numbers_percentages_and_negation_are_preserved(self):
        tokens = TextPreprocessor().tokenize("Ne pas dépasser 150 euros, sans remise de 30 %.")
        for expected in ("ne", "pas", "150", "sans", "30", "%"):
            self.assertIn(expected, tokens)

    def test_punctuation_only_has_no_tokens(self):
        self.assertEqual(TextPreprocessor().tokenize("... !"), [])
