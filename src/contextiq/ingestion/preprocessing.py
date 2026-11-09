"""Prétraitement lexical commun aux documents et aux questions."""

import re
import unicodedata

from contextiq.logging import log_operation


class TextPreprocessor:
    """Préparer la recherche sans modifier les sources originales."""

    @log_operation
    def tokenize(self, text: str) -> list[str]:
        """Passer en minuscules, retirer les accents et séparer les mots.

        On conserve les nombres, le signe % et les négations. Aucun stemming
        ni retrait de mots fréquents : leur effet sera évalué séparément.
        """
        normalized = unicodedata.normalize("NFKD", text.lower())
        normalized = "".join(
            character for character in normalized if not unicodedata.combining(character)
        )
        return re.findall(r"[^\W_]+|%", normalized)
