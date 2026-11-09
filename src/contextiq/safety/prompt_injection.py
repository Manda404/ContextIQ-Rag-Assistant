"""Filtre heuristique de quelques injections connues, sans garantie exhaustive."""

import re
import unicodedata

from contextiq.logging import log_operation


class PromptInjectionDetector:
    """Repérer des consignes suspectes avant de construire le contexte.

    Cette liste peut produire des faux positifs et manquer des reformulations.
    Elle ne remplace ni les contrôles d'accès ni les tests d'un éventuel LLM.
    """

    PATTERNS = (
        r"\b(?:ignore|ignorez|ignorer|oublie|oubliez|forget)\b.{0,100}\binstructions?\b",
        r"\b(?:revele|divulgue|donne|affiche|reveal|disclose)\b.{0,80}"
        r"\b(?:donnees|informations|data)\b.{0,60}\b(?:client|clients|personnelles|personal)\b",
        r"</?(?:system|assistant|developer)>|\[/?inst\]|<\|(?:im_start|system|assistant|developer)\|>",
    )

    @log_operation
    def is_suspicious(self, text: str) -> bool:
        """Tester une copie normalisée, sans modifier le texte de la source."""
        normalized = unicodedata.normalize("NFKD", text.lower())
        normalized = "".join(
            character for character in normalized if not unicodedata.combining(character)
        )
        normalized = " ".join(normalized.split())
        return any(re.search(pattern, normalized) for pattern in self.PATTERNS)
