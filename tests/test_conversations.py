"""Vérifier les échanges persistants, les erreurs et les revues humaines."""

import json
import os
import sqlite3
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4

from contextiq.monitoring.conversations import ConversationFeedback, ConversationStore
from contextiq.schemas.models import Dataset, Document, EvaluationQuestion
from contextiq.settings import Settings
from contextiq.ui.service import AssistantSession


class TestConversations(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        env = patch.dict(
            os.environ, {"CONTEXTIQ_LOG_DIR": str(self.root), "CONTEXTIQ_LOG_CONSOLE": "0"}
        )
        env.start()
        self.addCleanup(env.stop)
        self.settings = Settings.load()
        self.dataset = Dataset(
            documents=[Document(id="A", text="Le support est ouvert le lundi.")],
            questions=[EvaluationQuestion(q="support", gold_docs=["A"])],
        )
        self.session = AssistantSession(self.dataset, self.settings)

    def test_question_answer_and_original_sources_survive_new_session(self):
        turn = self.session.ask("Quand le support est-il ouvert ?", "TF-IDF", 1, 0)
        store = ConversationStore(self.root / "conversations.db")
        record = store.recent()[0]
        self.assertEqual(record.id, turn.interaction_id)
        self.assertEqual(record.question, turn.answer.question)
        self.assertEqual(record.answer, turn.answer.text)
        self.assertEqual(record.cited_source_ids, ["A"])
        self.assertEqual(record.retrieved_sources[0].document.text, self.dataset.documents[0].text)
        self.assertEqual(record.status, "answered")
        self.assertTrue(record.created_at.tzinfo)
        other = AssistantSession(self.dataset, self.settings)
        self.assertNotEqual(other.session_id, self.session.session_id)
        self.assertEqual(other.conversations.summarize().total, 1)

    def test_refusal_and_technical_error_are_persisted(self):
        self.session.ask("astronomie", "TF-IDF", 1, 0)
        with patch.object(self.session, "get_retriever", side_effect=OSError("secret-error-text")):
            with self.assertRaises(OSError):
                self.session.ask("support", "TF-IDF", 1, 0)
        records = self.session.conversations.recent()
        self.assertEqual(records[0].status, "error")
        self.assertEqual(records[0].error_type, "OSError")
        self.assertIsNone(records[0].answer)
        self.assertNotIn("secret-error-text", records[0].model_dump_json())
        self.assertEqual(records[1].status, "refused")
        summary = self.session.conversations.summarize()
        self.assertEqual((summary.total, summary.refused, summary.errors), (2, 1, 1))
        self.assertEqual(summary.error_rate, 0.5)
        self.assertIsNone(summary.flagged_review_rate)

    def test_feedback_update_does_not_duplicate_reviewed_count(self):
        turn = self.session.ask("support", "TF-IDF", 1, 0)
        store = self.session.conversations
        store.review(
            ConversationFeedback(
                interaction_id=turn.interaction_id, verdict="irrelevant", comment="Hors sujet"
            )
        )
        summary = store.summarize()
        self.assertEqual(
            (summary.reviewed, summary.flagged, summary.flagged_review_rate), (1, 1, 1.0)
        )
        store.review(ConversationFeedback(interaction_id=turn.interaction_id, verdict="helpful"))
        self.assertEqual(store.get_feedback(turn.interaction_id).verdict, "helpful")
        summary = store.summarize()
        self.assertEqual(
            (summary.reviewed, summary.flagged, summary.flagged_review_rate), (1, 0, 0.0)
        )
        with self.assertRaises(sqlite3.IntegrityError):
            store.review(ConversationFeedback(interaction_id="missing", verdict="helpful"))

    def test_disabled_journal_does_not_store_questions(self):
        self.settings.monitoring.log_conversations = False
        session = AssistantSession(self.dataset, self.settings)
        turn = session.ask("support", "TF-IDF", 1, 0)
        self.assertTrue(turn.answer.answered)
        self.assertIsNone(turn.interaction_id)
        self.assertIsNone(session.conversations)
        self.assertEqual(self.session.conversations.summarize().total, 0)

    def test_technical_log_contains_reference_but_not_question(self):
        question = "PRIVATE-QUESTION support"
        turn = self.session.ask(question, "TF-IDF", 1, 0)
        technical_log = (self.root / "contextiq-rag-assistant.log").read_text()
        self.assertIn(turn.interaction_id, technical_log)
        self.assertNotIn(question, technical_log)
        self.assertNotIn(self.dataset.documents[0].text, technical_log)
        self.assertIn(
            question,
            json.dumps([r.model_dump(mode="json") for r in self.session.conversations.recent()]),
        )

    def test_concurrent_writes_do_not_lose_interactions(self):
        self.session.ask("support", "TF-IDF", 1, 0)
        store = self.session.conversations
        record = store.recent()[0]
        records = [record.model_copy(update={"id": uuid4().hex}) for _ in range(12)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(store.append, records))
        self.assertEqual(store.summarize().total, 13)
