"""Tests for telegram_ingestor — config, message classification, polling."""

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from neuroflow_core.telegram_ingestor import (
    IngestorConfig,
    config_from_env,
    TelegramIngestor,
    UserStore,
)


# ======================================================================
# IngestorConfig
# ======================================================================


class TestIngestorConfig:
    def test_defaults(self):
        config = IngestorConfig(bot_token="123:abc")
        assert config.bot_token == "123:abc"
        assert config.api_base == "https://api.telegram.org"
        assert config.poll_interval_s == 5
        assert config.http_port == 8888
        assert config.db_path == "/tmp/telegram_ingestor.db"
        assert config.cold_days == 7
        assert config.churn_days == 30
        assert config.allowed_chat_ids is None

    def test_custom_values(self):
        config = IngestorConfig(
            bot_token="custom:token",
            api_base="https://custom.api",
            poll_interval_s=10,
            http_port=9090,
            db_path="/tmp/custom.db",
            cold_days=14,
            churn_days=60,
            allowed_chat_ids=[-100123, -100456],
        )
        assert config.bot_token == "custom:token"
        assert config.api_base == "https://custom.api"
        assert config.poll_interval_s == 10
        assert config.http_port == 9090
        assert config.db_path == "/tmp/custom.db"
        assert config.cold_days == 14
        assert config.churn_days == 60
        assert config.allowed_chat_ids == [-100123, -100456]


# ======================================================================
# config_from_env
# ======================================================================


class TestConfigFromEnv:
    def test_from_env_var(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "env:token123")
        config = config_from_env()
        assert config.bot_token == "env:token123"

    def test_fallback_to_env_file(self, monkeypatch, tmp_path):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        env_path = tmp_path / ".env"
        env_path.write_text("OTHER_VAR=1\nTELEGRAM_BOT_TOKEN=file:token456\nSOMETHING=else\n")

        # Patch os.path.exists to return True for the env path, and patch builtins.open
        original_exists = os.path.exists

        def mock_exists(p):
            if p == "/opt/data/.env":
                return True
            return original_exists(p)

        monkeypatch.setattr("neuroflow_core.telegram_ingestor.os.path.exists", mock_exists)
        # Patch builtins.open to redirect to our temp file
        import builtins
        original_open = builtins.open

        def mock_open(file, *args, **kwargs):
            if file == "/opt/data/.env":
                return original_open(str(env_path), *args, **kwargs)
            return original_open(file, *args, **kwargs)

        monkeypatch.setattr("builtins.open", mock_open)

        config = config_from_env()
        assert config.bot_token == "file:token456"

    def test_empty_when_no_token(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.setattr("neuroflow_core.telegram_ingestor.os.path.exists", lambda p: False)
        config = config_from_env()
        assert config.bot_token == ""


# ======================================================================
# UserStore — SQLite backend
# ======================================================================


class TestUserStore:
    @pytest.fixture
    def store(self, tmp_path) -> UserStore:
        return UserStore(db_path=str(tmp_path / "test_store.db"))

    def test_upsert_and_get_user(self, store):
        store.upsert_user(user_id=1, state="active", username="alice")
        user = store.get_user(1)
        assert user is not None
        assert user["state"] == "active"
        assert user["username"] == "alice"

    def test_upsert_updates_existing(self, store):
        store.upsert_user(user_id=1, state="lead", username="alice")
        store.upsert_user(user_id=1, state="hot", message_count=5)
        user = store.get_user(1)
        assert user["state"] == "hot"
        assert user["message_count"] == 5

    def test_get_user_nonexistent(self, store):
        assert store.get_user(999) is None

    def test_log_event(self, store):
        store.log_event(user_id=1, event_type="viewed", state_from="lead", state_to="active")
        events = store.recent_events(limit=10)
        assert len(events) >= 1
        assert events[0]["user_id"] == 1
        assert events[0]["event_type"] == "viewed"

    def test_segment_counts(self, store):
        store.upsert_user(user_id=1, state="active")
        store.upsert_user(user_id=2, state="active")
        store.upsert_user(user_id=3, state="warm")
        counts = store.segment_counts()
        assert counts["active"] == 2
        assert counts["warm"] == 1

    def test_total_users(self, store):
        store.upsert_user(user_id=1, state="lead")
        store.upsert_user(user_id=2, state="lead")
        assert store.total_users() == 2

    def test_recent_events_limit(self, store):
        for i in range(5):
            store.log_event(user_id=i, event_type="viewed")
        events = store.recent_events(limit=3)
        assert len(events) == 3


# ======================================================================
# TelegramIngestor — message classification
# ======================================================================


class TestClassifyMessage:
    def test_new_chat_members(self, ingestor_config):
        ingestor = TelegramIngestor(ingestor_config)
        result = ingestor._classify_message({"new_chat_members": [{"id": 1}]})
        assert result == "joined"

    def test_left_chat_member(self, ingestor_config):
        ingestor = TelegramIngestor(ingestor_config)
        result = ingestor._classify_message({"left_chat_member": {"id": 1}})
        assert result == "left"

    def test_question_message(self, ingestor_config):
        ingestor = TelegramIngestor(ingestor_config)
        result = ingestor._classify_message({"text": "How does this work?"})
        assert result == "question"

    def test_link_message(self, ingestor_config):
        ingestor = TelegramIngestor(ingestor_config)
        result = ingestor._classify_message({"text": "Check https://example.com"})
        assert result == "link"

    def test_regular_message(self, ingestor_config):
        ingestor = TelegramIngestor(ingestor_config)
        result = ingestor._classify_message({"text": "Hello everyone"})
        assert result == "message"

    def test_callback_query_handled_at_update_level(self, ingestor_config):
        """Callback queries are handled by _process_update, not _classify_message."""
        ingestor = TelegramIngestor(ingestor_config)
        result = ingestor._classify_message(
            {"callback_query": {"data": "btn_1"}}
        )
        assert result is None  # dead code removed — msg is not a real message

    def test_empty_message(self, ingestor_config):
        ingestor = TelegramIngestor(ingestor_config)
        result = ingestor._classify_message({})
        assert result is None

    def test_question_by_prefix(self, ingestor_config):
        """Messages starting with what/how/why/when/where/who should be questions."""
        for prefix in ["what", "how", "why", "when", "where", "who", "can", "do", "is"]:
            ingestor = TelegramIngestor(ingestor_config)
            result = ingestor._classify_message({"text": f"{prefix} is this thing?"})
            assert result == "question", f"'{prefix}' should be classified as question"

    def test_link_by_domain(self, ingestor_config):
        """Messages containing typical URL patterns."""
        for domain in [".com", ".ru", ".org", ".net", "t.me", "https://", "http://"]:
            ingestor = TelegramIngestor(ingestor_config)
            result = ingestor._classify_message({"text": f"Visit us at site{domain}/page"})
            assert result == "link", f"'{domain}' should be classified as link"


# ======================================================================
# TelegramIngestor — poll_once with mocked HTTP
# ======================================================================


class TestPollOnce:
    """Mock the internal _http_client attribute, not the readonly `client` property."""

    def _make_ingestor(self, ingestor_config):
        """Create ingestor and inject a mock httpx client via _http_client."""
        ingestor = TelegramIngestor(ingestor_config)
        mock_client = MagicMock()
        ingestor._http_client = mock_client
        return ingestor, mock_client

    def test_successful_poll_no_updates(self, ingestor_config):
        ingestor, mock_client = self._make_ingestor(ingestor_config)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True, "result": []}
        mock_client.get.return_value = mock_resp

        count = ingestor.poll_once()
        assert count == 0

    def test_successful_poll_with_updates(self, ingestor_config):
        ingestor, mock_client = self._make_ingestor(ingestor_config)
        update = {
            "update_id": 1,
            "message": {
                "message_id": 10,
                "chat": {"id": -100123},
                "from": {"id": 42, "username": "testuser"},
                "text": "Hello",
                "date": int(time.time()),
            },
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True, "result": [update]}
        mock_client.get.return_value = mock_resp

        count = ingestor.poll_once()
        assert count == 1
        user_profile = ingestor.segmenter.get_user(42)
        assert user_profile is not None

    def test_poll_api_error(self, ingestor_config, capsys):
        ingestor, mock_client = self._make_ingestor(ingestor_config)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": False, "description": "Bad token"}
        mock_client.get.return_value = mock_resp

        count = ingestor.poll_once()
        assert count == 0
        captured = capsys.readouterr()
        assert "API error" in captured.out

    def test_poll_http_error(self, ingestor_config, capsys):
        ingestor, mock_client = self._make_ingestor(ingestor_config)
        import httpx
        mock_client.get.side_effect = httpx.HTTPError("Connection failed")

        count = ingestor.poll_once()
        assert count == 0
        captured = capsys.readouterr()
        assert "poll error" in captured.out

    def test_poll_respects_allowed_chat_ids(self, ingestor_config):
        ingestor_config.allowed_chat_ids = [-100999]
        ingestor, mock_client = self._make_ingestor(ingestor_config)
        update = {
            "update_id": 2,
            "message": {
                "message_id": 11,
                "chat": {"id": -100123},  # NOT in allowed list
                "from": {"id": 99, "username": "blocked_user"},
                "text": "Hello",
                "date": int(time.time()),
            },
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True, "result": [update]}
        mock_client.get.return_value = mock_resp

        count = ingestor.poll_once()
        assert count == 1
        assert ingestor.segmenter.get_user(99) is None

    def test_poll_allowed_chat_id_included(self, ingestor_config):
        ingestor_config.allowed_chat_ids = [-100123]
        ingestor, mock_client = self._make_ingestor(ingestor_config)
        update = {
            "update_id": 3,
            "message": {
                "message_id": 12,
                "chat": {"id": -100123},
                "from": {"id": 77, "username": "allowed_user"},
                "text": "Hello",
                "date": int(time.time()),
            },
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True, "result": [update]}
        mock_client.get.return_value = mock_resp

        count = ingestor.poll_once()
        assert count == 1
        assert ingestor.segmenter.get_user(77) is not None

    def test_poll_processes_callback_query(self, ingestor_config):
        ingestor, mock_client = self._make_ingestor(ingestor_config)
        update = {
            "update_id": 4,
            "callback_query": {
                "id": "cb_1",
                "from": {"id": 55, "username": "clicker"},
                "message": {
                    "message_id": 13,
                    "chat": {"id": -100123},
                    "text": "Button message",
                    "date": int(time.time()),
                },
                "data": "btn_click",
            },
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True, "result": [update]}
        mock_client.get.return_value = mock_resp

        count = ingestor.poll_once()
        assert count == 1
        user = ingestor.segmenter.get_user(55)
        assert user is not None
        # callback_query msg has text "Button message" -> "message" -> REPLIED -> LEAD -> WARM
        assert user.state.value == "warm"

    def test_poll_skips_update_without_message(self, ingestor_config):
        ingestor, mock_client = self._make_ingestor(ingestor_config)
        update = {"update_id": 5}  # no message or callback_query
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ok": True, "result": [update]}
        mock_client.get.return_value = mock_resp

        count = ingestor.poll_once()
        assert count == 1
        # no user created
        assert ingestor.segmenter.get_user(0) is None


# ======================================================================
# TelegramIngestor — process_update integration
# ======================================================================


class TestProcessUpdate:
    def test_process_creates_user_and_logs_event(self, ingestor_config):
        ingestor = TelegramIngestor(ingestor_config)
        update = {
            "update_id": 10,
            "message": {
                "message_id": 100,
                "chat": {"id": -100777},
                "from": {"id": 200, "username": "new_user"},
                "text": "How does this work?",
                "date": int(time.time()),
            },
        }
        ingestor._process_update(update)
        user = ingestor.segmenter.get_user(200)
        assert user is not None
        assert user.username == "new_user"
        # Should be stored in DB
        db_user = ingestor.store.get_user(200)
        assert db_user is not None
        assert db_user["state"] == "warm"  # question from LEAD -> WARM

    def test_process_from_callback_query(self, ingestor_config):
        """callback_query update structure: from is inside callback_query."""
        ingestor = TelegramIngestor(ingestor_config)
        update = {
            "update_id": 11,
            "callback_query": {
                "id": "cb_2",
                "from": {"id": 300, "username": "voter"},
                "message": {
                    "message_id": 101,
                    "chat": {"id": -100777},
                    "text": "Vote now",
                    "date": int(time.time()),
                },
                "data": "vote_yes",
            },
        }
        ingestor._process_update(update)
        user = ingestor.segmenter.get_user(300)
        assert user is not None
        # inner message text "Vote now" -> "message" -> REPLIED -> LEAD -> WARM
        assert user.state.value == "warm"

    def test_process_unclassifiable_message(self, ingestor_config):
        ingestor = TelegramIngestor(ingestor_config)
        update = {
            "update_id": 12,
            "message": {
                "message_id": 102,
                "chat": {"id": -100777},
                "from": {"id": 400, "username": "silent"},
                "sticker": {"file_id": "abc123"},  # no text, not join/leave
                "date": int(time.time()),
            },
        }
        ingestor._process_update(update)
        assert ingestor.segmenter.get_user(400) is None
