"""Shared fixtures for neuroflow-core tests."""

from pathlib import Path
from typing import Generator

import pytest

from neuroflow_core.telegram_segmentation import (
    TelegramSegmenter,
    UserState,
    Trigger,
    TRANSITIONS,
)
from neuroflow_core.telegram_ingestor import IngestorConfig
from neuroflow_core.osw_engine import GraphMemory, DAG, AgentStateMachine, AgentCard


# ---------------------------------------------------------------------------
# Telegram segmentation fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def segmenter() -> TelegramSegmenter:
    """Fresh TelegramSegmenter with default thresholds."""
    return TelegramSegmenter(cold_threshold_days=7, churn_threshold_days=30)


@pytest.fixture
def populated_segmenter(segmenter: TelegramSegmenter) -> TelegramSegmenter:
    """Segmenter with users in various states."""
    # LEAD is default, transition to other states with TRIGGER_MAP-valid keys
    segmenter.process_message(101, "joined", username="lead_user")   # stays LEAD
    segmenter.process_message(102, "viewed", username="active_user") # LEAD->ACTIVE
    segmenter.process_message(103, "replied", username="warm_user")  # LEAD->WARM
    segmenter.process_message(104, "viewed", username="hot_user_first")  # ACTIVE
    segmenter.process_message(104, "dm", username="hot_user")        # ACTIVE->HOT
    segmenter.process_message(105, "left", username="churned_user")  # LEAD->CHURNED
    segmenter.process_message(106, "spam", username="banned_user")   # LEAD->BANNED
    return segmenter


# ---------------------------------------------------------------------------
# Ingestor fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ingestor_config() -> IngestorConfig:
    return IngestorConfig(
        bot_token="123:test_token",
        api_base="https://fake.telegram.org",
        poll_interval_s=1,
        http_port=0,
        db_path="/tmp/test_ingestor.db",
    )


# ---------------------------------------------------------------------------
# OSW engine fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dag() -> DAG:
    return DAG()


@pytest.fixture
def memory(tmp_path: Path) -> Generator[GraphMemory, None, None]:
    db_path = str(tmp_path / "test_memory.db")
    m = GraphMemory(db_path=db_path)
    yield m
    m.clear()


@pytest.fixture
def state_machine() -> AgentStateMachine:
    return AgentStateMachine()


@pytest.fixture
def agent_card() -> AgentCard:
    return AgentCard(name="test_agent", role="researcher", model="gpt-4")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_TRIGGERS = list(Trigger)
ALL_STATES = list(UserState)


def all_transition_combos() -> list[tuple[UserState, Trigger, UserState | None]]:
    """Generate every (from_state, trigger, expected_state) combination
    based on the TRANSITIONS table."""
    results: list[tuple[UserState, Trigger, UserState | None]] = []
    for state in UserState:
        for trigger in Trigger:
            expected = TRANSITIONS.get((state, trigger))
            results.append((state, trigger, expected))
    return results
