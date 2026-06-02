"""NeuroFlow Core — Telegram user segmentation and AI orchestration.

Packages:
  telegram_segmentation  — State machine for user lifecycle tracking
  telegram_ingestor      — Telegram poller with REST API + dashboard
  osw_engine             — Orchestrate-Synthesize-Workflow DAG engine
"""

__version__ = "0.1.0"

from neuroflow_core.telegram_segmentation import (
    TelegramSegmenter,
    UserProfile,
    UserState,
    Trigger,
    TRIGGER_MAP,
    classify_trigger,
)
from neuroflow_core.telegram_ingestor import TelegramIngestor, IngestorConfig, config_from_env
from neuroflow_core.osw_engine import (
    OSWEngine,
    DAG,
    GraphMemory,
    AgentStateMachine,
    AgentCard,
    AgentState,
    DispatchFn,
)

__all__ = [
    "TelegramSegmenter",
    "UserProfile",
    "UserState",
    "Trigger",
    "TRIGGER_MAP",
    "classify_trigger",
    "TelegramIngestor",
    "IngestorConfig",
    "config_from_env",
    "OSWEngine",
    "DAG",
    "GraphMemory",
    "AgentStateMachine",
    "AgentCard",
    "AgentState",
    "DispatchFn",
]
