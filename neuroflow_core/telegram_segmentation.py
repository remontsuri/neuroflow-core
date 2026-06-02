"""Telegram user state machine — tracks user lifecycle through messaging events.

Instead of static tags (noob, engaged, churned), each user follows a state
machine. Messages trigger transitions. Segments flow from current state,
not manual labelling.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class UserState(Enum):
    """Possible states for a Telegram user in the marketing funnel."""

    LEAD = "lead"
    ACTIVE = "active"
    WARM = "warm"
    HOT = "hot"
    COLD = "cold"
    CHURNED = "churned"
    BANNED = "banned"

    def is_convertible(self) -> bool:
        """WARM and HOT users are worth chasing."""
        return self in (UserState.WARM, UserState.HOT)

    def priority_score(self) -> int:
        """Higher score = higher value lead. HOT=100, BANNED=-1."""
        return {
            UserState.HOT: 100,
            UserState.WARM: 70,
            UserState.ACTIVE: 40,
            UserState.LEAD: 20,
            UserState.COLD: 5,
            UserState.CHURNED: 0,
            UserState.BANNED: -1,
        }[self]


class Trigger(Enum):
    """Events that can transition a user from one state to another."""

    JOINED = "joined"
    VIEWED = "viewed"
    REACTED = "reacted"
    REPLIED = "replied"
    CLICKED_LINK = "clicked_link"
    ASKED_QUESTION = "asked_question"
    DM_SENT = "dm_sent"
    PURCHASED = "purchased"
    GAVE_CONTACT = "gave_contact"
    SILENT_7D = "silent_7d"
    SILENT_30D = "silent_30d"
    LEFT = "left"
    SPAM = "spam"
    MANUAL_PROMOTE = "manual_promote"
    MANUAL_DEMOTE = "manual_demote"
    BANNED_USER = "banned_user"


# (current_state, trigger) -> new_state
TRANSITIONS: dict[tuple[UserState, Trigger], UserState] = {
    # LEAD
    (UserState.LEAD, Trigger.VIEWED): UserState.ACTIVE,
    (UserState.LEAD, Trigger.REACTED): UserState.ACTIVE,
    (UserState.LEAD, Trigger.REPLIED): UserState.WARM,
    (UserState.LEAD, Trigger.ASKED_QUESTION): UserState.WARM,
    (UserState.LEAD, Trigger.CLICKED_LINK): UserState.WARM,
    (UserState.LEAD, Trigger.SILENT_7D): UserState.COLD,
    (UserState.LEAD, Trigger.SILENT_30D): UserState.CHURNED,
    (UserState.LEAD, Trigger.LEFT): UserState.CHURNED,
    (UserState.LEAD, Trigger.SPAM): UserState.BANNED,
    # ACTIVE
    (UserState.ACTIVE, Trigger.REACTED): UserState.ACTIVE,
    (UserState.ACTIVE, Trigger.REPLIED): UserState.WARM,
    (UserState.ACTIVE, Trigger.ASKED_QUESTION): UserState.WARM,
    (UserState.ACTIVE, Trigger.CLICKED_LINK): UserState.WARM,
    (UserState.ACTIVE, Trigger.DM_SENT): UserState.HOT,
    (UserState.ACTIVE, Trigger.PURCHASED): UserState.HOT,
    (UserState.ACTIVE, Trigger.SILENT_7D): UserState.COLD,
    (UserState.ACTIVE, Trigger.LEFT): UserState.CHURNED,
    (UserState.ACTIVE, Trigger.SPAM): UserState.BANNED,
    # WARM
    (UserState.WARM, Trigger.REACTED): UserState.WARM,
    (UserState.WARM, Trigger.REPLIED): UserState.WARM,
    (UserState.WARM, Trigger.ASKED_QUESTION): UserState.WARM,
    (UserState.WARM, Trigger.CLICKED_LINK): UserState.WARM,
    (UserState.WARM, Trigger.DM_SENT): UserState.HOT,
    (UserState.WARM, Trigger.GAVE_CONTACT): UserState.HOT,
    (UserState.WARM, Trigger.PURCHASED): UserState.HOT,
    (UserState.WARM, Trigger.SILENT_7D): UserState.COLD,
    (UserState.WARM, Trigger.LEFT): UserState.CHURNED,
    (UserState.WARM, Trigger.BANNED_USER): UserState.BANNED,
    # HOT
    (UserState.HOT, Trigger.PURCHASED): UserState.HOT,
    (UserState.HOT, Trigger.REPLIED): UserState.HOT,
    (UserState.HOT, Trigger.DM_SENT): UserState.HOT,
    (UserState.HOT, Trigger.GAVE_CONTACT): UserState.HOT,
    (UserState.HOT, Trigger.SILENT_7D): UserState.COLD,
    (UserState.HOT, Trigger.SILENT_30D): UserState.CHURNED,
    (UserState.HOT, Trigger.MANUAL_DEMOTE): UserState.ACTIVE,
    (UserState.HOT, Trigger.LEFT): UserState.CHURNED,
    # COLD
    (UserState.COLD, Trigger.VIEWED): UserState.ACTIVE,
    (UserState.COLD, Trigger.REACTED): UserState.ACTIVE,
    (UserState.COLD, Trigger.REPLIED): UserState.WARM,
    (UserState.COLD, Trigger.ASKED_QUESTION): UserState.WARM,
    (UserState.COLD, Trigger.CLICKED_LINK): UserState.WARM,
    (UserState.COLD, Trigger.SILENT_30D): UserState.CHURNED,
    (UserState.COLD, Trigger.LEFT): UserState.CHURNED,
    (UserState.COLD, Trigger.SPAM): UserState.BANNED,
    (UserState.COLD, Trigger.MANUAL_PROMOTE): UserState.ACTIVE,
    # CHURNED (re-engagement)
    (UserState.CHURNED, Trigger.VIEWED): UserState.ACTIVE,
    (UserState.CHURNED, Trigger.REACTED): UserState.ACTIVE,
    (UserState.CHURNED, Trigger.REPLIED): UserState.WARM,
    (UserState.CHURNED, Trigger.MANUAL_PROMOTE): UserState.ACTIVE,
}


@dataclass
class UserProfile:
    """Stores state and metrics for a single Telegram user."""

    user_id: int
    state: UserState = UserState.LEAD
    username: str = ""
    first_seen: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    message_count: int = 0
    reactions_received: int = 0
    dm_count: int = 0
    tags: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_segment(self) -> str:
        """Shortcut: get the segment label."""
        return self.state.value

    def to_dict(self) -> dict[str, Any]:
        """Serialise the user profile for API responses."""
        return {
            "user_id": self.user_id,
            "segment": self.state.value,
            "priority": self.state.priority_score(),
            "convertible": self.state.is_convertible(),
            "username": self.username,
            "message_count": self.message_count,
            "dm_count": self.dm_count,
            "last_active_ago": int(time.time() - self.last_active),
            "tags": self.tags,
            "state_history": [h["to"] for h in self.history[-10:]],
        }


TRIGGER_MAP: dict[str, Trigger] = {
    "join": Trigger.JOINED,
    "joined": Trigger.JOINED,
    "new_chat_member": Trigger.JOINED,
    "view": Trigger.VIEWED,
    "viewed": Trigger.VIEWED,
    "read": Trigger.VIEWED,
    "reaction": Trigger.REACTED,
    "reacted": Trigger.REACTED,
    "like": Trigger.REACTED,
    "reply": Trigger.REPLIED,
    "replied": Trigger.REPLIED,
    "comment": Trigger.REPLIED,
    "message": Trigger.REPLIED,
    "link": Trigger.CLICKED_LINK,
    "click": Trigger.CLICKED_LINK,
    "clicked": Trigger.CLICKED_LINK,
    "question": Trigger.ASKED_QUESTION,
    "ask": Trigger.ASKED_QUESTION,
    "dm": Trigger.DM_SENT,
    "direct": Trigger.DM_SENT,
    "purchase": Trigger.PURCHASED,
    "bought": Trigger.PURCHASED,
    "order": Trigger.PURCHASED,
    "contact": Trigger.GAVE_CONTACT,
    "phone": Trigger.GAVE_CONTACT,
    "email": Trigger.GAVE_CONTACT,
    "leave": Trigger.LEFT,
    "left": Trigger.LEFT,
    "kick": Trigger.LEFT,
    "spam": Trigger.SPAM,
    "ban": Trigger.BANNED_USER,
    "banned": Trigger.BANNED_USER,
    "silent_7d": Trigger.SILENT_7D,
    "silent_30d": Trigger.SILENT_30D,
}


def classify_trigger(msg_type: str) -> Optional[Trigger]:
    """Map a message type string to a Trigger enum.

    Returns None if the type isn't recognised.
    """
    return TRIGGER_MAP.get(msg_type.lower())


@dataclass
class TelegramEvent:
    """Typed event for the Input contract — carries a resolved Trigger, not a raw string.

    Build this in your caller (TelegramIngestor, webhook, test) and pass it
    to TelegramSegmenter.process_event() for type-safe processing.
    """

    user_id: int
    trigger: Trigger
    username: str = ""
    metadata: dict[str, Any] | None = None
    timestamp: float = field(default_factory=time.time)


class TelegramSegmenter:
    """Thread-safe state machine segmenter for Telegram users.

    Usage:

        seg = TelegramSegmenter()
        seg.process_message(user_id=42, msg_type="reaction")
        print(seg.get_segment(42))  # 'active'
    """

    def __init__(self, cold_threshold_days: int = 7, churn_threshold_days: int = 30) -> None:
        """Set up the segmenter: in-memory user dict, thresholds in days."""
        self._users: dict[int, UserProfile] = {}
        self._lock = threading.Lock()
        self._cold_threshold = cold_threshold_days * 86400
        self._churn_threshold = churn_threshold_days * 86400

    def get_or_create(self, user_id: int, username: str = "") -> UserProfile:
        """Get a user profile or create one if it doesn't exist yet."""
        with self._lock:
            if user_id not in self._users:
                self._users[user_id] = UserProfile(user_id=user_id, username=username)
            return self._users[user_id]

    def process_event(self, event: TelegramEvent) -> UserState | None:
        """Process a typed TelegramEvent — the primary entry point for the Input contract.

        Uses the pre-resolved Trigger directly, no string classification.
        Returns the user's new state, or None if the transition was a no-op.
        """
        user = self.get_or_create(event.user_id, event.username)
        user.last_active = time.time()
        user.message_count += 1

        with self._lock:
            key = (user.state, event.trigger)
            new_state = TRANSITIONS.get(key)
            if new_state and new_state != user.state:
                user.history.append({
                    "from": user.state.value,
                    "to": new_state.value,
                    "trigger": event.trigger.value,
                    "ts": time.time(),
                })
                user.state = new_state

                if event.trigger == Trigger.DM_SENT:
                    user.dm_count += 1
                if event.trigger in (Trigger.REACTED, Trigger.REPLIED):
                    user.reactions_received += 1

        return user.state

    def process_message(
        self,
        user_id: int,
        msg_type: str,
        username: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> UserState | None:
        """String-based convenience wrapper — resolves msg_type to Trigger
        and delegates to process_event(). Prefer process_event() for new code.
        """
        trigger = classify_trigger(msg_type)
        if not trigger:
            return None
        return self.process_event(TelegramEvent(
            user_id=user_id,
            trigger=trigger,
            username=username,
            metadata=metadata,
        ))

    def get_segment(self, user_id: int) -> str | None:
        """Get a user's current segment label, or None if unknown."""
        user = self._users.get(user_id)
        return user.to_segment() if user else None

    def get_user(self, user_id: int) -> UserProfile | None:
        """Get a user profile, or None if not seen before."""
        return self._users.get(user_id)

    def run_decay(self) -> None:
        """Move silent users to COLD or CHURNED based on inactivity thresholds."""
        now = time.time()
        with self._lock:
            for user in list(self._users.values()):
                if user.state in (UserState.BANNED, UserState.CHURNED):
                    continue
                elapsed = now - user.last_active
                if elapsed > self._churn_threshold and user.state != UserState.CHURNED:
                    self._apply_forced_transition(user, Trigger.SILENT_30D)
                elif elapsed > self._cold_threshold and user.state not in (
                    UserState.COLD,
                    UserState.LEAD,
                    UserState.CHURNED,
                ):
                    self._apply_forced_transition(user, Trigger.SILENT_7D)

    def _apply_forced_transition(self, user: UserProfile, trigger: Trigger) -> None:
        """Move a user to a new state without an external event (e.g. decay)."""
        key = (user.state, trigger)
        new_state = TRANSITIONS.get(key)
        if new_state and new_state != user.state:
            user.history.append({
                "from": user.state.value,
                "to": new_state.value,
                "trigger": f"decay:{trigger.value}",
                "ts": time.time(),
            })
            user.state = new_state

    def segment_counts(self) -> dict[str, int]:
        """Count users in each segment, under the lock."""
        with self._lock:
            counts: dict[str, int] = {}
            for u in self._users.values():
                seg = u.state.value
                counts[seg] = counts.get(seg, 0) + 1
            return counts

    def hot_leads(self) -> list[UserProfile]:
        """Users most likely to convert, sorted by priority."""
        with self._lock:
            return sorted(
                [u for u in self._users.values() if u.state.is_convertible()],
                key=lambda u: u.state.priority_score(),
                reverse=True,
            )

    def export(self) -> dict[str, Any]:
        """Full snapshot: all users, segments, counts."""
        return {
            "total_users": len(self._users),
            "segments": self.segment_counts(),
            "hot_leads_count": len(self.hot_leads()),
            "users": {str(uid): u.to_dict() for uid, u in self._users.items()},
        }

    def export_json(self, path: str = "") -> str:
        """Dump the full snapshot to JSON. Default path uses $NEUROFLOW_SEGMENTS_PATH
        env var or /tmp/telegram_segments.json."""
        path = path or os.environ.get("NEUROFLOW_SEGMENTS_PATH", "/tmp/telegram_segments.json")
        with open(path, "w") as f:
            json.dump(self.export(), f, indent=2, ensure_ascii=False)
        return path
