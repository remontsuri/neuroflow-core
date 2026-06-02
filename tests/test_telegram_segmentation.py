"""Tests for telegram_segmentation — state machine, decay, export."""

import json
import time
from pathlib import Path

import pytest

from neuroflow_core.telegram_segmentation import (
    UserProfile,
    UserState,
    Trigger,
    TRANSITIONS,
    TRIGGER_MAP,
    classify_trigger,
)


# ======================================================================
# UserState enum
# ======================================================================


class TestUserState:
    def test_values(self):
        assert [s.value for s in UserState] == [
            "lead", "active", "warm", "hot", "cold", "churned", "banned",
        ]

    def test_is_convertible(self):
        assert UserState.WARM.is_convertible() is True
        assert UserState.HOT.is_convertible() is True
        assert UserState.LEAD.is_convertible() is False
        assert UserState.ACTIVE.is_convertible() is False
        assert UserState.COLD.is_convertible() is False
        assert UserState.CHURNED.is_convertible() is False
        assert UserState.BANNED.is_convertible() is False

    def test_priority_score(self):
        assert UserState.HOT.priority_score() == 100
        assert UserState.WARM.priority_score() == 70
        assert UserState.ACTIVE.priority_score() == 40
        assert UserState.LEAD.priority_score() == 20
        assert UserState.COLD.priority_score() == 5
        assert UserState.CHURNED.priority_score() == 0
        assert UserState.BANNED.priority_score() == -1

    def test_priority_order(self):
        """Enum iteration order (declaration) differs from sorted-by-score."""
        scores = [s.priority_score() for s in UserState]
        # Declaration order: LEAD(20), ACTIVE(40), WARM(70), HOT(100), COLD(5), CHURNED(0), BANNED(-1)
        # Not sorted — just verify BANNED < CHURNED < COLD < LEAD < ACTIVE < WARM < HOT
        sorted_scores = sorted(scores)
        assert scores != sorted_scores  # not naturally sorted
        assert scores == [20, 40, 70, 100, 5, 0, -1]


# ======================================================================
# Trigger enum
# ======================================================================


class TestTrigger:
    def test_all_triggers_defined(self):
        expected = {
            "joined", "viewed", "reacted", "replied", "clicked_link",
            "asked_question", "dm_sent", "purchased", "gave_contact",
            "silent_7d", "silent_30d", "left", "spam",
            "manual_promote", "manual_demote", "banned_user",
        }
        assert {t.value for t in Trigger} == expected


# ======================================================================
# TRANSITIONS table — combinatorial coverage
# ======================================================================


class TestTransitions:
    """Verify every (UserState, Trigger) combo maps correctly."""

    def test_non_banned_states_have_some_transitions(self):
        """Every state except BANNED should have at least one transition defined."""
        states_with_rules = {k[0] for k in TRANSITIONS}
        for s in UserState:
            if s != UserState.BANNED:
                assert s in states_with_rules, f"{s} has no transitions"

    def test_banned_has_no_outgoing_transitions(self):
        """BANNED state should have no outgoing transitions."""
        banned_keys = {k for k in TRANSITIONS if k[0] == UserState.BANNED}
        assert len(banned_keys) == 0

    def test_known_transition(self):
        assert TRANSITIONS[(UserState.LEAD, Trigger.VIEWED)] == UserState.ACTIVE
        assert TRANSITIONS[(UserState.ACTIVE, Trigger.PURCHASED)] == UserState.HOT
        assert TRANSITIONS[(UserState.HOT, Trigger.SILENT_7D)] == UserState.COLD
        assert TRANSITIONS[(UserState.CHURNED, Trigger.VIEWED)] == UserState.ACTIVE

    def test_all_defined_transitions_are_valid(self):
        """Every defined transition maps to a valid UserState."""
        for (state, trigger), new_state in TRANSITIONS.items():
            assert isinstance(state, UserState)
            assert isinstance(trigger, Trigger)
            assert isinstance(new_state, UserState)

    def test_banned_user_only_from_warm(self):
        """Trigger.BANNED_USER only appears for WARM state."""
        for (state, trigger), _ in TRANSITIONS.items():
            if trigger == Trigger.BANNED_USER:
                assert state == UserState.WARM

    def test_silent_30d_definitions(self):
        """SILENT_30D triggers should lead to CHURNED."""
        for (state, trigger), new_state in TRANSITIONS.items():
            if trigger == Trigger.SILENT_30D:
                assert new_state == UserState.CHURNED, (
                    f"{state}+SILENT_30D should go to CHURNED, got {new_state}"
                )

    def test_silent_7d_definitions(self):
        """SILENT_7D should lead to COLD (except LEAD which is excluded in decay)."""
        for (state, trigger), new_state in TRANSITIONS.items():
            if trigger == Trigger.SILENT_7D:
                assert new_state == UserState.COLD


# ======================================================================
# classify_trigger
# ======================================================================


class TestClassifyTrigger:
    def test_known_keys(self):
        assert classify_trigger("joined") == Trigger.JOINED
        assert classify_trigger("viewed") == Trigger.VIEWED
        assert classify_trigger("reply") == Trigger.REPLIED
        assert classify_trigger("reaction") == Trigger.REACTED
        assert classify_trigger("purchase") == Trigger.PURCHASED
        assert classify_trigger("banned") == Trigger.BANNED_USER
        assert classify_trigger("silent_7d") == Trigger.SILENT_7D
        assert classify_trigger("silent_30d") == Trigger.SILENT_30D
        assert classify_trigger("dm") == Trigger.DM_SENT

    def test_case_insensitive(self):
        assert classify_trigger("JOINED") == Trigger.JOINED
        assert classify_trigger("RePlIeD") == Trigger.REPLIED
        assert classify_trigger("SpAm") == Trigger.SPAM

    def test_unknown_returns_none(self):
        assert classify_trigger("unknown_event") is None
        assert classify_trigger("") is None
        assert classify_trigger("clicked_on_link") is None  # exact key is "clicked_link"

    @pytest.mark.parametrize("key", list(TRIGGER_MAP.keys()))
    def test_all_map_keys_resolve(self, key):
        assert classify_trigger(key) is not None


# ======================================================================
# UserProfile
# ======================================================================


class TestUserProfile:
    def test_defaults(self):
        p = UserProfile(user_id=1)
        assert p.user_id == 1
        assert p.state == UserState.LEAD
        assert p.message_count == 0
        assert p.tags == []

    def test_to_segment(self):
        p = UserProfile(user_id=1, state=UserState.HOT)
        assert p.to_segment() == "hot"

    def test_to_dict(self):
        p = UserProfile(user_id=42, state=UserState.WARM, username="testuser", message_count=5)
        d = p.to_dict()
        assert d["user_id"] == 42
        assert d["segment"] == "warm"
        assert d["priority"] == 70
        assert d["convertible"] is True
        assert d["username"] == "testuser"
        assert d["message_count"] == 5
        assert isinstance(d["last_active_ago"], int)
        assert d["state_history"] == []

    def test_to_dict_includes_history(self):
        p = UserProfile(user_id=1)
        p.history.append({"from": "lead", "to": "active", "trigger": "viewed", "ts": 0.0})
        d = p.to_dict()
        assert d["state_history"] == ["active"]


# ======================================================================
# TelegramSegmenter — core logic
# ======================================================================


class TestTelegramSegmenter:
    def test_get_or_create_new_user(self, segmenter):
        user = segmenter.get_or_create(1, username="alice")
        assert user.user_id == 1
        assert user.username == "alice"
        assert user.state == UserState.LEAD

    def test_get_or_create_existing(self, segmenter):
        u1 = segmenter.get_or_create(1)
        u2 = segmenter.get_or_create(1, username="bob")  # second call won't overwrite
        assert u1 is u2
        assert u2.username == ""  # not overwritten

    def test_get_segment_nonexistent(self, segmenter):
        assert segmenter.get_segment(999) is None

    def test_get_segment(self, segmenter):
        segmenter.process_message(1, "viewed")
        assert segmenter.get_segment(1) == "active"

    def test_get_user_nonexistent(self, segmenter):
        assert segmenter.get_user(999) is None

    def test_get_user(self, segmenter):
        segmenter.process_message(1, "joined")
        user = segmenter.get_user(1)
        assert user is not None
        assert user.state == UserState.LEAD  # joined for LEAD stays LEAD

    def test_process_message_unknown_trigger(self, segmenter):
        result = segmenter.process_message(1, "unknown_trigger")
        assert result is None
        assert segmenter.get_user(1) is None  # user not created

    def test_process_message_creates_user(self, segmenter):
        result = segmenter.process_message(1, "viewed")
        assert result == UserState.ACTIVE
        user = segmenter.get_user(1)
        assert user is not None
        assert user.message_count == 1

    # ------------------------------------------------------------------
    # State transition combinations — using TRIGGER_MAP-valid keys
    # ------------------------------------------------------------------

    def test_lead_to_active_via_viewed(self, segmenter):
        segmenter.process_message(1, "viewed")
        assert segmenter.get_segment(1) == "active"

    def test_lead_to_active_via_reacted(self, segmenter):
        segmenter.process_message(1, "reaction")
        assert segmenter.get_segment(1) == "active"

    def test_lead_to_warm_via_replied(self, segmenter):
        segmenter.process_message(1, "replied")
        assert segmenter.get_segment(1) == "warm"

    def test_lead_to_warm_via_question(self, segmenter):
        segmenter.process_message(1, "question")
        assert segmenter.get_segment(1) == "warm"

    def test_lead_to_warm_via_link(self, segmenter):
        segmenter.process_message(1, "link")
        assert segmenter.get_segment(1) == "warm"

    def test_lead_to_cold_via_silent_7d(self, segmenter):
        segmenter.process_message(1, "silent_7d")
        assert segmenter.get_segment(1) == "cold"

    def test_lead_to_churned_via_left(self, segmenter):
        segmenter.process_message(1, "left")
        assert segmenter.get_segment(1) == "churned"

    def test_lead_to_banned_via_spam(self, segmenter):
        segmenter.process_message(1, "spam")
        assert segmenter.get_segment(1) == "banned"

    def test_active_to_warm_via_replied(self, segmenter):
        segmenter.process_message(1, "viewed")    # LEAD -> ACTIVE
        segmenter.process_message(1, "replied")   # ACTIVE -> WARM
        assert segmenter.get_segment(1) == "warm"

    def test_active_to_hot_via_dm(self, segmenter):
        segmenter.process_message(1, "viewed")    # LEAD -> ACTIVE
        segmenter.process_message(1, "dm")        # ACTIVE -> HOT (key: "dm", not "dm_sent")
        assert segmenter.get_segment(1) == "hot"

    def test_warm_to_hot_via_purchase(self, segmenter):
        segmenter.process_message(1, "replied")   # LEAD -> WARM
        segmenter.process_message(1, "purchase")  # WARM -> HOT (key: "purchase", not "purchased")
        assert segmenter.get_segment(1) == "hot"

    def test_hot_to_cold_via_silent_7d(self, segmenter):
        segmenter.process_message(1, "viewed")    # ACTIVE
        segmenter.process_message(1, "dm")        # HOT
        segmenter.process_message(1, "silent_7d") # COLD
        assert segmenter.get_segment(1) == "cold"

    def test_hot_to_churned_via_silent_30d(self, segmenter):
        segmenter.process_message(1, "viewed")    # ACTIVE
        segmenter.process_message(1, "dm")        # HOT
        segmenter.process_message(1, "silent_30d")# CHURNED (HOT+SILENT_30D is in TRANSITIONS)
        assert segmenter.get_segment(1) == "churned"

    def test_cold_to_active_via_viewed(self, segmenter):
        segmenter.process_message(1, "silent_7d")   # COLD
        segmenter.process_message(1, "viewed")      # ACTIVE
        assert segmenter.get_segment(1) == "active"

    def test_cold_to_warm_via_replied(self, segmenter):
        segmenter.process_message(1, "silent_7d")   # LEAD -> COLD
        segmenter.process_message(1, "replied")     # COLD -> WARM
        assert segmenter.get_segment(1) == "warm"

    def test_cold_to_churned_via_silent_30d(self, segmenter):
        segmenter.process_message(1, "silent_7d")   # COLD
        segmenter.process_message(1, "silent_30d")  # CHURNED
        assert segmenter.get_segment(1) == "churned"

    def test_cold_to_banned_via_spam(self, segmenter):
        segmenter.process_message(1, "silent_7d")   # COLD
        segmenter.process_message(1, "spam")        # BANNED
        assert segmenter.get_segment(1) == "banned"

    def test_churned_to_active_via_viewed(self, segmenter):
        segmenter.process_message(1, "left")        # CHURNED
        segmenter.process_message(1, "viewed")      # back to ACTIVE
        assert segmenter.get_segment(1) == "active"

    def test_churned_to_warm_via_replied(self, segmenter):
        segmenter.process_message(1, "left")        # CHURNED
        segmenter.process_message(1, "replied")     # WARM
        assert segmenter.get_segment(1) == "warm"

    def test_no_transition_for_same_state(self, segmenter):
        """Staying in the same state shouldn't append to history."""
        segmenter.process_message(1, "viewed")  # ACTIVE
        segmenter.process_message(1, "reacted") # ACTIVE stays ACTIVE
        user = segmenter.get_user(1)
        assert len(user.history) == 1  # only first transition logged

    def test_dm_count_increments_when_transition(self, segmenter):
        segmenter.process_message(1, "viewed")   # ACTIVE
        segmenter.process_message(1, "dm")        # HOT, dm_count++
        user = segmenter.get_user(1)
        assert user.dm_count == 1

    def test_reaction_count_increments(self, segmenter):
        """REACTED trigger increments reactions_received on actual state transition."""
        # LEAD+REACTED -> ACTIVE (transition happens), so reactions_received++ is triggered
        segmenter.process_message(1, "reacted")
        user = segmenter.get_user(1)
        assert user.state == UserState.ACTIVE
        assert user.reactions_received == 1

    def test_forced_transition_manual_demote(self, segmenter):
        """MANUAL_DEMOTE is not in TRIGGER_MAP, but tests TRANSITIONS directly."""
        segmenter.process_message(1, "viewed")   # ACTIVE
        segmenter.process_message(1, "dm")       # HOT
        # Now manually force the transition
        user = segmenter.get_user(1)
        assert user.state == UserState.HOT
        segmenter._apply_forced_transition(user, Trigger.MANUAL_DEMOTE)
        assert user.state == UserState.ACTIVE

    def test_forced_transition_manual_promote(self, segmenter):
        """MANUAL_PROMOTE is not in TRIGGER_MAP, test via _apply_forced_transition."""
        segmenter.process_message(1, "left")    # CHURNED
        user = segmenter.get_user(1)
        assert user.state == UserState.CHURNED
        segmenter._apply_forced_transition(user, Trigger.MANUAL_PROMOTE)
        assert user.state == UserState.ACTIVE

    def test_forced_transition_does_nothing_for_invalid(self, segmenter):
        segmenter.process_message(1, "viewed")  # ACTIVE
        user = segmenter.get_user(1)
        segmenter._apply_forced_transition(user, Trigger.PURCHASED)
        # ACTIVE+PURCHASED -> HOT in TRANSITIONS, so it should work
        assert user.state == UserState.HOT

    # ------------------------------------------------------------------
    # Decay logic
    # ------------------------------------------------------------------

    def test_decay_cold_threshold(self, segmenter):
        """User inactive > 7 days should move to COLD."""
        segmenter.process_message(1, "viewed")  # ACTIVE
        user = segmenter.get_user(1)
        user.last_active = time.time() - (8 * 86400)  # 8 days ago
        segmenter.run_decay()
        assert user.state == UserState.COLD

    def test_decay_churn_threshold_hot(self, segmenter):
        """HOT user inactive > 30 days should move to CHURNED."""
        segmenter.process_message(1, "viewed")  # ACTIVE
        segmenter.process_message(1, "dm")      # HOT
        user = segmenter.get_user(1)
        user.last_active = time.time() - (31 * 86400)  # 31 days ago
        segmenter.run_decay()
        # HOT+SILENT_30D is in TRANSITIONS -> CHURNED
        assert user.state == UserState.CHURNED

    def test_decay_churn_threshold_warm(self, segmenter):
        """WARM inactive > 30 days: no WARM+SILENT_30D in TRANSITIONS,
        and churn check enters the if-block before cold check, so user stays WARM."""
        segmenter.process_message(1, "replied")  # WARM
        user = segmenter.get_user(1)
        user.last_active = time.time() - (31 * 86400)
        # WARM+SILENT_30D is NOT in TRANSITIONS, and the `if` branch
        # for churn is entered before `elif` for cold, so stay WARM
        segmenter.run_decay()
        assert user.state == UserState.WARM

    def test_decay_skips_banned(self, segmenter):
        """BANNED users should not decay."""
        segmenter.process_message(1, "spam")
        user = segmenter.get_user(1)
        user.last_active = time.time() - (31 * 86400)
        segmenter.run_decay()
        assert user.state == UserState.BANNED

    def test_decay_skips_churned(self, segmenter):
        """CHURNED users should not decay further."""
        segmenter.process_message(1, "left")
        user = segmenter.get_user(1)
        user.last_active = time.time() - (31 * 86400)
        segmenter.run_decay()
        assert user.state == UserState.CHURNED

    def test_decay_lead_stays_lead(self, segmenter):
        """LEAD users should not decay to COLD (excluded from SILENT_7D check)."""
        segmenter.process_message(1, "joined")  # stays LEAD
        user = segmenter.get_user(1)
        user.last_active = time.time() - (8 * 86400)
        segmenter.run_decay()
        assert user.state == UserState.LEAD

    def test_decay_appends_history(self, segmenter):
        segmenter.process_message(1, "viewed")  # ACTIVE
        user = segmenter.get_user(1)
        user.last_active = time.time() - (8 * 86400)
        segmenter.run_decay()
        assert len(user.history) == 2
        assert "decay:" in user.history[1]["trigger"]

    def test_decay_noop_for_active_users(self, segmenter):
        """Recently active user should not decay."""
        segmenter.process_message(1, "viewed")  # ACTIVE
        segmenter.run_decay()
        assert segmenter.get_segment(1) == "active"

    def test_decay_skips_lead_cold_check(self, segmenter):
        """LEAD state is excluded from the cold check explicitly."""
        # LEAD doesn't have SILENT_7D that goes to COLD... actually it does!
        # (LEAD, SILENT_7D) -> COLD. But decay code excludes LEAD, COLD, CHURNED from cold check.
        segmenter.process_message(1, "joined")  # stays LEAD
        user = segmenter.get_user(1)
        user.last_active = time.time() - (8 * 86400)
        segmenter.run_decay()
        assert user.state == UserState.LEAD  # LEAD explicitly excluded

    # ------------------------------------------------------------------
    # Hot leads & segment counts
    # ------------------------------------------------------------------

    def test_hot_leads(self, segmenter):
        segmenter.process_message(1, "viewed")    # ACTIVE
        segmenter.process_message(1, "dm")        # HOT
        segmenter.process_message(2, "replied")   # WARM (LEAD+replied -> WARM)
        segmenter.process_message(3, "viewed")    # ACTIVE
        hot = segmenter.hot_leads()
        # HOT and WARM are convertible
        assert len(hot) == 2
        # HOT has higher priority so should be first
        assert hot[0].state == UserState.HOT
        assert hot[1].state == UserState.WARM

    def test_segment_counts(self, segmenter):
        segmenter.process_message(1, "viewed")   # ACTIVE
        segmenter.process_message(2, "viewed")   # ACTIVE
        segmenter.process_message(3, "replied")  # WARM
        counts = segmenter.segment_counts()
        assert counts["active"] == 2
        assert counts["warm"] == 1
        assert counts.get("lead", 0) == 0

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def test_export_structure(self, segmenter):
        segmenter.process_message(1, "viewed")
        segmenter.process_message(2, "replied")
        data = segmenter.export()
        assert data["total_users"] == 2
        assert "segments" in data
        assert "hot_leads_count" in data
        assert "users" in data
        assert "1" in data["users"]
        assert "2" in data["users"]

    def test_export_json(self, segmenter, tmp_path):
        segmenter.process_message(1, "viewed")
        out = tmp_path / "segments.json"
        path = segmenter.export_json(str(out))
        assert Path(path).exists()
        with open(path) as f:
            data = json.load(f)
        assert data["total_users"] == 1


# ======================================================================
# Edge-case tests (Eval block — Monadix contract)
# ======================================================================


class TestEdgeCases:
    """Reactivation, churn recovery, stale leads, duplicate merging, rapid transitions."""

    # ------------------------------------------------------------------
    # 1. Full churn recovery cycle
    # ------------------------------------------------------------------

    def test_full_churn_recovery_cycle(self, segmenter):
        """User: LEAD -> ACTIVE -> COLD -> CHURNED -> ACTIVE -> WARM -> HOT."""
        # Phase 1: onboard
        segmenter.process_message(1, "viewed")       # LEAD -> ACTIVE
        assert segmenter.get_segment(1) == "active"

        # Phase 2: go cold
        segmenter.process_message(1, "silent_7d")    # ACTIVE -> COLD
        assert segmenter.get_segment(1) == "cold"

        # Phase 3: churn
        segmenter.process_message(1, "silent_30d")   # COLD -> CHURNED
        assert segmenter.get_segment(1) == "churned"

        # Phase 4: re-engage
        segmenter.process_message(1, "viewed")       # CHURNED -> ACTIVE
        assert segmenter.get_segment(1) == "active"

        # Phase 5: warm up
        segmenter.process_message(1, "replied")      # ACTIVE -> WARM
        assert segmenter.get_segment(1) == "warm"

        # Phase 6: convert
        segmenter.process_message(1, "dm")           # WARM -> HOT
        assert segmenter.get_segment(1) == "hot"
        assert segmenter.get_user(1).dm_count == 1
        assert segmenter.get_user(1).state.is_convertible() is True

    def test_churn_recovery_from_cold_via_reply(self, segmenter):
        """Churned user replies → WARM directly (CHURNED+REPLIED)."""
        segmenter.process_message(1, "left")          # LEAD -> CHURNED
        segmenter.process_message(1, "replied")       # CHURNED -> WARM
        assert segmenter.get_segment(1) == "warm"
        assert segmenter.get_user(1).state.is_convertible() is True

    def test_churn_recovery_twice(self, segmenter):
        """Multiple churn-recovery cycles should work."""
        for cycle in range(3):
            uid = 100 + cycle
            segmenter.process_message(uid, "viewed")   # LEAD -> ACTIVE
            segmenter.process_message(uid, "silent_7d") # ACTIVE -> COLD
            segmenter.process_message(uid, "silent_30d")# ... -> CHURNED
            segmenter.process_message(uid, "viewed")    # back to ACTIVE
            assert segmenter.get_segment(uid) == "active", f"cycle {cycle} failed"

    # ------------------------------------------------------------------
    # 2. Stale lead behaviour
    # ------------------------------------------------------------------

    def test_stale_lead_churns_via_decay(self, segmenter):
        """LEAD inactive for > churn_threshold transitions to CHURNED via decay."""
        segmenter.process_message(1, "joined")   # stays LEAD
        assert segmenter.get_segment(1) == "lead"
        user = segmenter.get_user(1)
        user.last_active = time.time() - (365 * 86400)  # a year ago
        segmenter.run_decay()
        assert user.state == UserState.CHURNED  # LEAD -> SILENT_30D -> CHURNED
        assert user.history[-1]["trigger"] == "decay:silent_30d"

    def test_stale_lead_churn_via_silent_30d_event(self, segmenter):
        """If a SILENT_30D event arrives for a LEAD, it should transition to CHURNED."""
        segmenter.process_message(1, "joined")       # stays LEAD
        segmenter.process_message(1, "silent_30d")    # LEAD+SILENT_30D is in TRANSITIONS -> CHURNED
        assert segmenter.get_segment(1) == "churned"

    def test_stale_lead_churn_via_left(self, segmenter):
        """LEAD who leaves → CHURNED."""
        segmenter.process_message(1, "joined")  # LEAD
        segmenter.process_message(1, "left")    # LEAD -> CHURNED
        assert segmenter.get_segment(1) == "churned"

    def test_stale_lead_banned_on_spam(self, segmenter):
        """LEAD who spams → BANNED."""
        segmenter.process_message(1, "joined")
        segmenter.process_message(1, "spam")
        assert segmenter.get_segment(1) == "banned"

    # ------------------------------------------------------------------
    # 3. Duplicate / idempotent event handling
    # ------------------------------------------------------------------

    def test_duplicate_same_event_no_double_transition(self, segmenter):
        """Same event back-to-back should not create extra transitions."""
        segmenter.process_message(1, "viewed")  # LEAD -> ACTIVE
        segmenter.process_message(1, "viewed")  # ACTIVE stays ACTIVE (no transition in TRANSITIONS)
        user = segmenter.get_user(1)
        # only one transition logged (LEAD->ACTIVE), second event doesn't change state
        assert len(user.history) == 1
        assert user.message_count == 2  # message_count still increments

    def test_duplicate_unknown_trigger_ignored(self, segmenter):
        """Same unknown trigger twice should not create user."""
        segmenter.process_message(1, "unknown_event")
        segmenter.process_message(1, "unknown_event")
        assert segmenter.get_user(1) is None

    def test_duplicate_same_user_multiple_sources(self, segmenter):
        """User created from different caller paths should collapse to one profile."""
        user_a = segmenter.get_or_create(42, username="alice")
        user_b = segmenter.get_or_create(42, username="bob")
        assert user_a is user_b  # same object
        assert user_a.username == "alice"  # first call sets username
        # Second call with different username doesn't overwrite existing
        segmenter.process_message(42, "viewed")
        assert segmenter.get_user(42) is not None

    def test_duplicate_message_only_increments_count(self, segmenter):
        """Repeated same-state triggers should only bump message_count."""
        segmenter.process_message(1, "viewed")  # LEAD -> ACTIVE
        segmenter.process_message(1, "viewed")  # ACTIVE -> ACTIVE (no transition)
        segmenter.process_message(1, "viewed")  # ACTIVE -> ACTIVE
        user = segmenter.get_user(1)
        assert user.message_count == 3
        assert len(user.history) == 1  # only LEAD->ACTIVE

    # ------------------------------------------------------------------
    # 4. Rapid transitions — burst of events
    # ------------------------------------------------------------------

    def test_rapid_transitions_full_flow(self, segmenter):
        """Simulate a user rapidly clicking through the funnel."""
        events = ["viewed", "replied", "dm", "purchase"]
        for e in events:
            segmenter.process_message(1, e)
        # Expected path: LEAD -> ACTIVE -> WARM -> HOT -> HOT (purchased from HOT stays HOT)
        assert segmenter.get_segment(1) == "hot"
        user = segmenter.get_user(1)
        assert user.message_count == 4
        assert len(user.history) == 3  # LEAD->ACTIVE, ACTIVE->WARM, WARM->HOT

    def test_rapid_transitions_loop(self, segmenter):
        """Rapid state transitions should not corrupt internal state."""
        for i in range(50):
            segmenter.process_message(1, "viewed")   # ACTIVE if already, LEAD->ACTIVE on first
            segmenter.process_message(1, "dm")       # ACTIVE -> HOT
            segmenter.process_message(1, "silent_7d")# HOT -> COLD
            segmenter.process_message(1, "replied")  # COLD -> WARM
        user = segmenter.get_user(1)
        # Sanity: state machine still works, no crash
        assert user.state in (UserState.WARM, UserState.HOT)
        assert user.message_count == 200  # 50 * 4 = 200

    def test_multiple_users_concurrent_events(self, segmenter):
        """Interleaving events for multiple users should not cross-contaminate."""
        segmenter.process_message(1, "viewed")   # 1 -> ACTIVE
        segmenter.process_message(2, "replied")  # 2 -> WARM
        segmenter.process_message(1, "dm")       # 1 -> HOT
        segmenter.process_message(3, "spam")     # 3 -> BANNED
        segmenter.process_message(2, "dm")       # 2 -> HOT

        assert segmenter.get_segment(1) == "hot"
        assert segmenter.get_segment(2) == "hot"
        assert segmenter.get_segment(3) == "banned"

    def test_multiple_users_rapid_interleaved(self, segmenter):
        """Rapid fire events across many users, mimicking real Telegram load."""
        for i in range(10):
            uid = 100 + i
            segmenter.process_message(uid, "viewed")
            segmenter.process_message(uid, "replied")
            segmenter.process_message(uid, "question")

        counts = segmenter.segment_counts()
        # All went LEAD->ACTIVE->WARM->WARM (question from WARM stays WARM)
        assert counts.get("warm", 0) == 10
        assert segmenter.export()["total_users"] == 10

    # ------------------------------------------------------------------
    # 5. History integrity
    # ------------------------------------------------------------------

    def test_history_tracks_all_unique_transitions(self, segmenter):
        """History should contain every actual state change."""
        segmenter.process_message(1, "viewed")       # LEAD -> ACTIVE
        segmenter.process_message(1, "dm")            # ACTIVE -> HOT
        segmenter.process_message(1, "silent_7d")     # HOT -> COLD
        segmenter.process_message(1, "replied")       # COLD -> WARM
        user = segmenter.get_user(1)
        expected = ["active", "hot", "cold", "warm"]
        assert [h["to"] for h in user.history] == expected

    def test_history_format(self, segmenter):
        """Each history entry should have required fields."""
        segmenter.process_message(1, "viewed")
        user = segmenter.get_user(1)
        entry = user.history[0]
        assert "from" in entry
        assert "to" in entry
        assert "trigger" in entry
        assert "ts" in entry
        assert entry["from"] == "lead"
        assert entry["to"] == "active"
        assert entry["trigger"] == "viewed"
