# CMN-C1-091 — Unit Tests: Main Node

import json


from framework.schemas.agent_status import AgentStatus
from src.nodes.main_node import MainNode


def _mention(text, source="twitter", author=None, url=""):
    mention = {"text": text, "source": source, "url": url}
    if author:
        mention["author"] = author
    return mention


def _base_state(**overrides):
    state = {
        "query_id": "q-1",
        "brand_query": "Acme Coffee",
        "date_range_days": 7,
        "competitor_brands": "[]",
        "max_mentions": 500,
        "crisis_spike_threshold": 0.75,
    }
    state.update(overrides)
    return state


class _StubFeed:
    def __init__(self, mentions=None, error=None):
        self.mentions = mentions or []
        self.error = error
        self.calls = []

    def fetch(self, brand, date_range_days, limit):
        self.calls.append({"brand": brand, "date_range_days": date_range_days, "limit": limit})
        if self.error:
            raise self.error
        return self.mentions


class TestMainNodeIngestion:
    """BL-03 / BL-04."""

    def setup_method(self):
        self.node = MainNode()

    def test_ingestion_success(self):
        feed = _StubFeed(mentions=[_mention("Love the new blend, excellent quality")])
        result = self.node.execute(_base_state(config={"configurable": {"mention_feed": feed}}))

        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["mention_count"] == 1
        assert feed.calls[0]["brand"] == "Acme Coffee"

    def test_no_feed_returns_no_evidence(self):
        result = self.node.execute(_base_state())
        assert result["mention_count"] == 0
        assert json.loads(result["mentions"]) == []

    def test_mentions_are_capped_at_max(self):
        feed = _StubFeed(mentions=[_mention(f"mention {i}") for i in range(50)])
        result = self.node.execute(_base_state(max_mentions=10, config={"configurable": {"mention_feed": feed}}))
        assert result["mention_count"] == 10

    def test_ingestion_failure_is_recoverable_and_hides_backend_detail(self):
        feed = _StubFeed(error=ConnectionError("private backend detail"))
        result = self.node.execute(_base_state(config={"configurable": {"mention_feed": feed}}))

        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["recoverable_error_code"] == "ingestion_unavailable"
        assert "private backend detail" not in result["error_message"]

    def test_upstream_error_short_circuits(self):
        assert self.node.execute({"error_message": "boom"})["status"] == AgentStatus.ERROR.value


class TestMainNodeSentimentAndCrisisGate:
    """BL-05: sentiment scoring and the non-suppressible crisis gate."""

    def setup_method(self):
        self.node = MainNode()

    def test_sentiment_counts_are_computed(self):
        feed = _StubFeed(
            mentions=[
                _mention("Absolutely love this, excellent quality"),
                _mention("Terrible experience, awful service"),
                _mention("Bought a bag today"),
            ]
        )
        result = self.node.execute(_base_state(config={"configurable": {"mention_feed": feed}}))
        summary = json.loads(result["sentiment_summary"])

        assert summary["positive"] == 1
        assert summary["negative"] == 1
        assert summary["neutral"] == 1

    def test_crisis_requires_both_volume_and_negativity(self):
        """Negativity alone at low volume is churn, not a crisis."""
        feed = _StubFeed(mentions=[_mention("terrible awful scam") for _ in range(3)])
        result = self.node.execute(_base_state(config={"configurable": {"mention_feed": feed}}))
        assert result["crisis_detected"] is False

    def test_high_volume_negative_wave_triggers_the_crisis_gate(self):
        feed = _StubFeed(mentions=[_mention("terrible awful scam boycott") for _ in range(120)])
        result = self.node.execute(_base_state(config={"configurable": {"mention_feed": feed}}))

        assert result["crisis_detected"] is True
        assert result["crisis_spike_score"] >= 0.75

    def test_high_volume_positive_wave_is_not_a_crisis(self):
        feed = _StubFeed(mentions=[_mention("excellent love amazing") for _ in range(120)])
        result = self.node.execute(_base_state(config={"configurable": {"mention_feed": feed}}))
        assert result["crisis_detected"] is False


class TestMainNodeRedactionGate:
    """BL-06: the non-suppressible personal-data redaction gate."""

    def setup_method(self):
        self.node = MainNode()

    def test_social_handles_are_redacted(self):
        feed = _StubFeed(mentions=[_mention("@angry_customer said the coffee was terrible")])
        result = self.node.execute(_base_state(config={"configurable": {"mention_feed": feed}}))

        assert "@angry_customer" not in result["mentions"]
        assert "[REDACTED]" in result["mentions"]
        assert result["redaction_count"] >= 1

    def test_author_field_is_always_redacted(self):
        feed = _StubFeed(mentions=[_mention("Great coffee", author="Jane Doe")])
        result = self.node.execute(_base_state(config={"configurable": {"mention_feed": feed}}))

        assert "Jane Doe" not in result["mentions"]

    def test_redaction_runs_before_mentions_reach_state(self):
        """No un-redacted handle may be persisted, since State is checkpointable."""
        feed = _StubFeed(mentions=[_mention("@someone complained")])
        result = self.node.execute(_base_state(config={"configurable": {"mention_feed": feed}}))
        assert "@someone" not in json.dumps(result)


class TestMainNodeCompetitors:
    """BL-07."""

    def test_competitor_scores_are_computed(self):
        feed = _StubFeed(
            mentions=[
                _mention("Globex Coffee is terrible and awful"),
                _mention("Acme Coffee is excellent"),
            ]
        )
        result = MainNode().execute(
            _base_state(
                competitor_brands=json.dumps(["Globex Coffee"]),
                config={"configurable": {"mention_feed": feed}},
            )
        )
        scores = json.loads(result["competitor_scores"])

        assert scores["Globex Coffee"]["mention_count"] == 1
        assert scores["Globex Coffee"]["negative_ratio"] == 1.0

    def test_brand_shaped_competitor_name_still_matches(self):
        """Regression: the PII detector masks a brand name as a person name.

        Scoring against the redacted text would report zero mentions for every
        competitor, so scoring runs on the pre-redaction text and returns counts
        only.
        """
        feed = _StubFeed(mentions=[_mention("Globex Coffee tastes terrible") for _ in range(3)])
        result = MainNode().execute(
            _base_state(
                competitor_brands=json.dumps(["Globex Coffee"]),
                config={"configurable": {"mention_feed": feed}},
            )
        )
        assert json.loads(result["competitor_scores"])["Globex Coffee"]["mention_count"] == 3

    def test_competitor_scores_carry_no_mention_text(self):
        feed = _StubFeed(mentions=[_mention("@someone says Globex Coffee is terrible")])
        result = MainNode().execute(
            _base_state(
                competitor_brands=json.dumps(["Globex Coffee"]),
                config={"configurable": {"mention_feed": feed}},
            )
        )
        assert "@someone" not in result["competitor_scores"]

    def test_no_competitors_yields_empty_scores(self):
        result = MainNode().execute(_base_state())
        assert json.loads(result["competitor_scores"]) == {}


class TestMainNodeAuditLogging:
    """TC-11 / PB-1."""

    def test_success_path_emits_domain_event(self, monkeypatch):
        import src.nodes.main_node as module

        events = []
        monkeypatch.setattr(module, "emit_trace_event", lambda name, payload, state: events.append(name))
        MainNode().execute(_base_state())
        assert "main_completed" in events

    def test_ingestion_error_emits_domain_event(self, monkeypatch):
        import src.nodes.main_node as module

        events = []
        monkeypatch.setattr(module, "emit_trace_event", lambda name, payload, state: events.append(name))
        feed = _StubFeed(error=TimeoutError("slow"))
        MainNode().execute(_base_state(config={"configurable": {"mention_feed": feed}}))
        assert "main_ingestion_error" in events
