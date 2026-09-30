"""Main node for BrandSentimentReportAgent.

Ingests brand mentions for the reporting window, scores sentiment, and applies
the two non-suppressible gates: crisis-spike detection and personal-data
redaction. Both run unconditionally, before any report text is assembled.
"""

from __future__ import annotations

import json
import re
from typing import Any, ClassVar, cast

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.security.pii_detector import detect_pii
from shared.utils.audit_logger import emit_trace_event

# Personal-data redaction gate: social handles are personal data under
# Japan's Act on the Protection of Personal Information, and mention text is
# third-party content this agent never authored.
_HANDLE_PATTERN = re.compile(r"@\w{1,50}")

_NEGATIVE_TERMS = frozenset(
    {
        "angry",
        "avoid",
        "awful",
        "bad",
        "boycott",
        "broken",
        "complaint",
        "defective",
        "delay",
        "disappointed",
        "disgusting",
        "fail",
        "failure",
        "fraud",
        "horrible",
        "lawsuit",
        "misleading",
        "never",
        "poor",
        "refund",
        "rude",
        "scam",
        "terrible",
        "unacceptable",
        "unsafe",
        "worst",
    }
)

_POSITIVE_TERMS = frozenset(
    {
        "amazing",
        "best",
        "brilliant",
        "delicious",
        "delighted",
        "excellent",
        "fantastic",
        "favourite",
        "favorite",
        "great",
        "happy",
        "impressed",
        "love",
        "perfect",
        "pleased",
        "quality",
        "recommend",
        "reliable",
        "satisfied",
        "smooth",
        "wonderful",
    }
)

# A spike needs both unusual volume and negative tone. Volume alone is a
# product launch; negativity alone at low volume is ordinary churn.
_BASELINE_MENTION_VOLUME = 100.0


class MainNode(FunctionNode):
    """Ingest mentions, score sentiment, and run both non-suppressible gates."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.INTERNAL

    def __init__(self, llm: Any | None = None) -> None:
        super().__init__()
        # Required scaffold injection point. Ingestion and scoring are
        # deterministic today; report synthesis consumes the client in
        # PostProcessNode.
        self._llm = llm

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        if state.get("input_error_message"):
            return {"status": AgentStatus.SUCCESS.value}

        if state.get("error_message"):
            result = {"status": AgentStatus.ERROR.value}
            emit_trace_event("main_skipped", {"reason": "upstream_error"}, result)
            return result

        cfg = (state.get("config") or {}).get("configurable", {})
        agent_config = cfg.get("agent_config", cfg)
        max_mentions = int(state.get("max_mentions") or agent_config.get("max_mentions_per_report", 500))
        threshold = float(state.get("crisis_spike_threshold") or agent_config.get("crisis_spike_threshold", 0.75))

        brand_query = str(state.get("brand_query") or "")
        competitors = self._load_json_list(state.get("competitor_brands"))

        try:
            mentions = self._ingest(
                brand=brand_query,
                date_range_days=int(state.get("date_range_days") or 7),
                limit=max_mentions,
                config=state.get("config"),
            )
        except Exception as exc:
            error_message = f"MainNode: the mention feed is temporarily unavailable ({type(exc).__name__})"
            emit_trace_event(
                "main_ingestion_error",
                {"query_id": state.get("query_id"), "error_type": type(exc).__name__},
                state,
            )
            return {
                "mentions": "[]",
                "mention_count": 0,
                "redaction_count": 0,
                "ingestion_source": "unavailable",
                "sentiment_summary": "{}",
                "crisis_spike_score": 0.0,
                "crisis_detected": False,
                "competitor_scores": "{}",
                "error_message": error_message,
                "recoverable_error_code": "ingestion_unavailable",
                # The failure is handled by PostProcessNode so Marketplace receives
                # an actionable response instead of treating the invocation itself
                # as failed.
                "status": AgentStatus.SUCCESS.value,
            }

        mentions = mentions[:max_mentions]
        summary = self._score(mentions)

        # NON-SUPPRESSIBLE gate 1 — crisis spike.
        crisis_spike_score = self._crisis_score(len(mentions), summary["negative_ratio"])
        crisis_detected = crisis_spike_score >= threshold

        # NON-SUPPRESSIBLE gate 2 — personal-data redaction. Runs before the
        # mentions reach the report prompt or State, so no un-redacted handle
        # is ever persisted or sent to the provider.
        redacted, redaction_count = self._redact(mentions)

        # Scored against the pre-redaction text: the PII detector treats a brand
        # name as a person name, so matching on redacted text would score every
        # competitor as zero mentions. Only the aggregate counts leave this call
        # — no mention text from `mentions` is carried into the result.
        competitor_scores = self._score_competitors(mentions, competitors)

        emit_trace_event(
            "main_completed",
            {
                "query_id": state.get("query_id"),
                "mention_count": len(redacted),
                "redaction_count": redaction_count,
                "crisis_detected": crisis_detected,
                "competitor_count": len(competitor_scores),
            },
            state,
        )

        return {
            "mentions": json.dumps(redacted),
            "mention_count": len(redacted),
            "redaction_count": redaction_count,
            "ingestion_source": "mention_feed" if redacted else "no_results",
            "sentiment_summary": json.dumps(summary),
            "crisis_spike_score": crisis_spike_score,
            "crisis_detected": crisis_detected,
            "competitor_scores": json.dumps(competitor_scores),
            "error_message": None,
            "recoverable_error_code": None,
            "status": AgentStatus.SUCCESS.value,
        }

    @staticmethod
    def _ingest(brand: str, date_range_days: int, limit: int, config: dict[str, Any] | None) -> list[dict[str, Any]]:
        configurable = (config or {}).get("configurable", {})
        feed = configurable.get("mention_feed")
        if feed is not None:
            return cast(
                list[dict[str, Any]],
                feed.fetch(brand=brand, date_range_days=date_range_days, limit=limit),
            )
        # No feed configured: report on no evidence rather than inventing
        # mentions. PostProcessNode turns this into an explicit no-data report.
        return []

    @staticmethod
    def _score(mentions: list[dict[str, Any]]) -> dict[str, Any]:
        """Lexicon scoring over mention text.

        Deterministic by design: the sentiment figures a report cites must be
        reproducible, and must survive the provider being unavailable.
        """
        if not mentions:
            return {"mention_count": 0, "positive": 0, "negative": 0, "neutral": 0, "negative_ratio": 0.0}

        positive = negative = 0
        for mention in mentions:
            score = MainNode._mention_polarity(str(mention.get("text", "")))
            if score > 0:
                positive += 1
            elif score < 0:
                negative += 1

        total = len(mentions)
        return {
            "mention_count": total,
            "positive": positive,
            "negative": negative,
            "neutral": total - positive - negative,
            "negative_ratio": round(negative / total, 3),
        }

    @staticmethod
    def _mention_polarity(text: str) -> int:
        tokens = {token.lower() for token in re.findall(r"[A-Za-z']{2,32}", text)}
        return len(tokens & _POSITIVE_TERMS) - len(tokens & _NEGATIVE_TERMS)

    @staticmethod
    def _crisis_score(mention_count: int, negative_ratio: float) -> float:
        """Combine volume and negativity — either alone is not a crisis."""
        volume_score = min(mention_count / _BASELINE_MENTION_VOLUME, 1.0)
        return round(volume_score * negative_ratio, 3)

    @staticmethod
    def _redact(mentions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        """Strip social handles and detected PII from mention text and author."""
        redacted: list[dict[str, Any]] = []
        count = 0
        for mention in mentions:
            safe = dict(mention)
            text = str(safe.get("text", ""))
            text, handle_hits = _HANDLE_PATTERN.subn("[REDACTED]", text)
            count += handle_hits
            for finding in reversed(detect_pii(text)):
                text = text[: finding["start"]] + "[REDACTED]" + text[finding["end"] :]
                count += 1
            safe["text"] = text
            if safe.get("author"):
                safe["author"] = "[REDACTED]"
                count += 1
            redacted.append(safe)
        return redacted, count

    @staticmethod
    def _score_competitors(mentions: list[dict[str, Any]], competitors: list[str]) -> dict[str, Any]:
        """Aggregate competitor counts from un-redacted text.

        Returns counts only, never mention text, so nothing personal escapes.
        """
        scores: dict[str, Any] = {}
        for brand in competitors:
            matching = [m for m in mentions if brand.lower() in str(m.get("text", "")).lower()]
            summary = MainNode._score(matching)
            scores[brand] = {
                "mention_count": summary["mention_count"],
                "negative_ratio": summary["negative_ratio"],
            }
        return scores

    @staticmethod
    def _load_json_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        if not isinstance(value, str):
            return []
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return []
        return [str(item) for item in parsed] if isinstance(parsed, list) else []

    def _extra_security_gate_input(self, state: dict[str, Any]) -> dict[str, Any]:
        """S-2: PII masking on the brand query before it reaches the mention feed."""
        raw = state.get("brand_query", "")
        if not raw:
            return state
        findings = detect_pii(raw)
        if findings:
            for finding in reversed(findings):
                raw = raw[: finding["start"]] + "[MASKED]" + raw[finding["end"] :]
            state["brand_query"] = raw
            emit_trace_event("pii_masked_brand_query", {}, state)
        return state
