"""Pre-process node for BrandSentimentReportAgent.

Sanitizes the report request, extracts the brand under analysis, the reporting
window, and any competitor brands. Blocks prompt-injection attempts (S-2).
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, ClassVar

from framework.errors import SecurityViolationError
from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

_INJECTION_PATTERNS = [
    re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<[^>]+>"),
    re.compile(r"javascript:", re.IGNORECASE),
    re.compile(r"on\w{1,20}\s{0,4}=", re.IGNORECASE),
]

# S-2: attempts to steer the report-writing LLM away from the retrieved
# evidence. Every quantifier is bounded so a hostile input cannot force
# quadratic-time backtracking.
_PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s{1,4}(?:all\s{1,4}){0,1}previous\s{1,4}instructions", re.IGNORECASE),
    re.compile(r"disregard\s{1,4}(?:all\s{1,4}){0,1}(?:the\s{1,4}){0,1}(?:above|prior)", re.IGNORECASE),
    re.compile(r"system\s{0,4}prompt", re.IGNORECASE),
    re.compile(r"you\s{1,4}are\s{1,4}now", re.IGNORECASE),
    re.compile(r"forget\s{1,4}(?:everything|all)\s{1,4}(?:above|before)", re.IGNORECASE),
]

_DATE_RANGE_RE = re.compile(r"\b(?:last|past)\s{1,3}(\d{1,3})\s{1,3}(day|week|month)s?\b", re.IGNORECASE)
_WINDOW_MULTIPLIER = {"day": 1, "week": 7, "month": 30}
_MAX_DATE_RANGE_DAYS = 365

_COMPETITOR_RE = re.compile(
    r"(?:compared?\s{1,3}(?:against|to|with)|versus|vs\.?)\s{1,3}(.{1,200}?)(?:\.|$)",
    re.IGNORECASE,
)
_BRAND_SPLIT_RE = re.compile(r"\s{0,3}(?:,|/|\band\b|&)\s{0,3}", re.IGNORECASE)

_BRAND_RE = re.compile(
    r"(?:report|sentiment|mentions?|monitoring|analysis)\s{1,3}(?:for|on|about)\s{1,3}(.{1,80}?)"
    r"(?=\s{1,3}(?:over|during|in|for|compared?|versus|vs\.?)\b|\.|$)",
    re.IGNORECASE,
)


class PreProcessNode(FunctionNode):
    """Sanitize the request and extract brand, window, and competitors."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.INTERNAL

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        if state.get("error_message"):
            result = {"status": AgentStatus.ERROR.value}
            emit_trace_event("pre_process_skipped", {"reason": "upstream_error"}, result)
            return result

        raw_query = str(state.get("raw_query") or state.get("user_input") or "")
        cfg = (state.get("config") or {}).get("configurable", {})
        agent_config = cfg.get("agent_config", cfg)
        max_input_length = int(state.get("max_input_length") or agent_config.get("max_input_length", 200))

        if len(raw_query) > max_input_length:
            result = self._input_error(
                raw_query,
                f"The request is too long ({len(raw_query)} characters). The maximum is {max_input_length} characters.",
            )
            emit_trace_event("pre_process_rejected", {"reason": "input_too_long"}, result)
            return result

        normalized_query = self._sanitize(raw_query)
        if not normalized_query.strip():
            if raw_query.strip():
                result = {
                    "raw_query": raw_query,
                    "error_message": "S-1 violation: request is empty after sanitization",
                    "status": AgentStatus.ERROR.value,
                }
            else:
                result = self._input_error(raw_query, "No usable brand reporting request was provided.")
            emit_trace_event("pre_process_rejected", {"reason": "empty_after_sanitization"}, result)
            return result

        if len(normalized_query) < 10:
            result = self._input_error(
                normalized_query,
                "The request is too short to identify a brand to report on.",
            )
            emit_trace_event("pre_process_rejected", {"reason": "input_too_short"}, result)
            return result

        brand_query = self._extract_brand(normalized_query)
        if not brand_query:
            result = self._input_error(
                normalized_query,
                "No brand name could be identified in the request.",
            )
            emit_trace_event("pre_process_rejected", {"reason": "no_brand_identified"}, result)
            return result

        query_id = str(uuid.uuid4())
        date_range_days = self._extract_date_range(
            normalized_query,
            int(state.get("date_range_days") or agent_config.get("date_range_days", 7)),
        )
        competitor_brands = self._extract_competitors(normalized_query, brand_query)

        emit_trace_event(
            "pre_process_completed",
            {
                "query_id": query_id,
                "date_range_days": date_range_days,
                "competitor_count": len(competitor_brands),
            },
            state,
        )

        return {
            "raw_query": raw_query,
            "query_id": query_id,
            "normalized_query": normalized_query,
            "brand_query": brand_query,
            "date_range_days": date_range_days,
            "competitor_brands": json.dumps(competitor_brands),
            "error_message": None,
            "status": AgentStatus.SUCCESS.value,
        }

    def _extra_security_gate_input(self, state: dict[str, Any]) -> dict[str, Any]:
        """S-2: block attempts to steer the report LLM away from the evidence."""
        query = str(state.get("raw_query") or state.get("user_input") or "")
        if any(pattern.search(query) for pattern in _PROMPT_INJECTION_PATTERNS):
            raise SecurityViolationError("PreProcessNode: blocked prompt-injection pattern in the report request")
        return state

    @staticmethod
    def _input_error(raw_query: str, message: str) -> dict[str, Any]:
        return {
            "raw_query": raw_query,
            "error_message": message,
            "input_error_message": message,
            "input_error_guidance": [
                'Name the brand, for example: Weekly sentiment report for "Acme Coffee".',
                "Optionally add a window such as over the last 7 days, and competitors to compare against.",
            ],
            "status": AgentStatus.SUCCESS.value,
        }

    @staticmethod
    def _sanitize(text: str) -> str:
        result = text
        for pattern in _INJECTION_PATTERNS:
            result = pattern.sub("", result)
        return " ".join(result.split())

    @staticmethod
    def _extract_brand(query: str) -> str:
        quoted = re.search(r"[\"'‘’“”]([^\"'‘’“”]{1,80})", query)
        if quoted:
            return quoted.group(1).strip()
        match = _BRAND_RE.search(query)
        if match:
            return match.group(1).strip()
        return ""

    @staticmethod
    def _extract_date_range(query: str, default_days: int) -> int:
        match = _DATE_RANGE_RE.search(query)
        if not match:
            return default_days
        value = int(match.group(1))
        days = value * _WINDOW_MULTIPLIER.get(match.group(2).lower(), 1)
        # A window beyond the retention horizon silently returns nothing, so
        # clamp rather than issuing an unbounded ingestion request.
        return max(1, min(days, _MAX_DATE_RANGE_DAYS))

    @staticmethod
    def _extract_competitors(query: str, brand_query: str) -> list[str]:
        match = _COMPETITOR_RE.search(query)
        if not match:
            return []
        competitors = []
        for candidate in _BRAND_SPLIT_RE.split(match.group(1)):
            name = candidate.strip(" \"'.,")
            if name and name.lower() != brand_query.lower() and name not in competitors:
                competitors.append(name)
        return competitors
