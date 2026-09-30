"""Post-process node for BrandSentimentReportAgent.

Assembles the brand intelligence report from the scored, redacted mentions and
applies the S-3 output gate. The gate re-verifies that no personal data reached
the report text and withholds the report if any did.
"""

from __future__ import annotations

import json
import re
from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.security.credential_detector import detect_credentials
from shared.security.pii_detector import detect_pii
from shared.utils.audit_logger import emit_trace_event
from src.services.llm_runtime import complete_text

_HANDLE_PATTERN = re.compile(r"@\w{1,50}")


def _emit_progress(message: str, metadata: dict[str, Any]) -> None:
    """Emit Marketplace progress when the platform event transport is available."""
    try:
        from shared.services.events import emitter
        from shared.services.events.types import EventType
    except (ImportError, ModuleNotFoundError):
        # Lightweight local AgentCore installations may omit Marketplace events.
        # Synthesis and its terminal fallback must remain available regardless.
        return
    emitter().emit_event(event_type=EventType.PROGRESS_UPDATE, message=message, metadata=metadata)


_SYSTEM_PROMPT = (
    "You are a brand intelligence analyst writing a weekly report for a brand team.\n"
    "Given aggregate sentiment metrics and a sample of already-redacted brand mentions, "
    "write a structured report.\n\n"
    "Your report MUST:\n"
    "1. Open with a summary of overall sentiment and mention volume for the period\n"
    "2. Call out a crisis signal prominently when one is flagged, with the observed figures\n"
    "3. Compare competitor sentiment when competitor figures are supplied\n"
    "4. End with concrete recommended actions\n"
    "5. Use only the supplied figures -- never invent mention counts, scores, or quotes\n"
    "6. Never reproduce a social handle, personal name, or contact detail, and never "
    "reverse a [REDACTED] marker\n\n"
    "Use Markdown headings. Keep it suitable for circulation to management."
)

_NO_DATA_REPORT = (
    "## No brand mentions found\n\n"
    "No mentions matching this brand were found for the requested period. Verify the brand "
    "name and the reporting window, then try again.\n\n"
    "_Azure OpenAI synthesis was temporarily unavailable; this response is based on "
    "ingestion status only._"
)

_INGESTION_UNAVAILABLE_REPORT = (
    "## Brand mention feed temporarily unavailable\n\n"
    "I couldn't reach the mention feed, so I can't produce a sentiment report or confirm "
    "whether a crisis signal is present right now. Please try again shortly. If the problem "
    "continues, contact support and include the reference below."
)

_WITHHELD_REPORT = (
    "## Report withheld\n\n"
    "The generated report did not pass the personal-data output check, so it has not been "
    "released. This is a safety outcome, not a failure of the analysis. Re-run the report; "
    "if the problem persists, contact support with the reference above."
)


class PostProcessNode(FunctionNode):
    """Generate the brand report and enforce the personal-data output gate."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.INTERNAL

    def __init__(
        self,
        llm: Any | None = None,
        *,
        llm_timeout_s: float = 30.0,
        llm_max_retry: int = 3,
    ) -> None:
        super().__init__()
        # Unit-test seam only. Production resolves Azure OpenAI from the
        # invocation-scoped secrets declared by the agent manifest.
        self._llm = llm
        self._llm_timeout_s = llm_timeout_s
        self._llm_max_retry = llm_max_retry

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        if state.get("input_error_message"):
            message = str(state["input_error_message"])
            guidance = state.get("input_error_guidance", [])
            return {
                "answer": message,
                "citations": "[]",
                "confidence": "low",
                "validation_passed": True,
                "formatted_output": json.dumps(
                    {"status": "input_error", "message": message, "guidance": guidance},
                    ensure_ascii=False,
                ),
                "status": AgentStatus.SUCCESS.value,
            }

        if state.get("recoverable_error_code") == "ingestion_unavailable":
            answer = _INGESTION_UNAVAILABLE_REPORT
            trace_id = str(state.get("trace_id") or "").strip()
            if trace_id:
                answer = f"{answer}\n\nReference: `{trace_id}`"
            emit_trace_event(
                "post_process_ingestion_unavailable",
                {"query_id": state.get("query_id"), "trace_id": trace_id},
                state,
            )
            return {
                "answer": answer,
                "citations": "[]",
                "confidence": "low",
                "validation_passed": True,
                "formatted_output": json.dumps(
                    {
                        "status": "temporarily_unavailable",
                        "answer": answer,
                        "citations": [],
                        "confidence": "low",
                        "mention_count": 0,
                    },
                    ensure_ascii=False,
                ),
                "error_message": None,
                "recoverable_error_code": None,
                "status": AgentStatus.SUCCESS.value,
            }

        if state.get("error_message"):
            result = {"status": AgentStatus.ERROR.value}
            emit_trace_event("post_process_skipped", {"reason": "upstream_error"}, result)
            return result

        cfg = (state.get("config") or {}).get("configurable", {})
        agent_config = cfg.get("agent_config", cfg)
        llm_max_tokens = int(state.get("response_budget") or agent_config.get("llm_max_tokens", 2000))

        mentions = self._load_json_list(state.get("mentions"))
        summary = self._load_json_dict(state.get("sentiment_summary"))
        competitor_scores = self._load_json_dict(state.get("competitor_scores"))
        crisis_detected = bool(state.get("crisis_detected"))
        crisis_score = state.get("crisis_spike_score", 0.0)

        if not mentions:
            return self._no_data(state, llm_max_tokens)

        user_prompt = (
            f"Brand: {state.get('brand_query', '')}\n"
            f"Reporting window: last {state.get('date_range_days', 7)} days\n"
            f"Sentiment summary: {json.dumps(summary, ensure_ascii=False)}\n"
            f"Crisis signal: {'YES' if crisis_detected else 'no'} (score {crisis_score})\n"
            f"Competitor figures: {json.dumps(competitor_scores, ensure_ascii=False) or 'none supplied'}\n\n"
            f"---\nRedacted mention sample:\n{self._build_sample(mentions)}"
        )

        _emit_progress(
            "Writing the brand intelligence report with Azure OpenAI...",
            {"stage": "llm_synthesis", "mention_count": len(mentions)},
        )
        try:
            answer = self._generate(state, _SYSTEM_PROMPT, user_prompt, llm_max_tokens)
            synthesis_mode = "azure_openai"
            _emit_progress(
                "Azure OpenAI report synthesis completed.",
                {"stage": "llm_synthesis", "status": "completed", "mention_count": len(mentions)},
            )
        except Exception as exc:
            emit_trace_event(
                "post_process_llm_fallback",
                {"query_id": state.get("query_id"), "reason": type(exc).__name__},
                state,
            )
            answer = self._fallback_report(state, summary, competitor_scores, crisis_detected, crisis_score)
            synthesis_mode = "evidence_only"
            _emit_progress(
                "Azure OpenAI synthesis is temporarily unavailable. Returning the metrics report only.",
                {"stage": "llm_synthesis", "status": "error", "mention_count": len(mentions)},
            )

        citations = self._extract_citations(mentions)
        confidence = self._compute_confidence(mentions)

        emit_trace_event(
            "post_process_completed",
            {
                "query_id": state.get("query_id"),
                "mention_count": len(mentions),
                "citations_count": len(citations),
                "crisis_detected": crisis_detected,
                "confidence": confidence,
            },
            state,
        )

        return {
            "answer": answer,
            "citations": json.dumps(citations),
            "confidence": confidence,
            "synthesis_mode": synthesis_mode,
            "validation_passed": True,
            "formatted_output": json.dumps(
                {
                    "answer": answer,
                    "citations": citations,
                    "confidence": confidence,
                    "mention_count": len(mentions),
                    "crisis_detected": crisis_detected,
                    "synthesis_mode": synthesis_mode,
                },
                ensure_ascii=False,
            ),
            "error_message": None,
            "status": AgentStatus.SUCCESS.value,
        }

    def _no_data(self, state: dict[str, Any], llm_max_tokens: int) -> dict[str, Any]:
        _emit_progress(
            "Generating the report with Azure OpenAI...",
            {"stage": "llm_synthesis", "mention_count": 0},
        )
        prompt = (
            f"Brand: {state.get('brand_query', '')}\n"
            f"Reporting window: last {state.get('date_range_days', 7)} days\n\n"
            "No mentions were ingested for this brand and period. Explain this concisely and "
            "suggest what the analyst should verify."
        )
        try:
            answer = self._generate(state, _SYSTEM_PROMPT, prompt, min(llm_max_tokens, 400))
            synthesis_mode = "azure_openai"
            _emit_progress(
                "Azure OpenAI report synthesis completed.",
                {"stage": "llm_synthesis", "status": "completed", "mention_count": 0},
            )
        except Exception as exc:
            answer = _NO_DATA_REPORT
            synthesis_mode = "evidence_only"
            _emit_progress(
                "Azure OpenAI synthesis is temporarily unavailable. Returning ingestion status only.",
                {"stage": "llm_synthesis", "status": "error", "mention_count": 0},
            )
            emit_trace_event("post_process_llm_fallback", {"reason": type(exc).__name__}, state)

        emit_trace_event(
            "post_process_no_results",
            {"query_id": state.get("query_id"), "confidence": "low"},
            state,
        )
        return {
            "answer": answer,
            "citations": "[]",
            "confidence": "low",
            "synthesis_mode": synthesis_mode,
            "validation_passed": True,
            "formatted_output": json.dumps(
                {
                    "answer": answer,
                    "citations": [],
                    "confidence": "low",
                    "mention_count": 0,
                    "synthesis_mode": synthesis_mode,
                },
                ensure_ascii=False,
            ),
            "error_message": None,
            "status": AgentStatus.SUCCESS.value,
        }

    def _generate(self, state: dict[str, Any], system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        config = state.get("config")
        configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
        injected = self._llm or configurable.get("llm")
        return complete_text(
            state,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            injected,
            max_tokens=max_tokens,
            timeout_s=self._llm_timeout_s,
            max_retry=self._llm_max_retry,
        )

    @staticmethod
    def _fallback_report(
        state: dict[str, Any],
        summary: dict[str, Any],
        competitor_scores: dict[str, Any],
        crisis_detected: bool,
        crisis_score: Any,
    ) -> str:
        lines = [
            f"## Brand sentiment report — {state.get('brand_query', 'unknown brand')}",
            "",
            "LLM synthesis was unavailable; the measured figures follow.",
            "",
            f"- Reporting window: last {state.get('date_range_days', 7)} days",
            f"- Mentions analysed: {summary.get('mention_count', 0)}",
            f"- Positive / negative / neutral: {summary.get('positive', 0)} / "
            f"{summary.get('negative', 0)} / {summary.get('neutral', 0)}",
            f"- Negative ratio: {summary.get('negative_ratio', 0.0)}",
        ]
        if crisis_detected:
            lines.extend(["", f"### Crisis signal detected (score {crisis_score})", "Escalate to the brand lead."])
        if competitor_scores:
            lines.extend(["", "### Competitor comparison"])
            lines.extend(
                f"- {brand}: {figures.get('mention_count', 0)} mentions, "
                f"negative ratio {figures.get('negative_ratio', 0.0)}"
                for brand, figures in competitor_scores.items()
            )
        return "\n".join(lines)

    @staticmethod
    def _build_sample(mentions: list[dict[str, Any]], limit: int = 20) -> str:
        return "\n".join(
            f"[{index}] ({mention.get('source', 'unknown')}) {str(mention.get('text', ''))[:300]}"
            for index, mention in enumerate(mentions[:limit], 1)
        )

    @staticmethod
    def _extract_citations(mentions: list[dict[str, Any]]) -> list[dict[str, str]]:
        seen: set[str] = set()
        citations = []
        for mention in mentions:
            source = str(mention.get("source") or "")
            if not source or source in seen:
                continue
            seen.add(source)
            citations.append({"source": source, "url": str(mention.get("url") or "")})
        return citations

    @staticmethod
    def _compute_confidence(mentions: list[dict[str, Any]]) -> str:
        """Confidence tracks sample size — a report from five mentions is weak."""
        if len(mentions) >= 100:
            return "high"
        if len(mentions) >= 20:
            return "medium"
        return "low"

    @staticmethod
    def _load_json_list(value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if not isinstance(value, str):
            return []
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return []
        return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []

    @staticmethod
    def _load_json_dict(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if not isinstance(value, str):
            return {}
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _extra_security_gate_output(self, result: dict[str, Any]) -> dict[str, Any]:
        """S-3: non-suppressible personal-data check on the generated report.

        The redaction gate already stripped handles and PII from the mentions
        before they reached the prompt. This gate verifies the model did not
        reintroduce any, and withholds the report if it did — a report naming a
        real member of the public must not leave the agent even once.
        """
        answer = result.get("answer", "")
        if not answer:
            return result

        if detect_credentials(answer):
            emit_trace_event("post_process_s3_credential_blocked", {}, result)
            raise RuntimeError("S-3 blocked: credential detected in output")

        findings = [f"handle: {handle}" for handle in _HANDLE_PATTERN.findall(answer)]
        findings.extend(f"pii: {finding.get('type')}" for finding in detect_pii(answer))

        if findings:
            emit_trace_event(
                "post_process_s3_report_withheld",
                {"finding_count": len(findings)},
                result,
            )
            result["answer"] = _WITHHELD_REPORT
            result["validation_passed"] = False
            result["validation_findings"] = json.dumps(findings, ensure_ascii=False)
            result["formatted_output"] = json.dumps(
                {
                    "status": "withheld",
                    "answer": _WITHHELD_REPORT,
                    "citations": [],
                    "confidence": "low",
                    "validation_passed": False,
                },
                ensure_ascii=False,
            )
            return result

        result["validation_passed"] = True
        result["validation_findings"] = "[]"
        return result
