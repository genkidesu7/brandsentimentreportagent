# CMN-C1-091 — Unit Tests: PostProcess Node

import json

import pytest

from framework.schemas.agent_status import AgentStatus
from src.nodes.post_process_node import PostProcessNode


class _StubLLM:
    def __init__(self, answer="## Brand sentiment report\n\nOverall sentiment is positive."):
        self.answer = answer
        self.calls = []

    def complete(self, messages):
        self.calls.append(messages)
        return {"content": self.answer}


class _FailingLLM:
    def complete(self, messages):
        raise RuntimeError("provider unavailable")


def _mentions(count=3):
    return [
        {"text": f"Mention {index} about the brand", "source": "twitter", "url": f"https://x/{index}"}
        for index in range(count)
    ]


def _state(**overrides):
    state = {
        "query_id": "q-1",
        "brand_query": "Acme Coffee",
        "date_range_days": 7,
        "mentions": json.dumps(_mentions()),
        "mention_count": 3,
        "sentiment_summary": json.dumps({"mention_count": 3, "positive": 2, "negative": 1, "negative_ratio": 0.33}),
        "competitor_scores": "{}",
        "crisis_detected": False,
        "crisis_spike_score": 0.1,
        "response_budget": 2000,
    }
    state.update(overrides)
    return state


class TestPostProcessReportGeneration:
    """BL-08 / BL-09."""

    def test_report_is_generated_from_metrics(self):
        result = PostProcessNode(llm=_StubLLM()).execute(_state())
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["synthesis_mode"] == "azure_openai"
        assert "Brand sentiment report" in result["answer"]

    def test_metrics_and_crisis_flag_reach_the_prompt(self):
        llm = _StubLLM()
        PostProcessNode(llm=llm).execute(_state(crisis_detected=True, crisis_spike_score=0.9))

        user_prompt = llm.calls[0][1]["content"]
        assert "Acme Coffee" in user_prompt
        assert "YES" in user_prompt
        assert "0.9" in user_prompt

    def test_provider_failure_degrades_to_measured_figures(self):
        result = PostProcessNode(llm=_FailingLLM()).execute(_state())
        assert result["synthesis_mode"] == "evidence_only"
        assert "Mentions analysed: 3" in result["answer"]

    def test_fallback_report_still_flags_a_crisis(self):
        """A crisis must remain visible even when synthesis is unavailable."""
        result = PostProcessNode(llm=_FailingLLM()).execute(_state(crisis_detected=True, crisis_spike_score=0.88))
        assert "Crisis signal detected" in result["answer"]

    def test_no_mentions_reports_no_data(self):
        result = PostProcessNode(llm=_FailingLLM()).execute(_state(mentions="[]"))
        assert result["confidence"] == "low"
        assert "No brand mentions found" in result["answer"]

    def test_input_error_is_surfaced_as_guidance(self):
        result = PostProcessNode(llm=_StubLLM()).execute(
            {"input_error_message": "No brand name", "input_error_guidance": ["Name the brand"]}
        )
        payload = json.loads(result["formatted_output"])
        assert payload["status"] == "input_error"

    def test_ingestion_outage_returns_actionable_report(self):
        result = PostProcessNode(llm=_StubLLM()).execute(
            {"recoverable_error_code": "ingestion_unavailable", "trace_id": "tr-9"}
        )
        assert "temporarily unavailable" in result["answer"].lower()
        assert "tr-9" in result["answer"]

    def test_upstream_error_short_circuits(self):
        assert PostProcessNode(llm=_StubLLM()).execute({"error_message": "boom"})["status"] == AgentStatus.ERROR.value


class TestPostProcessOutputShaping:
    """BL-10."""

    def test_citations_are_deduplicated_by_source(self):
        duplicated = [{"text": "x", "source": "twitter", "url": "u"} for _ in range(4)]
        result = PostProcessNode(llm=_StubLLM()).execute(_state(mentions=json.dumps(duplicated)))
        assert len(json.loads(result["citations"])) == 1

    @pytest.mark.parametrize("count,expected", [(120, "high"), (30, "medium"), (5, "low")])
    def test_confidence_tracks_sample_size(self, count, expected):
        assert PostProcessNode._compute_confidence(_mentions(count)) == expected


class TestPostProcessS3PersonalDataGate:
    """TC-10 / BL-11: the non-suppressible personal-data output gate."""

    def setup_method(self):
        self.node = PostProcessNode(llm=_StubLLM())

    def _gate(self, answer):
        return self.node._extra_security_gate_output(
            {
                "answer": answer,
                "citations": "[]",
                "confidence": "medium",
                "formatted_output": json.dumps({"answer": answer}),
            }
        )

    def test_clean_report_passes(self):
        answer = "## Report\n\nSentiment improved this week across 120 mentions."
        gated = self._gate(answer)
        assert gated["validation_passed"] is True
        assert gated["answer"] == answer

    def test_report_containing_a_handle_is_withheld(self):
        gated = self._gate("## Report\n\n@angry_customer drove the negative spike.")
        assert gated["validation_passed"] is False
        assert "withheld" in gated["answer"].lower()
        assert "@angry_customer" not in gated["answer"]

    def test_report_containing_personal_data_is_withheld(self):
        gated = self._gate("## Report\n\nContact jane.doe@example.com about the complaint.")
        assert gated["validation_passed"] is False
        assert "jane.doe@example.com" not in gated["answer"]

    def test_withheld_report_records_findings(self):
        gated = self._gate("## Report\n\n@someone complained loudly.")
        assert json.loads(gated["validation_findings"])

    def test_withheld_formatted_output_is_replaced(self):
        gated = self._gate("## Report\n\n@someone complained loudly.")
        payload = json.loads(gated["formatted_output"])
        assert payload["status"] == "withheld"
        assert payload["validation_passed"] is False

    def test_credential_in_report_blocks_the_output(self, monkeypatch):
        import src.nodes.post_process_node as module

        monkeypatch.setattr(module, "detect_credentials", lambda text: [{"type": "custom"}])
        with pytest.raises(RuntimeError, match="S-3 blocked"):
            self._gate("## Report\n\nnothing personal here")


class TestPostProcessAuditLogging:
    """TC-11 / PB-1."""

    def test_success_path_emits_domain_event(self, monkeypatch):
        import src.nodes.post_process_node as module

        events = []
        monkeypatch.setattr(module, "emit_trace_event", lambda name, payload, state: events.append(name))
        PostProcessNode(llm=_StubLLM()).execute(_state())
        assert "post_process_completed" in events

    def test_fallback_path_emits_domain_event(self, monkeypatch):
        import src.nodes.post_process_node as module

        events = []
        monkeypatch.setattr(module, "emit_trace_event", lambda name, payload, state: events.append(name))
        PostProcessNode(llm=_FailingLLM()).execute(_state())
        assert "post_process_llm_fallback" in events

    def test_withheld_report_emits_domain_event(self, monkeypatch):
        import src.nodes.post_process_node as module

        events = []
        monkeypatch.setattr(module, "emit_trace_event", lambda name, payload, state: events.append(name))
        PostProcessNode(llm=_StubLLM())._extra_security_gate_output(
            {"answer": "@someone complained", "formatted_output": "{}"}
        )
        assert "post_process_s3_report_withheld" in events
