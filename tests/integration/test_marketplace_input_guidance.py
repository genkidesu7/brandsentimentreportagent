# CMN-C1-091 — Integration: Marketplace output shaping and safe degradation

import json

from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

from src.graph.graph import Graph
from src.nodes.main_node import MainNode
from src.nodes.post_process_node import PostProcessNode

_REPORT_REQUEST = 'Weekly sentiment report for "Acme Coffee" over the last 7 days'


class _ReportLLM:
    def generate(self, system_prompt, user_prompt):
        return "## Brand sentiment report\n\nSentiment improved across the reporting window."


class _LeakyLLM:
    """Reintroduces a handle the redaction gate had already stripped."""

    def generate(self, system_prompt, user_prompt):
        return "## Brand sentiment report\n\nThe spike was driven by @angry_customer."


class _FeedMainNode(MainNode):
    @staticmethod
    def _ingest(brand, date_range_days, limit, config):
        return [
            {
                "text": "@angry_customer says the coffee is terrible",
                "source": "twitter",
                "author": "Jane Doe",
                "url": "https://x/1",
            }
        ]


class _UnavailableMainNode(MainNode):
    @staticmethod
    def _ingest(brand, date_range_days, limit, config):
        raise ConnectionError("private backend detail")


class _ReportGraph(Graph):
    def register_nodes(self):
        super().register_nodes()
        self._nodes["main"] = _FeedMainNode()
        self._nodes["post_process"] = PostProcessNode(llm=_ReportLLM())


class _LeakyGraph(Graph):
    def register_nodes(self):
        super().register_nodes()
        self._nodes["main"] = _FeedMainNode()
        self._nodes["post_process"] = PostProcessNode(llm=_LeakyLLM())


class _UnavailableGraph(Graph):
    def register_nodes(self):
        super().register_nodes()
        self._nodes["main"] = _UnavailableMainNode()


def _invoke(graph, query=_REPORT_REQUEST):
    return graph.invoke(
        query,
        ctx=InvocationContext(caller_trust_level=TrustLevel.INTERNAL),
        input_context={"conversation_history": []},
    )


def test_short_marketplace_request_returns_readable_guidance():
    graph = Graph(config={})
    graph.compile()

    result = _invoke(graph, "Hello")

    assert result["status"] == "success"
    assert result["output"].startswith("The brand sentiment report could not be produced.")
    assert "How to continue:" in result["output"]


def test_success_marketplace_output_is_report_but_api_stays_json():
    graph = Graph(config={})
    formatted_output = json.dumps({"answer": "Sentiment improved.", "citations": [], "confidence": "high"})

    api_result = graph.get_output({"formatted_output": formatted_output})
    marketplace_result = graph.get_output(
        {"formatted_output": formatted_output, "input_context": {"conversation_history": []}}
    )

    assert api_result["output"] == formatted_output
    assert marketplace_result["output"] == "Sentiment improved."


def test_marketplace_loads_runtime_config_by_default():
    graph = Graph()
    graph.register_nodes()

    assert graph._extra_initial_state() == {
        "max_mentions": 500,
        "crisis_spike_threshold": 0.75,
        "date_range_days": 7,
        "response_budget": 2000,
        "max_input_length": 200,
    }
    assert graph._nodes["post_process"]._llm_timeout_s == 30
    assert graph._nodes["post_process"]._llm_max_retry == 3


def test_personal_data_never_reaches_the_marketplace_output():
    graph = _ReportGraph(config={})
    graph.compile()

    result = _invoke(graph)

    assert result["status"] == AgentStatus.SUCCESS
    assert "@angry_customer" not in result["output"]
    assert "Jane Doe" not in result["output"]


def test_report_reintroducing_a_handle_is_withheld():
    """End-to-end proof that the S-3 gate withholds rather than releases."""
    graph = _LeakyGraph(config={})
    graph.compile()

    result = _invoke(graph)

    assert result["status"] == AgentStatus.SUCCESS
    assert "@angry_customer" not in result["output"]
    assert "withheld" in result["output"].lower()


def test_ingestion_outage_returns_guidance_without_failing_invocation():
    graph = _UnavailableGraph(config={})
    graph.compile()

    result = _invoke(graph)

    assert result["status"] == AgentStatus.SUCCESS
    assert "temporarily unavailable" in result["output"].lower()
    assert "private backend detail" not in result["output"]


def test_non_suppressible_gates_cannot_be_disabled_by_config():
    """The two mandatory gates reject a config that claims to turn them off.

    Validation runs at compile(), not construction, so the graph never reaches a
    servable state with either gate nominally disabled.
    """
    import pytest
    from framework.errors import SecurityViolationError

    for key in ("crisis_gate_suppressible", "redaction_gate_suppressible"):
        graph = Graph(config={key: True})
        with pytest.raises(SecurityViolationError, match=key):
            graph.compile()


def test_gates_stay_enabled_when_config_sets_them_false():
    """`False` is the only accepted value — it matches actual behaviour."""
    graph = Graph(config={"crisis_gate_suppressible": False, "redaction_gate_suppressible": False})
    graph.compile()
    assert graph._nodes["main"] is not None
