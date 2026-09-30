"""Graph definition for CMN-C1-091 (fixed five-node pipeline)."""

import json
from pathlib import Path
from typing import Any, cast

from framework.errors import SecurityViolationError
from framework.graph.agent_base_graph import AgentBaseGraph
from framework.utils.config_loader import load_config
from src.nodes.main_node import MainNode
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


class Graph(AgentBaseGraph):
    """Brand sentiment and market signal reporting graph."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        # Marketplace constructs Graph() directly. Load the same runtime config
        # used by the standalone HTTP adapter so both entrypoints behave alike.
        runtime_config = config
        if runtime_config is None:
            runtime_config = load_config(str(_CONFIG_PATH)) if _CONFIG_PATH.exists() else {}
        super().__init__(config=runtime_config)

    @property
    def name(self) -> str:
        return "BrandSentimentReportAgent"

    @property
    def state_schema(self) -> type[State]:
        return State

    def _validate_config(self) -> None:
        """Reject any attempt to disable the two non-suppressible gates.

        The crisis-spike gate and the personal-data redaction gate run
        unconditionally inside the nodes. Config keys that claim to suppress
        them are rejected at construction rather than silently ignored, so a
        deployment cannot believe it turned them off.
        """
        super()._validate_config()
        for key, gate in (
            ("crisis_gate_suppressible", "the crisis-spike gate"),
            ("redaction_gate_suppressible", "the personal-data redaction gate"),
        ):
            if self.config.get(key) is True:
                raise SecurityViolationError(f"[Graph] '{key}' cannot be true — {gate} is non-suppressible.")

    def _extra_initial_state(self) -> dict[str, Any]:
        """Expose runtime settings to nodes for every entrypoint."""
        return {
            "max_mentions": int(self.config.get("max_mentions_per_report", 500)),
            "crisis_spike_threshold": float(self.config.get("crisis_spike_threshold", 0.75)),
            "date_range_days": int(self.config.get("date_range_days", 7)),
            "response_budget": int(self.config.get("llm_max_tokens", 2000)),
            "max_input_length": int(self.config.get("max_input_length", 200)),
        }

    def get_output(self, state: dict[str, Any]) -> dict[str, Any]:
        """Keep API metadata intact and expose the report to Marketplace."""
        envelope = cast(dict[str, Any], super().get_output(state))
        context = state.get("input_context")
        is_marketplace = isinstance(context, dict) and "conversation_history" in context
        if not is_marketplace:
            return envelope

        if state.get("input_error_message"):
            lines = [
                "The brand sentiment report could not be produced.",
                "",
                f"Reason: {state['input_error_message']}",
            ]
            guidance = state.get("input_error_guidance")
            if isinstance(guidance, list) and guidance:
                lines.extend(["", "How to continue:"])
                lines.extend(f"- {item}" for item in guidance)
            envelope["output"] = "\n".join(lines)
            return envelope

        response = self._parse_response(envelope.get("output"))
        if response is not None:
            answer = str(response.get("answer", "")).strip()
            if answer:
                envelope["output"] = answer
        return envelope

    @staticmethod
    def _parse_response(output: Any) -> dict[str, Any] | None:
        if isinstance(output, dict):
            return output
        if not isinstance(output, str):
            return None
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def register_nodes(self) -> None:
        super().register_nodes()

        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = MainNode()
        self._nodes["post_process"] = PostProcessNode(
            llm_timeout_s=float(self.config.get("timeout_s", 30.0)),
            llm_max_retry=int(self.config.get("max_retry", 3)),
        )
