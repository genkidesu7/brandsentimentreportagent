"""Flat, msgpack-safe state for CMN-C1-091."""

# ADR-005: State must be a flat TypedDict — never Pydantic BaseModel.
# LangGraph checkpoints use msgpack serialization; Pydantic objects
# cause silent corruption.  Extend AgentState with agent-specific
# fields only.  Do NOT add credentials, secrets, or Pydantic models.

from framework.schemas.agent_state import AgentState


class State(AgentState):
    """Agent state for BrandSentimentReportAgent.

    Add only fields specific to this agent below.
    All shared fields (user_input, status, session_id, node_history,
    error_log, hitl_*, etc.) are inherited from AgentState.
    """

    # --- Input ---
    raw_query: str
    brand_query: str
    date_range_days: int
    competitor_brands: str  # JSON string: list of competitor brand names
    max_mentions: int
    crisis_spike_threshold: float
    response_budget: int
    max_input_length: int

    # --- Pre-process outputs ---
    query_id: str
    normalized_query: str

    # --- Main (ingestion + scoring) outputs ---
    mentions: str  # JSON string: list of redacted mention dicts
    mention_count: int
    redaction_count: int
    ingestion_source: str
    sentiment_summary: str  # JSON string: aggregate sentiment metrics
    crisis_spike_score: float
    crisis_detected: bool
    competitor_scores: str  # JSON string: {brand: {avg_score}}

    # --- Post-process outputs ---
    answer: str
    citations: str  # JSON string: list of citation dicts
    confidence: str  # "high" | "medium" | "low"
    synthesis_mode: str  # "azure_openai" | "evidence_only"
    validation_passed: bool
    validation_findings: str  # JSON string: list of finding descriptions
    formatted_output: str

    # --- Error ---
    error_message: str | None
    recoverable_error_code: str | None
    input_error_message: str | None
    input_error_guidance: list[str]
