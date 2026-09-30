# Template Design Specification

## Position in AgentCore Architecture

- **Agent Class**: BrandSentimentReportAgent
- **L1 Base**: AgentBaseGraph
- **Category**: Cat 1
- **Generation Mode**: llm
- **Three-Layer Separation**:
  - State: flat TypedDict fields containing only msgpack-safe data.
  - Node: FunctionNode subclasses override execute and optional extra security hooks.
  - Graph: register_nodes composes the fixed framework pipeline.

## Architecture Overview

| Node | Responsibility | Input State | Output State | Inherits |
|------|----------------|-------------|--------------|----------|
| initialize | Framework invocation metadata | framework-managed | framework-managed | InitializeNode |
| pre_process | Sanitize the request; extract brand, window, and competitors | user_input or raw_query | brand_query, date_range_days, competitor_brands | FunctionNode |
| main | Ingest mentions, score sentiment, run both non-suppressible gates | brand and window | mentions, sentiment_summary, crisis fields, competitor_scores | FunctionNode |
| post_process | Write the report or safely degrade, then enforce the output gate | scored mentions and metrics | answer, citations, confidence, validation_passed, formatted_output | FunctionNode |
| finalize | Framework response metadata | framework-managed | framework-managed | FinalizeNode |

### Data Flow

    START → initialize → pre_process → main → post_process → finalize → END
                                      ↘ retry → pre_process

Graph registers ingestion and request-scoped synthesis separately:

    self._nodes["main"] = MainNode()
    self._nodes["post_process"] = PostProcessNode(
        llm_timeout_s=config["timeout_s"],
        llm_max_retry=config["max_retry"],
    )

Production creates the Azure client from invocation-scoped secrets inside PostProcessNode.
Constructor injection is a test seam only; clients are never written to State or checkpoint surfaces.

### Request Parsing

The brand is taken from a quoted string when present, otherwise from the phrase following
"report for" / "sentiment on". A request with no identifiable brand returns actionable guidance
rather than reporting on an empty query.

The reporting window is parsed from phrases such as "over the last 7 days" or "past 2 weeks" and
clamped to one year. An unbounded window would issue an ingestion request that silently returns
nothing, which reads to the user as "no mentions" rather than "window too large".

Competitors are taken from a "compared against X and Y" clause and split on commas, slashes, and
conjunctions.

## State Definition

| Field group | Types | Purpose |
|-------------|-------|---------|
| raw_query, normalized_query, query_id | str | Safe request processing and correlation |
| brand_query, date_range_days, competitor_brands | str, int, str | Ingestion parameters; competitors JSON-serialized |
| mentions, mention_count, redaction_count | str, int, int | Redacted mention corpus and gate evidence |
| ingestion_source | str | Which feed produced the corpus, or `no_results` |
| sentiment_summary | str | JSON-serialized aggregate metrics |
| crisis_spike_score, crisis_detected | float, bool | Crisis gate output |
| competitor_scores | str | JSON-serialized per-competitor counts |
| answer, citations, confidence | str | Domain response |
| validation_passed, validation_findings | bool, str | Output gate verdict and evidence |
| synthesis_mode | str | azure_openai or evidence_only |
| formatted_output | str | Canonical JSON response exposed by AgentBaseGraph |
| error_message, recoverable_error_code | str or None | Short-circuit and degradation control |
| input_error_message, input_error_guidance | str or None, list[str] | User-facing input guidance |

**State Constraints:**

- Flat TypedDict only; collections contain primitive or JSON-safe values.
- No API keys, bearer tokens, credentials, feeds, LLM clients, or Pydantic objects.
- Mention text written to State is always post-redaction. State is checkpointable, so an
  un-redacted handle in State would be personal data at rest.
- InvocationContext is reconstructed with InvocationContext.from_state(state) inside a node if needed.

## Framework Utilization

- [x] InvocationContext and caller trust at the graph boundary.
- [x] Framework S-1 required_trust_level on every domain node.
- [x] Framework final S-2/S-3 gates remain non-overridden.
- [x] PreProcessNode extends S-2 through _extra_security_gate_input to block prompt injection.
- [x] MainNode extends S-2 through _extra_security_gate_input for brand-query PII masking.
- [x] PostProcessNode extends S-3 through _extra_security_gate_output for the personal-data check.
- [x] Every execute method emits a domain trace event on success, failure, or short circuit.
- [x] AgentRegistry and standalone server both load config/config.yaml separately from agent.yaml.
- [x] Marketplace and standalone entrypoints bind secrets per invocation and degrade safely when Azure is unavailable.

## Security and Runtime Decisions

- Minimum caller trust: INTERNAL. This agent reads a brand's mention corpus and produces a
  management-facing document; it is not a public-facing capability.
- External bearer credentials map only to VERIFIED_EXTERNAL, so an external caller cannot reach
  these nodes at all. A separately configured STG runner token is required for INTERNAL.
- Bearers are compared in constant time and invalid configured credentials return HTTP 401.
- memory_enabled and hitl.enabled are false, so no checkpointer is created.
- A missing mention feed returns zero mentions; no mention counts or scores are invented.

### Non-suppressible gate 1 — personal-data redaction

Handles, detected personal data, and author fields are stripped in MainNode, before the mentions
are written to State or reach the report prompt. PostProcessNode then re-checks the generated
report and **withholds it entirely** if any personal data is present, rather than masking and
releasing. Mention text is third-party content written by members of the public who did not
consent to appear in a corporate report; the safe outcome of a leak is no report.

Competitor scoring is computed against the pre-redaction text and returns **counts only**. The
shared PII detector treats a brand name as a person name, so scoring against redacted text would
report zero mentions for every competitor. No mention text from that path enters the result.

### Non-suppressible gate 2 — crisis-spike detection

The spike score multiplies normalised volume by the negative ratio, so a spike requires both.
The score is computed in code, not asked of the model, so a crisis signal survives a provider
outage and cannot be talked down by injected text. Sentiment counts are likewise deterministic,
which is why a report can cite figures that are reproducible.

Both gates are enforced at compile time: a config key claiming to suppress either raises
SecurityViolationError rather than being silently ignored.

### Degradation

Azure client creation is lazy, so missing provider configuration does not break startup. At
invocation time, provider failure emits progress and returns `synthesis_mode=evidence_only` with
the measured figures, the crisis flag, and the competitor table — everything a reader needs to
act, without a generated narrative.

## EU AI Act Art.13 Design-Time Evidence

The proposal declares this brand-reporting capability outside Annex III scope.
The README documents intended purpose and limitations, and response metadata exposes
mention count, crisis status, synthesis mode, and confidence for user transparency.

## Import Isolation Confirmation

- [x] Source imports framework and shared layers, never the Level 0 platform SDK.
- [x] Agent-local imports use the src package path.

## Design Decision Record

| Decision | Options | Chosen | Rationale |
|----------|---------|--------|-----------|
| L1 base | AgentBaseGraph / AutonomousBaseGraph | AgentBaseGraph | Fixed ingest, score, report stages need no autonomous loop |
| Composition | Standalone / subgraph / remote | Standalone | One independently deployable technical capability |
| Caller trust | VERIFIED_EXTERNAL / INTERNAL | INTERNAL | Reads a brand's mention corpus and produces a management document |
| HITL | Enabled / disabled | Disabled | The report is advisory and authorizes no action |
| Memory | Enabled / disabled | Disabled | Each report request is self-contained |
| Sentiment scoring | LLM-scored / deterministic lexicon | Deterministic | Cited figures must be reproducible and survive provider outage |
| Crisis signal | LLM-judged / computed | Computed | Must survive provider outage and resist injected text |
| Personal data in output | Mask and release / withhold report | Withhold | Mention authors never consented to appear in a corporate report |
| Competitor scoring input | Redacted text / pre-redaction text, counts only | Pre-redaction, counts only | The PII detector masks brand names, which would zero every competitor |
| Gate suppression keys | Ignore / reject at compile | Reject | A deployment must not believe it disabled a mandatory gate |
