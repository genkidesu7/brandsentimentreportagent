# Test Specification

## Test Strategy

- Coverage target: 80%
- Test types: unit, framework compliance, proof-of-boundary, integration, and local STG invocation.
- Unit tests may call execute directly for isolated business logic.
- Security-boundary tests call node(state) through FunctionNode.__call__.
- No live Azure OpenAI or mention feed is required: the provider is exercised through a recording
  double, and ingestion through an injected stub feed.

## Framework Compliance Tests

| TC-ID | Test | Expected Result |
|-------|------|-----------------|
| TC-01 | State contract is a flat TypedDict | No Pydantic, dataclass, secret, or runtime client fields |
| TC-02 | Invalid or unsafe input is rejected or sanitized | Safe failure, markup stripped, or blocked injection attempt |
| TC-03 | No credential in State | Credential scan reports zero violations |
| TC-04 | InvocationContext construction inside nodes | Only from_state is allowed; authenticated server adapter is exempt |
| TC-05 | No duplicate lifecycle trace events | execute methods do not emit node_start, node_complete, or node_error |
| TC-06 | FunctionNode input security gate is final | Override attempt raises TypeError |
| TC-07 | FunctionNode output security gate is final | Override attempt raises TypeError |
| TC-08 | required_trust_level is enforced through node(state) | Denial happens before execute; every domain node requires INTERNAL |
| TC-09 | PreProcessNode and MainNode extra S-2 hooks | Prompt injection blocked; PII in the brand query masked before ingestion |
| TC-10 | PostProcessNode extra S-3 hook | Report containing personal data is withheld, not released |
| TC-11 | Domain tracing | Each execute path emits a domain event |

## Proof-of-Boundary Tests

| PB-ID | Boundary | Expected Result |
|-------|----------|-----------------|
| PB-1 | BaseNode to trace emitter | Domain and lifecycle trace evidence exists |
| PB-2 | State serialization | Only msgpack-safe annotated fields |
| PB-3 | Template to external service | Injected mention feed and LLM doubles exercise the boundary |
| PB-4 | Import isolation | No Level 0 imports |
| PB-5 | Checkpoint safety, conditional | Auto-waived — checkpointing disabled |
| PB-6 | Invoke order | S-1 → node_start → S-2 → execute → S-3 → node_complete |
| PB-7 | HITL interrupt propagation, conditional | Auto-waived — non-HITL |

PB-1 through PB-4 and PB-6 are mandatory. PB-5 applies only when memory or HITL enables
checkpointing and the framework provides both ingress hooks. PB-7 applies only when
config/config.yaml sets hitl.enabled to true.

## Business Logic Tests

| TC-ID | Test | Input | Expected Result |
|-------|------|-------|-----------------|
| BL-01 | Request parsing | quoted and unquoted brand, window phrases, competitor clause | Correct brand, window, and competitor list |
| BL-02 | Window clamping | "over the last 99 months" | Clamped to 365 days rather than issuing an unbounded request |
| BL-03 | Input validation | markup-only, over-length, too-short, and brandless requests | Actionable guidance or safe rejection |
| BL-04 | Prompt injection | instruction-override and system-prompt probes | SecurityViolationError before execute |
| BL-05 | Ingestion | injected feed, missing feed, and feed raising | Mentions returned, no evidence, or recoverable outage without backend detail |
| BL-06 | Mention cap | feed returning more than max_mentions | Corpus truncated to the cap |
| BL-07 | Sentiment scoring | positive, negative, and neutral mentions | Correct counts and negative ratio |
| BL-08 | Crisis gate — negativity alone | few strongly negative mentions | No crisis: low volume is churn |
| BL-09 | Crisis gate — volume alone | many positive mentions | No crisis: volume alone is a launch |
| BL-10 | Crisis gate — true spike | high volume of negative mentions | Crisis detected, score above threshold |
| BL-11 | Redaction gate | mention containing a handle and an author | Handle and author redacted before reaching State |
| BL-12 | Competitor scoring | brand-shaped competitor name | Counts still match — regression for the PII-masked brand name |
| BL-13 | Competitor output safety | mention containing a handle | Competitor scores carry counts only, no mention text |
| BL-14 | Report generation | metrics plus injected LLM | Report, deduplicated citations, confidence, canonical output |
| BL-15 | Provider failure | LLM raises | Evidence-only report carrying the measured figures |
| BL-16 | Crisis visibility under degradation | crisis flagged and LLM raises | Crisis still prominent in the fallback report |
| BL-17 | Output gate | report reintroducing a handle or personal data | Report withheld, findings recorded, formatted_output replaced |
| BL-18 | Confidence tiers | mention counts | High, medium, or low by sample size |
| BL-19 | Gate suppression | config claiming to disable either gate | SecurityViolationError at compile |
| BL-20 | Azure wiring | Azure secrets present | Request-scoped client built from ctx.secrets, no process-global client |
| BL-21 | Missing secrets | no Azure secrets bound | Terminal evidence-only report rather than a failed invocation |
| BL-22 | Bearer mapping | external, runner, wrong, and absent bearer | VERIFIED_EXTERNAL, INTERNAL, HTTP 401, and ANONYMOUS respectively |
| BL-23 | Marketplace shaping | same state with and without conversation_history | Marketplace receives the report text, the API keeps canonical JSON |

## Test Execution Summary

- Execution date: 2026-09-30
- Framework: agenticstar-agentcore 1.0.3 wheel (not a CI import shim)
- Full suite: 86 passed, 0 failed, 3 skipped
- Local gates: forbidden strings, manifest schema, trust level, audit trace, dependency pinning,
  stub tests, credential scan, OSS license, and Cat consistency all PASS
- Lint and types: ruff format, ruff check, and mypy strict on src/ all clean

The three expected skips are PB-5 (checkpointing disabled) and both PB-7 cases (non-HITL).

One defect was found and fixed during this cycle, covered by regression tests BL-12 and BL-13:
competitor scoring ran against redacted mention text, and because the shared PII detector masks
a brand name as a person name, every competitor scored zero mentions.
