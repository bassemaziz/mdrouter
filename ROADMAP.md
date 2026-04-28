# mdrouter Roadmap

This roadmap focuses on reliability, cost efficiency, and compatibility for production routing workloads.

## Principles

- Preserve OpenAI/Ollama compatibility.
- Keep behavior auditable via structured logs.
- Optimize for predictable latency and cost, not experimental features.
- Prefer provider-agnostic controls and alias-based routing.

## Current Focus (Q2 2026)

### 1. Cache Safety and Rollout Discipline

- Keep router-side response cache disabled by default while semantic reuse is re-validated on live traffic.
- Maintain regression coverage for stale-response scenarios.
- Re-enable cache behind explicit operator opt-in.

Success criteria:
- Zero stale semantic replay incidents in validation suite.
- Documented enablement checklist for production rollout.

### 2. Smart Routing Hardening

- Stabilize `mdrouter/auto` class-based routing (`default_coding`, `heavy_refactor`, `long_context`, `tool_heavy`).
- Improve fallback behavior under provider errors/timeouts.
- Keep vision/tool constraints strict and explicit.

Success criteria:
- Deterministic route selection for equivalent requests.
- Graceful fallback coverage for transient upstream failures.

### 3. Operational Clarity

- Keep JSONL lifecycle events complete and parse-friendly.
- Maintain `mdrouterctl` operator commands:
  - `status`
  - `stats`
  - `cachestatus`
- Ensure cache and token/cost diagnostics remain actionable.

Success criteria:
- Operators can identify top models, request classes, and cache behavior in under 30 seconds.

## Next Phase

### Policy and Budget Controls

- Add configurable routing policy tiers (cost-first, balanced, quality-first).
- Introduce enforceable budget guards and degrade modes.
- Expand per-tenant/per-route policy overlays.

### Token Efficiency

- Add safe context compaction strategies for long sessions.
- Add stricter controls for tool-output verbosity.

### Production Tooling

- Expand deployment and observability guidance.
- Add SLO-oriented runtime metrics and dashboards.

## Out of Scope (for now)

- Experimental memory systems not tied to measurable routing outcomes.
- Feature expansion that weakens compatibility guarantees.

