# mdrouter Roadmap

This roadmap focuses on lowering token cost, improving routing quality, and enabling tool-augmented responses safely.

## Product direction (validated)

The following ideas are validated and should remain core to mdrouter:

- Centralize all LLM providers behind one consistent API layer.
- Keep OpenAI/Ollama compatibility so clients can run anywhere with minimal changes.
- Maintain semantic cache as a first-class cost and latency feature.
- Introduce `mdrouter/auto` smart routing to combine local and cloud models.
- Add shared memory (project + chat summaries) for local and team cloud workflows.
- Add tool support for models/providers that do not support tools natively.
- Add usage learner pipeline to export supervised fine-tuning datasets from usage data.
- Add budget controls and policy enforcement.
- Add local and cloud dashboards, including team capabilities.

## Phase 0: Platform Baseline and Positioning

- Harden provider centralization (one control plane for model aliases, routing policy, and credentials).
- Preserve OpenAI/Ollama compatibility for all main endpoints.
- Stabilize semantic cache defaults and observability.
- Publish migration and compatibility guarantees.

Success criteria:
- Existing clients can switch to mdrouter without code changes beyond endpoint/model alias.
- Cache hit behavior and request logs are auditable in production.

## Phase 1: Cost Visibility and Baseline Control

- Add per-model cost analytics in mdrouterctl (done for token-based estimates).
- Add daily rolling report command and CSV export.
- Track cache savings metrics explicitly:
  - `estimated_uncached_tokens`
  - `estimated_saved_tokens`
  - `estimated_saved_cost`

Success criteria:
- Operator can answer "which model is costing the most" in under 30 seconds.
- Operator can answer "how much did cache save this week" in under 30 seconds.

## Phase 2: Policy Engine and Smart Routing

- Add policy engine with routing modes:
  - lowest_cost
  - fastest
  - balanced
  - quality_first
- Add `mdrouter/auto` virtual model:
  - classify request complexity
  - pick local model for low-risk/low-complexity prompts
  - escalate to cloud model when quality threshold or tool requirements demand it
- Add per-route and per-tenant policy selection via header.
- Add health and circuit breaker scoring per provider.
- Add automatic fallback on timeout and 5xx with configurable retry budget.

Success criteria:
- Fallback success rate > 99% for transient upstream failures.
- Median latency reduced by at least 20% in balanced mode.
- At least 30% of eligible traffic served by local models without quality regression.

## Phase 3: Better Token Efficiency

- Add request compaction for conversation history:
  - sliding window by token budget
  - semantic summarization of old turns
  - tool output truncation rules
- Add cacheability classifier to decide when semantic cache is likely safe.
- Add prefix partitioning strategy for long prompts (static prefix vs dynamic suffix).

Success criteria:
- Input token usage reduced by at least 25% on long sessions.

## Phase 4: Tool Gateway and Tool Augmentation

- Add first-party tool gateway with allowlist and timeouts:
  - web_search tool
  - web_scrape tool (robots-respecting)
  - fetch_url tool
- Add tool augmentation mode for providers/models with no native tool support:
  - model emits structured intent
  - mdrouter executes tool calls and injects tool results back into context
  - enforce deterministic loop/timeout/token guards
- Add tool execution telemetry and token-to-tool ROI metrics.
- Add policy controls per model/provider for tool access.

Success criteria:
- Tool calls remain auditable and bounded by strict execution limits.
- Tool-augmented responses are deterministic and policy-compliant across supported models.

## Phase 5: Shared Memory and Knowledge Store

- Add shared memory service (local-first, cloud-sync optional):
  - project summaries
  - chat summaries
  - task context snapshots
- Add query API for memory retrieval by scope:
  - local user scope
  - project scope
  - team scope (cloud)
- Add privacy controls, retention policies, and memory provenance metadata.

Success criteria:
- Agents can retrieve concise project context in < 300ms from local memory store.
- Team memory sync supports conflict-safe updates and access control.

## Phase 6: Advanced Cache Stack

- Add embedding-based semantic cache index with pluggable vector backend.
- Add cache invalidation policies by model/provider/tool-set signature.
- Add cache warmup jobs for repeated enterprise prompts.

Success criteria:
- Semantic cache precision >= 95% on curated regression suite.

## Phase 7: Budget Controls and Usage Learner

- Add budget controls:
  - per-user, per-team, per-model monthly caps
  - soft limit alerts and hard-stop enforcement
  - emergency degrade policy (force lowest_cost/local-only)
- Add usage learner export pipeline:
  - redact and normalize request/response traces
  - export SFT and preference dataset formats
  - quality filters for bad/incomplete samples

Success criteria:
- Budget overruns are prevented by policy, not manual intervention.
- Fine-tuning export job produces reproducible datasets with audit metadata.

## Phase 8: Local and Cloud Management Dashboards

- Add local dashboard:
  - route health
  - cache hit rates
  - cost and budget status
  - tool activity
- Add cloud dashboard:
  - team management
  - API keys and policy management
  - usage analytics and billing views
  - shared memory administration

Success criteria:
- Operators can manage routing, budgets, and policies without editing files manually.
- Team admin flows are RBAC-controlled and fully auditable.

## Phase 9: Production Operations

- Add native systemd install command.
- Add Prometheus metrics endpoint.
- Add SLO dashboards (latency, error, cache hit, cost/hour).
- Add configurable retention and redaction policies for logs.

Success criteria:
- Single-command deployment and observable runtime state.

## Internet research highlights informing this roadmap

- OpenAI Prompt Caching docs: static prefix ordering and consistent `prompt_cache_key` improve cache hit rate and cost.
- Anthropic Prompt Caching docs: explicit/automatic breakpoints, cache invalidation hierarchy (tools -> system -> messages), and usage fields to measure cache reads/writes.
- Redis semantic/embedding cache guidance: TTL-controlled embedding caches and batch operations for lower latency and compute overhead.
