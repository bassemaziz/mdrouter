# mdrouter Copilot Instructions

## Project goals
- Keep mdrouter OpenAI/Ollama-compatible while improving cost efficiency and reliability.
- Prioritize minimal-latency, production-safe defaults over experimental behavior.
- Preserve backward compatibility unless a migration note is added in README.

## Coding conventions
- Python 3.11+ only.
- Keep dependencies minimal and justified in pull requests.
- Prefer explicit typing and small focused functions.
- Avoid hidden magic; surface behavior via env vars and logs.

## Router-specific rules
- Do not cache tool-call payloads as final assistant text.
- Any cache feature must emit enough log fields to audit hit quality and savings.
- Provider-specific logic should stay in adapters or dedicated helper functions.

## Logging and observability
- Keep logs JSONL and parse-friendly.
- Include event names for request lifecycle transitions.
- Avoid logging secrets (API keys, auth headers, raw credentials).

## Tests
- Add or update tests for changed behavior.
- For cache logic changes, include at least one positive and one negative scenario.

## Documentation requirements
- Update README when adding env vars, commands, or API behavior.
- Keep ROADMAP focused on concrete milestones and measurable outcomes.
