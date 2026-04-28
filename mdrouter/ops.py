from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass
class Price:
    prompt_per_mtok: float
    completion_per_mtok: float


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _load_pricing(path: Path | None) -> tuple[Price | None, dict[str, Price]]:
    if path is None:
        return None, {}
    if not path.exists():
        raise FileNotFoundError(f"Pricing file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        blob = json.load(f)

    default_price = None
    if isinstance(blob.get("default"), dict):
        default_price = Price(
            prompt_per_mtok=float(blob["default"].get("prompt_per_mtok", 0.0)),
            completion_per_mtok=float(blob["default"].get("completion_per_mtok", 0.0)),
        )

    per_model: dict[str, Price] = {}
    models = blob.get("models", {})
    if isinstance(models, dict):
        for model, row in models.items():
            if not isinstance(row, dict):
                continue
            per_model[str(model)] = Price(
                prompt_per_mtok=float(row.get("prompt_per_mtok", 0.0)),
                completion_per_mtok=float(row.get("completion_per_mtok", 0.0)),
            )

    return default_price, per_model


def _price_for_model(
    model: str, default_price: Price | None, per_model: dict[str, Price]
) -> Price | None:
    if model in per_model:
        return per_model[model]
    return default_price


def _fmt_money(value: float) -> str:
    return f"${value:,.6f}"


def _fmt_num(value: int) -> str:
    return f"{value:,}"


def _status(args: argparse.Namespace) -> int:
    log_file = Path(args.log_file)
    records = _read_jsonl(log_file)
    cutoff = datetime.now(UTC) - timedelta(hours=args.hours)

    default_price, model_prices = _load_pricing(
        Path(args.pricing) if args.pricing else None
    )

    model_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "estimated_cost": 0.0,
            "cache_exact": 0,
            "cache_semantic": 0,
            "cache_miss": 0,
            "upstream": 0,
        }
    )

    totals = {
        "requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "estimated_cost": 0.0,
        "cache_exact": 0,
        "cache_semantic": 0,
        "cache_miss": 0,
        "upstream": 0,
    }

    for rec in records:
        ts = _parse_ts(rec.get("ts"))
        if ts is not None and ts < cutoff:
            continue

        if rec.get("method") != "POST":
            continue

        model = str(rec.get("model_alias") or rec.get("model") or "unknown")
        event = str(rec.get("event") or "")
        path = str(rec.get("path") or "")

        is_chat_request = path in {"/api/chat", "/api/generate", "/v1/chat/completions"}
        if not is_chat_request:
            continue

        # Count requests from start events only to avoid duplicate counting
        if event == "request_start":
            model_stats[model]["requests"] += 1
            totals["requests"] += 1

        cache_hit = str(rec.get("cache_hit_type") or rec.get("cache_hit") or "")
        if cache_hit == "exact":
            model_stats[model]["cache_exact"] += 1
            totals["cache_exact"] += 1
        elif cache_hit == "semantic":
            model_stats[model]["cache_semantic"] += 1
            totals["cache_semantic"] += 1
        elif cache_hit == "miss":
            model_stats[model]["cache_miss"] += 1
            totals["cache_miss"] += 1
        elif cache_hit == "upstream":
            model_stats[model]["upstream"] += 1
            totals["upstream"] += 1

        prompt_tokens = int(rec.get("prompt_tokens") or 0)
        completion_tokens = int(rec.get("completion_tokens") or 0)
        cached_tokens = int(rec.get("cached_tokens") or 0)

        if prompt_tokens or completion_tokens or cached_tokens:
            model_stats[model]["prompt_tokens"] += prompt_tokens
            model_stats[model]["completion_tokens"] += completion_tokens
            model_stats[model]["cached_tokens"] += cached_tokens
            totals["prompt_tokens"] += prompt_tokens
            totals["completion_tokens"] += completion_tokens
            totals["cached_tokens"] += cached_tokens

            price = _price_for_model(model, default_price, model_prices)
            if price is not None:
                cost = (prompt_tokens / 1_000_000.0) * price.prompt_per_mtok + (
                    completion_tokens / 1_000_000.0
                ) * price.completion_per_mtok
                model_stats[model]["estimated_cost"] += cost
                totals["estimated_cost"] += cost

    print(f"mdrouter status (last {args.hours}h)")
    print(f"log file: {log_file}")
    print()
    print("Totals")
    print(f"  Requests:           {_fmt_num(totals['requests'])}")
    print(f"  Prompt tokens:      {_fmt_num(totals['prompt_tokens'])}")
    print(f"  Completion tokens:  {_fmt_num(totals['completion_tokens'])}")
    print(f"  Cached tokens:      {_fmt_num(totals['cached_tokens'])}")
    print(f"  Cache exact hits:   {_fmt_num(totals['cache_exact'])}")
    print(f"  Cache semantic:     {_fmt_num(totals['cache_semantic'])}")
    print(f"  Cache misses:       {_fmt_num(totals['cache_miss'])}")
    print(f"  Upstream events:    {_fmt_num(totals['upstream'])}")
    if args.pricing:
        print(f"  Estimated cost:     {_fmt_money(totals['estimated_cost'])}")
    print()

    if not model_stats:
        print("No matching events found for selected time window.")
        return 0

    print("Per model")
    for model, stat in sorted(model_stats.items(), key=lambda kv: kv[0]):
        print(f"- {model}")
        print(f"    requests:         {_fmt_num(stat['requests'])}")
        print(f"    prompt_tokens:    {_fmt_num(stat['prompt_tokens'])}")
        print(f"    completion_tokens:{_fmt_num(stat['completion_tokens'])}")
        print(f"    cached_tokens:    {_fmt_num(stat['cached_tokens'])}")
        print(f"    cache_exact:      {_fmt_num(stat['cache_exact'])}")
        print(f"    cache_semantic:   {_fmt_num(stat['cache_semantic'])}")
        print(f"    cache_miss:       {_fmt_num(stat['cache_miss'])}")
        if args.pricing:
            print(f"    est_cost:         {_fmt_money(stat['estimated_cost'])}")

    return 0


def _cachestatus(args: argparse.Namespace) -> int:
    log_file = Path(args.log_file)
    records = _read_jsonl(log_file)
    cutoff = datetime.now(UTC) - timedelta(hours=args.hours)

    totals = {
        "requests": 0,
        "cache_exact": 0,
        "cache_semantic": 0,
        "cache_miss": 0,
        "cache_upstream": 0,
    }
    per_model: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "requests": 0,
            "cache_exact": 0,
            "cache_semantic": 0,
            "cache_miss": 0,
            "cache_upstream": 0,
        }
    )

    for rec in records:
        ts = _parse_ts(rec.get("ts"))
        if ts is not None and ts < cutoff:
            continue
        if rec.get("method") != "POST":
            continue

        path = str(rec.get("path") or "")
        if path not in {"/api/chat", "/api/generate", "/v1/chat/completions"}:
            continue

        model = str(rec.get("model_alias") or rec.get("model") or "unknown")
        event = str(rec.get("event") or "")
        if event == "request_start":
            totals["requests"] += 1
            per_model[model]["requests"] += 1

        cache_hit = str(rec.get("cache_hit_type") or rec.get("cache_hit") or "")
        if cache_hit == "exact":
            totals["cache_exact"] += 1
            per_model[model]["cache_exact"] += 1
        elif cache_hit == "semantic":
            totals["cache_semantic"] += 1
            per_model[model]["cache_semantic"] += 1
        elif cache_hit == "miss":
            totals["cache_miss"] += 1
            per_model[model]["cache_miss"] += 1
        elif cache_hit == "upstream":
            totals["cache_upstream"] += 1
            per_model[model]["cache_upstream"] += 1

    cache_events = (
        totals["cache_exact"]
        + totals["cache_semantic"]
        + totals["cache_miss"]
        + totals["cache_upstream"]
    )
    hit_events = totals["cache_exact"] + totals["cache_semantic"]
    hit_ratio = (hit_events / cache_events) if cache_events else 0.0

    print(f"mdrouter cachestatus (last {args.hours}h)")
    print(f"log file: {log_file}")
    print()
    print("Totals")
    print(f"  Requests:           {_fmt_num(totals['requests'])}")
    print(f"  Cache events:       {_fmt_num(cache_events)}")
    print(f"  Cache exact hits:   {_fmt_num(totals['cache_exact'])}")
    print(f"  Cache semantic:     {_fmt_num(totals['cache_semantic'])}")
    print(f"  Cache misses:       {_fmt_num(totals['cache_miss'])}")
    print(f"  Upstream events:    {_fmt_num(totals['cache_upstream'])}")
    print(f"  Effective hit rate: {hit_ratio:.2%}")
    print()

    if not per_model:
        print("No matching events found for selected time window.")
        return 0

    print("Per model")
    for model, stat in sorted(per_model.items(), key=lambda kv: kv[0]):
        model_events = (
            stat["cache_exact"]
            + stat["cache_semantic"]
            + stat["cache_miss"]
            + stat["cache_upstream"]
        )
        model_hits = stat["cache_exact"] + stat["cache_semantic"]
        model_hit_ratio = (model_hits / model_events) if model_events else 0.0
        print(f"- {model}")
        print(f"    requests:         {_fmt_num(stat['requests'])}")
        print(f"    cache_exact:      {_fmt_num(stat['cache_exact'])}")
        print(f"    cache_semantic:   {_fmt_num(stat['cache_semantic'])}")
        print(f"    cache_miss:       {_fmt_num(stat['cache_miss'])}")
        print(f"    upstream_events:  {_fmt_num(stat['cache_upstream'])}")
        print(f"    hit_rate:         {model_hit_ratio:.2%}")

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="mdrouter operations command")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser(
        "status", help="Show traffic, cache and cost summary from logs"
    )
    status.add_argument(
        "--log-file",
        default="logs/router_requests.jsonl",
        help="Path to JSONL request log file",
    )
    status.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Lookback window in hours",
    )
    status.add_argument(
        "--pricing",
        default="",
        help="Optional pricing JSON file for cost estimation",
    )

    stats = subparsers.add_parser("stats", help="Alias for status")
    stats.add_argument(
        "--log-file",
        default="logs/router_requests.jsonl",
        help="Path to JSONL request log file",
    )
    stats.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Lookback window in hours",
    )
    stats.add_argument(
        "--pricing",
        default="",
        help="Optional pricing JSON file for cost estimation",
    )

    cachestatus = subparsers.add_parser(
        "cachestatus", help="Show cache-hit effectiveness from logs"
    )
    cachestatus.add_argument(
        "--log-file",
        default="logs/router_requests.jsonl",
        help="Path to JSONL request log file",
    )
    cachestatus.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Lookback window in hours",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "status":
        raise SystemExit(_status(args))
    if args.command == "stats":
        raise SystemExit(_status(args))
    if args.command == "cachestatus":
        raise SystemExit(_cachestatus(args))
    parser.print_help()


if __name__ == "__main__":
    main()
