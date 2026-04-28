from __future__ import annotations

import argparse
import json
from pathlib import Path

from mdrouter.ops import _cachestatus
from mdrouter.ops import _status


def _write_log(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def test_status_reads_model_alias_fields(tmp_path, capsys):
    log_file = tmp_path / "router_requests.jsonl"
    _write_log(
        log_file,
        [
            {
                "ts": "2026-04-28T10:00:00+00:00",
                "event": "request_start",
                "path": "/v1/chat/completions",
                "method": "POST",
                "model_alias": "go/glm-5",
                "model": "go/glm-5",
            },
            {
                "ts": "2026-04-28T10:00:01+00:00",
                "path": "/v1/chat/completions",
                "method": "POST",
                "model_alias": "go/glm-5",
                "model": "go/glm-5",
                "cache_hit_type": "exact",
                "prompt_tokens": 100,
                "completion_tokens": 30,
                "cached_tokens": 20,
            },
        ],
    )

    exit_code = _status(
        argparse.Namespace(log_file=str(log_file), hours=100000, pricing="")
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "go/glm-5" in output
    assert "Cache exact hits" in output


def test_cachestatus_reports_effective_hit_rate(tmp_path, capsys):
    log_file = tmp_path / "router_requests.jsonl"
    _write_log(
        log_file,
        [
            {
                "ts": "2026-04-28T10:00:00+00:00",
                "event": "request_start",
                "path": "/api/chat",
                "method": "POST",
                "model_alias": "novita/demo-model",
            },
            {
                "ts": "2026-04-28T10:00:01+00:00",
                "path": "/api/chat",
                "method": "POST",
                "model_alias": "novita/demo-model",
                "cache_hit_type": "semantic",
            },
            {
                "ts": "2026-04-28T10:00:02+00:00",
                "path": "/api/chat",
                "method": "POST",
                "model_alias": "novita/demo-model",
                "cache_hit_type": "miss",
            },
        ],
    )

    exit_code = _cachestatus(argparse.Namespace(log_file=str(log_file), hours=100000))

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "mdrouter cachestatus" in output
    assert "Effective hit rate" in output
    assert "50.00%" in output
