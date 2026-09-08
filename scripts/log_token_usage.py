#!/usr/bin/env python3
"""Log Cursor/agent token usage into RoboLab TokenLedger.

Examples:
  python scripts/log_token_usage.py --model composer --input 12000 --output 3000 --session kickstart
  python scripts/log_token_usage.py --headroom-json /tmp/headroom_stats.json
  echo '{"total_tokens_saved": 5000}' | python scripts/log_token_usage.py --headroom-stdin

Clones / references (local, not vendored into RoboLab):
  ~/Projects/headroom
  ~/Projects/token-tools/Minimize-Cursor-Cost
  ~/Projects/token-tools/Cursor-Token-Saver-and-Customizer  (archived; monitor pattern only)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any


def _api_base() -> str:
    return (os.environ.get("ROBOLAB_API") or "http://127.0.0.1:8000").rstrip("/")


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{_api_base()}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="cursor")
    p.add_argument("--source", default="manual")
    p.add_argument("--session", default="", dest="session_label")
    p.add_argument("--input", type=int, default=0, dest="input_tokens")
    p.add_argument("--output", type=int, default=0, dest="output_tokens")
    p.add_argument("--saved", type=int, default=0, dest="tokens_saved")
    p.add_argument("--usd", type=float, default=None, dest="estimated_usd")
    p.add_argument("--note", default="")
    p.add_argument("--headroom-json", default="", help="Path to headroom_stats JSON dump")
    p.add_argument(
        "--headroom-stdin",
        action="store_true",
        help="Read headroom_stats-like JSON from stdin",
    )
    args = p.parse_args()

    try:
        if args.headroom_json or args.headroom_stdin:
            raw = (
                sys.stdin.read()
                if args.headroom_stdin
                else open(args.headroom_json, encoding="utf-8").read()
            )
            snap = json.loads(raw)
            body = {
                "session_label": args.session_label or snap.get("session_label") or "headroom",
                "model": args.model or "headroom",
                "total_input_tokens": int(snap.get("total_input_tokens") or 0),
                "total_output_tokens": int(snap.get("total_output_tokens") or 0),
                "total_tokens_saved": int(snap.get("total_tokens_saved") or 0),
                "estimated_cost_saved_usd": float(snap.get("estimated_cost_saved_usd") or 0),
                "compressions": int(snap.get("compressions") or 0),
                "note": args.note,
                "meta": {k: snap[k] for k in snap if k not in {
                    "total_input_tokens",
                    "total_output_tokens",
                    "total_tokens_saved",
                    "estimated_cost_saved_usd",
                    "compressions",
                }},
            }
            out = _post("/api/tokens/headroom", body)
        else:
            body = {
                "source": args.source,
                "model": args.model,
                "session_label": args.session_label,
                "input_tokens": args.input_tokens,
                "output_tokens": args.output_tokens,
                "tokens_saved": args.tokens_saved,
                "note": args.note,
            }
            if args.estimated_usd is not None:
                body["estimated_usd"] = args.estimated_usd
            out = _post("/api/tokens", body)
    except urllib.error.URLError as e:
        print(f"API unreachable at {_api_base()}: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"failed: {e}", file=sys.stderr)
        return 1

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
