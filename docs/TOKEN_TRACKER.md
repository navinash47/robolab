# Cursor / agent token cost tooling (ops)

Local references (cloned outside the RoboLab git tree):

| Repo | Path | What we took |
|------|------|----------------|
| [headroomlabs-ai/headroom](https://github.com/headroomlabs-ai/headroom) | `~/Projects/headroom` (+ Cursor MCP `user-headroom`) | Compress / retrieve / stats; optional proxy `:8787` |
| [inboxpraveen/Minimize-Cursor-Cost](https://github.com/inboxpraveen/Minimize-Cursor-Cost) | `~/Projects/token-tools/Minimize-Cursor-Cost` | Lean agent rules → `.cursor/rules/token-efficiency.mdc` |
| [Bob8259/Cursor-Token-Saver-and-Customizer](https://github.com/Bob8259/Cursor-Token-Saver-and-Customizer) | `~/Projects/token-tools/Cursor-Token-Saver-and-Customizer` | **Archived/closed** — UI monitor pattern only; do **not** wire Base URL proxy into Cursor |

## RoboLab surfaces

- **Page:** `GET /tokens` (nav **Tokens** next to Spend)
- **JSON:** `GET /api/tokens`
- **Log event:** `POST /api/tokens`
- **Headroom snapshot:** `POST /api/tokens/headroom`
- **CLI:** `python scripts/log_token_usage.py --model … --input N --output M`

Rates (estimate only): `TOKEN_USD_PER_1M_INPUT` (default 3) · `TOKEN_USD_PER_1M_OUTPUT` (default 15).

This ledger is **not** Cursor billing truth — use Cursor Usage for plan quota; use `/tokens` for session discipline alongside `/spend` (GPU).
