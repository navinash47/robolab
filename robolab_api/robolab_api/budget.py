"""Budget helpers — Phase 0 reads BUDGET_USD_CAP from the environment."""

from __future__ import annotations

import os


def get_budget() -> dict[str, float]:
    """Return budget cap, month spend, and remaining USD.

    Phase 0: month spend is always 0 (no cost ledger yet).
    """
    cap = float(os.environ.get("BUDGET_USD_CAP", "100"))
    month_spend = 0.0
    return {
        "budget_usd_cap": cap,
        "month_spend_usd": month_spend,
        "remaining_usd": cap - month_spend,
    }
