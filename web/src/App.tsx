import { useEffect, useState } from "react";

type Budget = {
  budget_usd_cap: number;
  month_spend_usd: number;
  remaining_usd: number;
};

type ExperimentsResponse = {
  experiments: { id: number; name: string }[];
  count: number;
};

function formatUsd(n: number): string {
  return n.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  });
}

export default function App() {
  const [budget, setBudget] = useState<Budget | null>(null);
  const [count, setCount] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const [budgetRes, experimentsRes] = await Promise.all([
          fetch("/api/budget"),
          fetch("/api/experiments"),
        ]);
        if (!budgetRes.ok || !experimentsRes.ok) {
          throw new Error(
            `API error: budget ${budgetRes.status}, experiments ${experimentsRes.status}`,
          );
        }
        const budgetJson: Budget = await budgetRes.json();
        const experimentsJson: ExperimentsResponse = await experimentsRes.json();
        if (!cancelled) {
          setBudget(budgetJson);
          setCount(experimentsJson.count);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setError(
            err instanceof Error
              ? err.message
              : "Failed to reach RoboLab API (is make dev running?)",
          );
        }
      }
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  const remainingLabel =
    budget === null
      ? "Budget: …"
      : `Budget: ${formatUsd(budget.remaining_usd)} remaining`;

  const experimentsLabel =
    count === null ? "…" : `${count} experiment${count === 1 ? "" : "s"}`;

  return (
    <div className="min-h-screen">
      <header className="flex items-center justify-between border-b border-[var(--border)] bg-[var(--surface)]/80 px-6 py-4 backdrop-blur">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-[var(--muted)]">
            Research orchestrator
          </p>
          <h1 className="text-2xl font-semibold tracking-tight text-[var(--text)]">
            RoboLab
          </h1>
        </div>
        <div className="text-right">
          <p className="text-sm font-medium text-[var(--ok)]" data-testid="budget">
            {remainingLabel}
          </p>
          {budget !== null && (
            <p className="text-xs text-[var(--muted)]">
              Cap {formatUsd(budget.budget_usd_cap)} · spent{" "}
              {formatUsd(budget.month_spend_usd)}
            </p>
          )}
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-10">
        {error && (
          <div
            className="mb-6 rounded border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200"
            role="alert"
          >
            {error}
          </div>
        )}

        <div className="mb-4 flex items-baseline justify-between gap-4">
          <h2 className="text-lg font-medium text-[var(--text)]">Experiments</h2>
          <p className="text-sm text-[var(--muted)]" data-testid="experiment-count">
            {experimentsLabel}
          </p>
        </div>

        <div className="overflow-hidden rounded-lg border border-[var(--border)] bg-[var(--surface)]">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-[var(--border)] bg-black/20 text-[var(--muted)]">
              <tr>
                <th className="px-4 py-3 font-medium">Name</th>
                <th className="px-4 py-3 font-medium">Sim</th>
                <th className="px-4 py-3 font-medium">Arch</th>
                <th className="px-4 py-3 font-medium">Status</th>
              </tr>
            </thead>
            <tbody>
              {count === 0 ? (
                <tr>
                  <td
                    colSpan={4}
                    className="px-4 py-16 text-center text-[var(--muted)]"
                    data-testid="empty-experiments"
                  >
                    0 experiments
                  </td>
                </tr>
              ) : count === null ? (
                <tr>
                  <td
                    colSpan={4}
                    className="px-4 py-16 text-center text-[var(--muted)]"
                  >
                    Loading…
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </main>
    </div>
  );
}
