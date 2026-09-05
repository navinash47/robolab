import { useCallback, useEffect, useRef, useState } from "react";

type Budget = {
  budget_usd_cap: number;
  month_spend_usd: number;
  remaining_usd: number;
};

type WandbStatus = {
  present: boolean;
  well_formed: boolean;
  key_len: number;
  auth_ok: boolean | null;
  status: string;
  project: string;
  entity: string | null;
  mode: string | null;
  detail: string;
};

type RunRow = {
  id: string;
  name: string;
  sim: string;
  arch: string;
  task?: string;
  compute?: string;
  status: string;
  progress: number;
  mean_return: number | null;
  wandb_url: string | null;
  step: number;
  total_steps: number;
  error?: string | null;
};

type ExperimentsResponse = {
  experiments: RunRow[];
  count: number;
};

function formatUsd(n: number): string {
  return n.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  });
}

function statusClass(status: string): string {
  switch (status) {
    case "COMPLETE":
      return "bg-emerald-100 text-emerald-800";
    case "RUNNING":
    case "QUEUED":
      return "bg-sky-100 text-sky-800";
    case "FAILED":
    case "KILLED_BY_WATCHDOG":
      return "bg-red-100 text-red-800";
    default:
      return "bg-slate-100 text-slate-700";
  }
}

export default function App() {
  const [budget, setBudget] = useState<Budget | null>(null);
  const [wandb, setWandb] = useState<WandbStatus | null>(null);
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [count, setCount] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showNewRun, setShowNewRun] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [form, setForm] = useState({
    compute: "local",
    arch: "mlp",
    task: "wall_follow",
    sim: "mujoco",
    robot: "diffdrive_lidar",
    timesteps: 50_000,
  });
  const esRef = useRef<Map<string, EventSource>>(new Map());

  const refresh = useCallback(async () => {
    try {
      const [budgetRes, experimentsRes, wandbRes] = await Promise.all([
        fetch("/api/budget"),
        fetch("/api/experiments"),
        fetch("/api/wandb/status"),
      ]);
      if (!budgetRes.ok || !experimentsRes.ok) {
        throw new Error(
          `API error: budget ${budgetRes.status}, experiments ${experimentsRes.status}`,
        );
      }
      const budgetJson: Budget = await budgetRes.json();
      const experimentsJson: ExperimentsResponse = await experimentsRes.json();
      setBudget(budgetJson);
      setRuns(experimentsJson.experiments);
      setCount(experimentsJson.count);
      if (wandbRes.ok) {
        setWandb((await wandbRes.json()) as WandbStatus);
      } else {
        setWandb(null);
      }
      setError(null);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to reach RoboLab API (is make dev running?)",
      );
    }
  }, []);

  useEffect(() => {
    void refresh();
    const t = setInterval(() => void refresh(), 5000);
    return () => clearInterval(t);
  }, [refresh]);

  // Subscribe SSE for live runs
  useEffect(() => {
    const live = runs.filter(
      (r) => r.status === "RUNNING" || r.status === "QUEUED",
    );
    const map = esRef.current;
    for (const run of live) {
      if (map.has(run.id)) continue;
      const es = new EventSource(`/api/runs/${run.id}/events`);
      es.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data) as {
            id: string;
            status: string;
            step: number;
            total: number;
            mean_return: number | null;
            wandb_url: string | null;
            progress: number;
            error?: string | null;
          };
          setRuns((prev) =>
            prev.map((r) =>
              r.id === data.id
                ? {
                    ...r,
                    status: data.status,
                    step: data.step,
                    total_steps: data.total,
                    mean_return: data.mean_return,
                    wandb_url: data.wandb_url,
                    progress: data.progress,
                    error: data.error ?? null,
                  }
                : r,
            ),
          );
          if (
            data.status === "COMPLETE" ||
            data.status === "FAILED" ||
            data.status === "KILLED_BY_WATCHDOG"
          ) {
            es.close();
            map.delete(data.id);
            void refresh();
          }
        } catch {
          /* ignore malformed */
        }
      };
      es.onerror = () => {
        /* browser will retry; refresh on interval as backup */
      };
      map.set(run.id, es);
    }
    // Close SSE for runs no longer live
    for (const [id, es] of map) {
      if (!live.some((r) => r.id === id)) {
        es.close();
        map.delete(id);
      }
    }
  }, [runs, refresh]);

  async function startRun() {
    setSubmitting(true);
    setError(null);
    try {
      const body = {
        name: `${form.task}-${form.arch}-${form.compute}`,
        sim: form.sim,
        task: form.task,
        robot: form.robot,
        arch: form.arch,
        arch_cfg: { hidden_sizes: [64, 64], activation: "tanh" },
        trainer: {
          algo: "ppo",
          timesteps: form.timesteps,
          lr: 0.0003,
          batch_size: 64,
          n_steps: 2048,
          n_envs: 1,
          gamma: 0.99,
          device: "cpu",
          seed: 0,
        },
        seeds: [0],
        eval_suite: [],
        domain: {
          control_hz: 50,
          physics_substeps: 5,
          friction: 1.0,
          mass_scale: 1.0,
          sensor_noise_std: 0.0,
          action_delay_steps: 0,
        },
        compute: form.compute,
        gpu_type: null,
        budget_usd: 0,
        transfer_to: null,
      };
      const res = await fetch("/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`POST /api/runs failed (${res.status}): ${text}`);
      }
      setShowNewRun(false);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start run");
    } finally {
      setSubmitting(false);
    }
  }

  const remainingLabel =
    budget === null
      ? "Budget: …"
      : `Budget: ${formatUsd(budget.remaining_usd)} remaining`;

  const wandbLabel = (() => {
    if (wandb === null) return "W&B: …";
    switch (wandb.status) {
      case "ok":
        return "W&B: key valid";
      case "missing":
        return "W&B: key missing";
      case "invalid_shape":
        return "W&B: key invalid (truncated?)";
      case "invalid":
        return "W&B: key rejected (401)";
      case "unreachable":
        return "W&B: unreachable";
      default:
        return `W&B: ${wandb.status}`;
    }
  })();

  const wandbOk = wandb?.status === "ok";

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
          <p
            className={`text-xs ${wandbOk ? "text-[var(--ok)]" : "text-red-700"}`}
            data-testid="wandb-status"
            title={wandb?.detail ?? ""}
          >
            {wandbLabel}
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
            className="mb-6 rounded border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-800"
            role="alert"
          >
            {error}
          </div>
        )}

        <div className="mb-4 flex items-baseline justify-between gap-4">
          <div>
            <h2 className="text-lg font-medium text-[var(--text)]">Experiments</h2>
            <p className="text-sm text-[var(--muted)]" data-testid="experiment-count">
              {experimentsLabel}
            </p>
          </div>
          <button
            type="button"
            data-testid="new-run"
            onClick={() => setShowNewRun(true)}
            className="rounded bg-[var(--accent)] px-4 py-2 text-sm font-medium text-white hover:opacity-90"
          >
            New Run
          </button>
        </div>

        {showNewRun && (
          <div
            className="mb-6 rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5"
            data-testid="new-run-form"
          >
            <h3 className="mb-4 text-base font-medium">New Run</h3>
            <div className="grid gap-4 sm:grid-cols-2">
              <label className="text-sm">
                <span className="mb-1 block text-[var(--muted)]">Compute</span>
                <select
                  className="w-full rounded border border-[var(--border)] bg-white px-3 py-2"
                  value={form.compute}
                  onChange={(e) => setForm({ ...form, compute: e.target.value })}
                >
                  <option value="local">local</option>
                </select>
              </label>
              <label className="text-sm">
                <span className="mb-1 block text-[var(--muted)]">Arch</span>
                <select
                  className="w-full rounded border border-[var(--border)] bg-white px-3 py-2"
                  value={form.arch}
                  onChange={(e) => setForm({ ...form, arch: e.target.value })}
                >
                  <option value="mlp">mlp</option>
                </select>
              </label>
              <label className="text-sm">
                <span className="mb-1 block text-[var(--muted)]">Task</span>
                <select
                  className="w-full rounded border border-[var(--border)] bg-white px-3 py-2"
                  value={form.task}
                  onChange={(e) => setForm({ ...form, task: e.target.value })}
                >
                  <option value="wall_follow">wall_follow</option>
                </select>
              </label>
              <label className="text-sm">
                <span className="mb-1 block text-[var(--muted)]">Timesteps</span>
                <select
                  className="w-full rounded border border-[var(--border)] bg-white px-3 py-2"
                  value={form.timesteps}
                  onChange={(e) =>
                    setForm({ ...form, timesteps: Number(e.target.value) })
                  }
                >
                  <option value={50_000}>50k</option>
                  <option value={5_000}>5k (smoke)</option>
                  <option value={2_048}>2k (quick)</option>
                </select>
              </label>
            </div>
            <div className="mt-4 flex gap-3">
              <button
                type="button"
                disabled={submitting}
                onClick={() => void startRun()}
                className="rounded bg-[var(--accent)] px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
                data-testid="start-run"
              >
                {submitting ? "Starting…" : "Start"}
              </button>
              <button
                type="button"
                onClick={() => setShowNewRun(false)}
                className="rounded border border-[var(--border)] px-4 py-2 text-sm"
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        <div className="overflow-hidden rounded-lg border border-[var(--border)] bg-[var(--surface)]">
          <table className="w-full text-left text-sm">
            <thead className="border-b border-[var(--border)] bg-[#f0f4f7] text-[var(--muted)]">
              <tr>
                <th className="px-4 py-3 font-medium">Name</th>
                <th className="px-4 py-3 font-medium">Sim</th>
                <th className="px-4 py-3 font-medium">Arch</th>
                <th className="px-4 py-3 font-medium">Status</th>
                <th className="px-4 py-3 font-medium">Progress</th>
                <th className="px-4 py-3 font-medium">Mean return</th>
                <th className="px-4 py-3 font-medium">W&amp;B</th>
              </tr>
            </thead>
            <tbody>
              {count === 0 ? (
                <tr>
                  <td
                    colSpan={7}
                    className="px-4 py-16 text-center text-[var(--muted)]"
                    data-testid="empty-experiments"
                  >
                    0 experiments
                  </td>
                </tr>
              ) : count === null ? (
                <tr>
                  <td
                    colSpan={7}
                    className="px-4 py-16 text-center text-[var(--muted)]"
                  >
                    Loading…
                  </td>
                </tr>
              ) : (
                runs.map((r) => (
                  <tr
                    key={r.id}
                    className="border-b border-[var(--border)] last:border-0"
                    data-testid={`run-${r.id}`}
                  >
                    <td className="px-4 py-3 font-medium">{r.name}</td>
                    <td className="px-4 py-3">{r.sim}</td>
                    <td className="px-4 py-3">{r.arch}</td>
                    <td className="px-4 py-3">
                      <span
                        className={`inline-block rounded px-2 py-0.5 text-xs font-medium ${statusClass(r.status)}`}
                        data-testid="status-chip"
                      >
                        {r.status}
                      </span>
                      {r.error ? (
                        <p className="mt-1 max-w-xs text-xs text-red-700" title={r.error}>
                          {r.error.slice(0, 120)}
                          {r.error.length > 120 ? "…" : ""}
                        </p>
                      ) : null}
                    </td>
                    <td className="px-4 py-3 min-w-[140px]">
                      <div className="h-2 w-full overflow-hidden rounded bg-slate-200">
                        <div
                          className="h-full bg-[var(--accent)] transition-all duration-500"
                          style={{ width: `${Math.round((r.progress || 0) * 100)}%` }}
                          data-testid="progress-bar"
                        />
                      </div>
                      <p className="mt-1 text-xs text-[var(--muted)]">
                        {r.step.toLocaleString()} / {r.total_steps.toLocaleString()}
                      </p>
                    </td>
                    <td className="px-4 py-3 tabular-nums" data-testid="mean-return">
                      {r.mean_return == null ? "—" : r.mean_return.toFixed(2)}
                    </td>
                    <td className="px-4 py-3">
                      {r.wandb_url ? (
                        <a
                          href={r.wandb_url}
                          target="_blank"
                          rel="noreferrer"
                          className="text-[var(--accent)] underline"
                          data-testid="wandb-link"
                        >
                          W&amp;B
                        </a>
                      ) : (
                        <span className="text-[var(--muted)]">—</span>
                      )}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </main>
    </div>
  );
}
