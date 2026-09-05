import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

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
  param_count?: number | null;
  wandb_url: string | null;
  step: number;
  total_steps: number;
  error?: string | null;
};

type ExperimentsResponse = {
  experiments: RunRow[];
  count: number;
};

type HistoryPoint = { step: number; mean_return: number | null };

type CompareRun = {
  id: string;
  name: string;
  arch: string;
  status: string;
  param_count: number | null;
  mean_return: number | null;
  wandb_url: string | null;
  wandb_run_id: string | null;
  history: HistoryPoint[];
  error: string | null;
};

type CompareResponse = {
  metric_key: string;
  runs: CompareRun[];
};

const SERIES_COLORS = ["#0f766e", "#b45309", "#1d4ed8", "#be123c", "#7c3aed"];

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

function archCfgFor(arch: string): Record<string, unknown> {
  if (arch === "kan") {
    return { hidden_sizes: [32, 32], grid_size: 5, spline_order: 3 };
  }
  return { hidden_sizes: [64, 64], activation: "tanh" };
}

function seriesKey(run: CompareRun): string {
  return `${run.arch}:${run.id.slice(0, 8)}`;
}

function buildChartData(runs: CompareRun[]): Record<string, number | null>[] {
  const stepSet = new Set<number>();
  for (const r of runs) {
    for (const p of r.history) stepSet.add(p.step);
  }
  const steps = [...stepSet].sort((a, b) => a - b);
  return steps.map((step) => {
    const row: Record<string, number | null> = { step };
    for (const r of runs) {
      const hit = r.history.find((p) => p.step === step);
      row[seriesKey(r)] = hit?.mean_return ?? null;
    }
    return row;
  });
}

export default function App() {
  const [view, setView] = useState<"experiments" | "compare">("experiments");
  const [budget, setBudget] = useState<Budget | null>(null);
  const [wandb, setWandb] = useState<WandbStatus | null>(null);
  const [archs, setArchs] = useState<string[]>(["mlp", "kan"]);
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [count, setCount] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showNewRun, setShowNewRun] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [selectedCompare, setSelectedCompare] = useState<string[]>([]);
  const [compareData, setCompareData] = useState<CompareResponse | null>(null);
  const [comparing, setComparing] = useState(false);
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
      const [budgetRes, experimentsRes, wandbRes, archsRes] = await Promise.all([
        fetch("/api/budget"),
        fetch("/api/experiments"),
        fetch("/api/wandb/status"),
        fetch("/api/archs"),
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
      if (archsRes.ok) {
        const a = (await archsRes.json()) as { archs: string[] };
        if (a.archs?.length) setArchs(a.archs);
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
      const nSteps =
        form.timesteps <= 2048 ? 512 : form.timesteps <= 5000 ? 1024 : 2048;
      const body = {
        name: `${form.task}-${form.arch}-${form.compute}`,
        sim: form.sim,
        task: form.task,
        robot: form.robot,
        arch: form.arch,
        arch_cfg: archCfgFor(form.arch),
        trainer: {
          algo: "ppo",
          timesteps: form.timesteps,
          lr: 0.0003,
          batch_size: 64,
          n_steps: nSteps,
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
      setView("experiments");
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start run");
    } finally {
      setSubmitting(false);
    }
  }

  function toggleCompareSelect(id: string) {
    setSelectedCompare((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id],
    );
  }

  async function loadCompare() {
    if (selectedCompare.length < 1) {
      setError("Select at least one run (pick mlp + kan for the gate).");
      return;
    }
    setComparing(true);
    setError(null);
    try {
      const res = await fetch("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_ids: selectedCompare }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`POST /api/compare failed (${res.status}): ${text}`);
      }
      const data = (await res.json()) as CompareResponse;
      setCompareData(data);
      setView("compare");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Compare failed");
    } finally {
      setComparing(false);
    }
  }

  const chartData = useMemo(
    () => (compareData ? buildChartData(compareData.runs.filter((r) => r.history.length)) : []),
    [compareData],
  );

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

  const selectable = runs.filter((r) => r.wandb_url);

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
        <nav className="mb-6 flex gap-2" data-testid="main-nav">
          <button
            type="button"
            data-testid="nav-experiments"
            className={`rounded px-3 py-1.5 text-sm font-medium ${
              view === "experiments"
                ? "bg-[var(--accent)] text-white"
                : "border border-[var(--border)] bg-[var(--surface)]"
            }`}
            onClick={() => setView("experiments")}
          >
            Experiments
          </button>
          <button
            type="button"
            data-testid="nav-compare"
            className={`rounded px-3 py-1.5 text-sm font-medium ${
              view === "compare"
                ? "bg-[var(--accent)] text-white"
                : "border border-[var(--border)] bg-[var(--surface)]"
            }`}
            onClick={() => setView("compare")}
          >
            Compare
          </button>
        </nav>

        {error && (
          <div
            className="mb-6 rounded border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-800"
            role="alert"
          >
            {error}
          </div>
        )}

        {view === "experiments" && (
          <>
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
                      data-testid="arch-select"
                    >
                      {archs.map((a) => (
                        <option key={a} value={a}>
                          {a}
                        </option>
                      ))}
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
                      data-testid="timesteps-select"
                    >
                      <option value={50_000}>50k</option>
                      <option value={5_000}>5k (smoke)</option>
                      <option value={2_048}>2k (quick)</option>
                    </select>
                  </label>
                </div>
                {form.arch === "kan" && (
                  <p className="mt-3 text-xs text-[var(--muted)]">
                    KAN on Mac CPU is slow — use 5k smoke for a quick gate, or 50k overnight.
                  </p>
                )}
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

            <div className="mb-3 flex flex-wrap items-center gap-3">
              <p className="text-sm text-[var(--muted)]">
                Select runs with W&amp;B links, then open Compare.
              </p>
              <button
                type="button"
                data-testid="open-compare"
                disabled={selectedCompare.length < 1 || comparing}
                onClick={() => void loadCompare()}
                className="rounded border border-[var(--border)] bg-[var(--surface)] px-3 py-1.5 text-sm font-medium disabled:opacity-40"
              >
                {comparing ? "Loading curves…" : `Compare selected (${selectedCompare.length})`}
              </button>
            </div>

            <div className="overflow-hidden rounded-lg border border-[var(--border)] bg-[var(--surface)]">
              <table className="w-full text-left text-sm">
                <thead className="border-b border-[var(--border)] bg-[#f0f4f7] text-[var(--muted)]">
                  <tr>
                    <th className="px-4 py-3 font-medium">Pick</th>
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
                        colSpan={8}
                        className="px-4 py-16 text-center text-[var(--muted)]"
                        data-testid="empty-experiments"
                      >
                        0 experiments
                      </td>
                    </tr>
                  ) : count === null ? (
                    <tr>
                      <td
                        colSpan={8}
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
                        <td className="px-4 py-3">
                          <input
                            type="checkbox"
                            data-testid={`pick-${r.id}`}
                            disabled={!r.wandb_url}
                            checked={selectedCompare.includes(r.id)}
                            onChange={() => toggleCompareSelect(r.id)}
                            title={
                              r.wandb_url
                                ? "Include in Compare"
                                : "Needs wandb_url"
                            }
                          />
                        </td>
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
                            <p
                              className="mt-1 max-w-xs text-xs text-red-700"
                              title={r.error}
                            >
                              {r.error.slice(0, 120)}
                              {r.error.length > 120 ? "…" : ""}
                            </p>
                          ) : null}
                        </td>
                        <td className="px-4 py-3 min-w-[140px]">
                          <div className="h-2 w-full overflow-hidden rounded bg-slate-200">
                            <div
                              className="h-full bg-[var(--accent)] transition-all duration-500"
                              style={{
                                width: `${Math.round((r.progress || 0) * 100)}%`,
                              }}
                              data-testid="progress-bar"
                            />
                          </div>
                          <p className="mt-1 text-xs text-[var(--muted)]">
                            {r.step.toLocaleString()} / {r.total_steps.toLocaleString()}
                          </p>
                        </td>
                        <td
                          className="px-4 py-3 tabular-nums"
                          data-testid="mean-return"
                        >
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
            {selectable.length === 0 && count !== null && count > 0 && (
              <p className="mt-3 text-sm text-[var(--muted)]">
                No runs with W&amp;B URLs yet — finish a training run first.
              </p>
            )}
          </>
        )}

        {view === "compare" && (
          <div data-testid="compare-view">
            <div className="mb-4 flex flex-wrap items-baseline justify-between gap-4">
              <div>
                <h2 className="text-lg font-medium text-[var(--text)]">Compare</h2>
                <p className="text-sm text-[var(--muted)]">
                  Overlay of real W&amp;B{" "}
                  <code className="text-xs">{compareData?.metric_key ?? "rollout/ep_rew_mean"}</code>{" "}
                  curves
                </p>
              </div>
              <button
                type="button"
                className="rounded border border-[var(--border)] px-3 py-1.5 text-sm"
                onClick={() => setView("experiments")}
              >
                Back to experiments
              </button>
            </div>

            {!compareData && (
              <p className="text-sm text-[var(--muted)]">
                Pick runs on Experiments, then click <strong>Compare selected</strong>.
              </p>
            )}

            {compareData && (
              <>
                <div
                  className="mb-6 h-80 w-full rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4"
                  data-testid="compare-chart"
                >
                  {chartData.length === 0 ? (
                    <p className="flex h-full items-center justify-center text-sm text-[var(--muted)]">
                      No history points returned from W&amp;B for the selected runs.
                    </p>
                  ) : (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                        <XAxis
                          dataKey="step"
                          type="number"
                          tick={{ fontSize: 12 }}
                          label={{
                            value: "step",
                            position: "insideBottom",
                            offset: -2,
                            fontSize: 12,
                          }}
                        />
                        <YAxis
                          tick={{ fontSize: 12 }}
                          label={{
                            value: "mean return",
                            angle: -90,
                            position: "insideLeft",
                            fontSize: 12,
                          }}
                        />
                        <Tooltip />
                        <Legend />
                        {compareData.runs
                          .filter((r) => r.history.length > 0)
                          .map((r, i) => (
                            <Line
                              key={r.id}
                              type="monotone"
                              dataKey={seriesKey(r)}
                              name={`${r.arch} (${r.id.slice(0, 8)})`}
                              stroke={SERIES_COLORS[i % SERIES_COLORS.length]}
                              dot={false}
                              connectNulls
                              strokeWidth={2}
                            />
                          ))}
                      </LineChart>
                    </ResponsiveContainer>
                  )}
                </div>

                <div className="overflow-hidden rounded-lg border border-[var(--border)] bg-[var(--surface)]">
                  <table className="w-full text-left text-sm" data-testid="compare-table">
                    <thead className="border-b border-[var(--border)] bg-[#f0f4f7] text-[var(--muted)]">
                      <tr>
                        <th className="px-4 py-3 font-medium">Run</th>
                        <th className="px-4 py-3 font-medium">Arch</th>
                        <th className="px-4 py-3 font-medium">Param count</th>
                        <th className="px-4 py-3 font-medium">Final mean return</th>
                        <th className="px-4 py-3 font-medium">History pts</th>
                        <th className="px-4 py-3 font-medium">W&amp;B</th>
                      </tr>
                    </thead>
                    <tbody>
                      {compareData.runs.map((r) => (
                        <tr
                          key={r.id}
                          className="border-b border-[var(--border)] last:border-0"
                          data-testid={`compare-row-${r.id}`}
                        >
                          <td className="px-4 py-3 font-medium">
                            {r.name}
                            <span className="ml-2 text-xs text-[var(--muted)]">
                              {r.id}
                            </span>
                            {r.error ? (
                              <p className="mt-1 text-xs text-red-700">{r.error}</p>
                            ) : null}
                          </td>
                          <td className="px-4 py-3">{r.arch}</td>
                          <td className="px-4 py-3 tabular-nums" data-testid="param-count">
                            {r.param_count == null
                              ? "—"
                              : r.param_count.toLocaleString()}
                          </td>
                          <td className="px-4 py-3 tabular-nums">
                            {r.mean_return == null ? "—" : r.mean_return.toFixed(2)}
                          </td>
                          <td className="px-4 py-3 tabular-nums">{r.history.length}</td>
                          <td className="px-4 py-3">
                            {r.wandb_url ? (
                              <a
                                href={r.wandb_url}
                                target="_blank"
                                rel="noreferrer"
                                className="text-[var(--accent)] underline"
                              >
                                {r.wandb_run_id ?? "open"}
                              </a>
                            ) : (
                              "—"
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
