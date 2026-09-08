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
import {
  EMPTY_FILTERS,
  SORT_OPTIONS,
  filtersActive,
  queryExperiments,
  uniqueValues,
  type ExperimentFilters,
  type SortKey,
} from "./experimentsQuery";
import { ArchitecturesList } from "./ArchitecturesList";
import { ArchHoverLabel } from "./ArchHover";
import { builtinMeta, BUILTIN_DEFAULT_CFG, PAGE_SIZES, type PageSize } from "./archMeta";

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
  obs_dim?: number | null;
  act_dim?: number | null;
  control_hz?: number | null;
  physics_substeps?: number | null;
  wandb_url: string | null;
  step: number;
  total_steps: number;
  error?: string | null;
  pod_id?: string | null;
  gpu_type?: string | null;
  hourly_rate?: number | null;
  cost_usd?: number | null;
  budget_usd?: number | null;
  video_status?: string | null;
  video_url?: string | null;
  video_error?: string | null;
  checkpoint_artifact?: string | null;
  checkpoints?: Array<{
    kind: string;
    name: string;
    path?: string | null;
    exists?: boolean;
  }>;
  transfer_status?: string | null;
  transfer_error?: string | null;
  transfer_summary?: {
    source_mean?: number | null;
    targets?: Array<{
      sim?: string;
      status?: string;
      mean?: number | null;
      transfer_ratio?: number | null;
      gap?: number | null;
    }>;
  } | null;
  created_at?: string | null;
  config?: Record<string, unknown>;
  group?: string | null;
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

type TransferReport = {
  run_id: string;
  status: string;
  source_sim?: string;
  arch?: string;
  task?: string;
  n_episodes?: number;
  source?: {
    sim?: string;
    status?: string;
    mean?: number | null;
    std?: number | null;
    ci_low?: number | null;
    ci_high?: number | null;
  };
  targets?: Array<{
    sim?: string;
    status?: string;
    mean?: number | null;
    ci_low?: number | null;
    ci_high?: number | null;
    transfer_ratio?: number | null;
    gap?: number | null;
    reason?: string;
  }>;
  robustness?: {
    friction?: Array<{ sim: string; value: number; mean_return: number | null; status?: string }>;
    mass_scale?: Array<{ sim: string; value: number; mean_return: number | null; status?: string }>;
    sensor_noise_std?: Array<{
      sim: string;
      value: number;
      mean_return: number | null;
      status?: string;
    }>;
  };
  error?: string;
};


type SavedArch = {
  id: string;
  name: string;
  base_arch: string;
  cfg: Record<string, unknown>;
  notes?: string;
};

type ArchsResponse = {
  archs: string[];
  saved?: SavedArch[];
  defaults?: Record<string, Record<string, unknown>>;
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
    case "PROVISIONING":
      return "bg-sky-100 text-sky-800";
    case "ABORTED":
      return "bg-amber-100 text-amber-900";
    case "FAILED":
    case "KILLED_BY_WATCHDOG":
      return "bg-red-100 text-red-800";
    default:
      return "bg-slate-100 text-slate-700";
  }
}

function archCfgFor(
  arch: string,
  defaults?: Record<string, Record<string, unknown>>,
  saved?: SavedArch[],
): Record<string, unknown> {
  if (arch.startsWith("custom:")) {
    const id = arch.slice("custom:".length);
    const hit = saved?.find((s) => s.id === id);
    if (hit) return { ...hit.cfg };
  }
  const byName = saved?.find((s) => s.name === arch);
  if (byName) return { ...byName.cfg };
  if (defaults?.[arch]) return { ...defaults[arch] };
  if (arch === "kan") {
    return { hidden_sizes: [32, 32], grid_size: 5, spline_order: 3 };
  }
  if (arch === "kaf") {
    return {
      hidden_sizes: [64, 64],
      num_grids: 8,
      activation_expectation: 1.64,
      use_layernorm: true,
      spline_dropout: 0.0,
    };
  }
  if (arch === "gpkan") {
    return {
      hidden_sizes: [32, 32],
      num_basis: 8,
      init_bandwidth: 1.0,
      base_activation: "gelu",
    };
  }
  if (arch === "fan") {
    return { hidden_sizes: [64, 64], p_ratio: 0.25, activation: "gelu" };
  }
  if (arch === "avinash_wall") {
    return {
      algorithm: "q_learning",
      alpha: 0.1,
      gamma: 1.0,
      epsilon_start: 1.0,
      epsilon_end: 0.05,
      epsilon_decay: 0.05,
      explore_episodes: 200,
      episode_max_steps: 1200,
      linear_vel: 0.3,
      angular_vel: 0.7,
      near_max: 0.7,
      medium_max: 0.9,
      lr_default: 0.1,
    };
  }
  return { hidden_sizes: [64, 64], activation: "tanh" };
}

function parseHiddenSizes(raw: string): number[] {
  const parts = raw
    .split(/[,\s]+/)
    .map((s) => s.trim())
    .filter(Boolean)
    .map(Number)
    .filter((n) => Number.isFinite(n) && n > 0);
  return parts.length ? parts.map((n) => Math.floor(n)) : [64, 64];
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
  const [view, setView] = useState<"experiments" | "compare" | "architectures" | "transfer">(
    "experiments",
  );
  const [archDetailKey, setArchDetailKey] = useState<string | null>(null);
  const [budget, setBudget] = useState<Budget | null>(null);
  const [wandb, setWandb] = useState<WandbStatus | null>(null);
  const [archs, setArchs] = useState<string[]>([
    "mlp",
    "kan",
    "kaf",
    "gpkan",
    "fan",
    "avinash_wall",
  ]);
  const [savedArchs, setSavedArchs] = useState<SavedArch[]>([]);
  const [archDefaults, setArchDefaults] = useState<
    Record<string, Record<string, unknown>>
  >({});
  const [sims, setSims] = useState<string[]>(["mujoco", "pybullet"]);
  const [simMeta, setSimMeta] = useState<Record<string, { capabilities: string[] }>>({});
  const [tasks, setTasks] = useState<string[]>(["wall_follow"]);
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [count, setCount] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showNewRun, setShowNewRun] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [selectedCompare, setSelectedCompare] = useState<string[]>([]);
  const [compareData, setCompareData] = useState<CompareResponse | null>(null);
  const [comparing, setComparing] = useState(false);
  const [transferRunId, setTransferRunId] = useState<string>("");
  const [transferReport, setTransferReport] = useState<TransferReport | null>(null);
  const [transferLoading, setTransferLoading] = useState(false);
  const [transferRunning, setTransferRunning] = useState(false);
  const [renderingId, setRenderingId] = useState<string | null>(null);
  const [abortingId, setAbortingId] = useState<string | null>(null);
  const [flaggingId, setFlaggingId] = useState<string | null>(null);
  const [detailId, setDetailId] = useState<string | null>(null);
  const [searchInput, setSearchInput] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [filters, setFilters] = useState<ExperimentFilters>(EMPTY_FILTERS);
  const [sortKey, setSortKey] = useState<SortKey>("created_desc");
  const [expPageSize, setExpPageSize] = useState<PageSize>(10);
  const [expPage, setExpPage] = useState(0);
  const [timestepMode, setTimestepMode] = useState<"preset" | "custom">("preset");
  const [form, setForm] = useState({
    compute: "local",
    arch: "mlp",
    algo: "ppo" as "ppo" | "q_learning",
    task: "wall_follow",
    sim: "mujoco",
    robot: "diffdrive_lidar",
    timesteps: 50_000,
    gpu_type: "best",
    budget_usd: 0,
    transfer_to: [] as string[],
  });
  const [builderBase, setBuilderBase] = useState("kaf");
  const [builderName, setBuilderName] = useState("kaf_wall_follow");
  const [builderHidden, setBuilderHidden] = useState("64, 64");
  const [builderNumGrids, setBuilderNumGrids] = useState(8);
  const [builderNumBasis, setBuilderNumBasis] = useState(8);
  const [builderPRatio, setBuilderPRatio] = useState(0.25);
  const [builderLayernorm, setBuilderLayernorm] = useState(true);
  const [builderActExpect, setBuilderActExpect] = useState(1.64);
  const [builderBandwidth, setBuilderBandwidth] = useState(1.0);
  const [builderGridSize, setBuilderGridSize] = useState(5);
  const [builderSplineOrder, setBuilderSplineOrder] = useState(3);
  const [builderActivation, setBuilderActivation] = useState("gelu");
  const [builderLr, setBuilderLr] = useState(0.0003);
  const [builderNotes, setBuilderNotes] = useState("");
  const [builderSaving, setBuilderSaving] = useState(false);
  const esRef = useRef<Map<string, EventSource>>(new Map());
  const detailRef = useRef<HTMLDivElement>(null);

  function openRunDetail(runId: string) {
    setDetailId(runId);
  }

  useEffect(() => {
    if (!detailId) return;
    detailRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [detailId]);

  useEffect(() => {
    const t = window.setTimeout(() => setDebouncedSearch(searchInput), 220);
    return () => window.clearTimeout(t);
  }, [searchInput]);

  const refresh = useCallback(async () => {
    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), 12_000);
    try {
      const signal = controller.signal;
      const [budgetRes, experimentsRes, wandbRes, archsRes, simsRes, tasksRes] =
        await Promise.all([
          fetch("/api/budget", { signal }),
          fetch("/api/experiments", { signal }),
          fetch("/api/wandb/status", { signal }),
          fetch("/api/archs", { signal }),
          fetch("/api/sims", { signal }),
          fetch("/api/tasks", { signal }),
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
        const a = (await archsRes.json()) as ArchsResponse;
        if (a.archs?.length) {
          // Keep known builtins visible even if a stale API process omits a new arch.
          const known = Object.keys(BUILTIN_DEFAULT_CFG);
          setArchs(
            Array.from(new Set([...a.archs, ...known])).sort((x, y) =>
              x.localeCompare(y),
            ),
          );
        }
        if (a.saved) setSavedArchs(a.saved);
        if (a.defaults) setArchDefaults(a.defaults);
      }
      if (simsRes.ok) {
        const s = (await simsRes.json()) as {
          sims: string[];
          meta?: Record<string, { capabilities: string[] }>;
        };
        if (s.sims?.length) setSims(s.sims);
        if (s.meta) setSimMeta(s.meta);
      }
      if (tasksRes.ok) {
        const t = (await tasksRes.json()) as { tasks: string[] };
        if (t.tasks?.length) setTasks(t.tasks);
      }
      setError(null);
    } catch (err) {
      const timedOut =
        err instanceof DOMException && err.name === "AbortError";
      setError(
        timedOut
          ? "API unreachable (timed out). Is the backend on :8000 up? Try: launchctl kickstart -k gui/$(id -u)/com.robolab.api8000"
          : err instanceof Error
            ? err.message
            : "Failed to reach RoboLab API (is :8000 up?)",
      );
    } finally {
      window.clearTimeout(timeoutId);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const t = setInterval(() => void refresh(), 5000);
    return () => clearInterval(t);
  }, [refresh]);

  useEffect(() => {
    const live = runs.filter(
      (r) =>
        r.status === "RUNNING" ||
        r.status === "QUEUED" ||
        r.status === "PROVISIONING",
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
            pod_id?: string | null;
            hourly_rate?: number | null;
            cost_usd?: number | null;
            gpu_type?: string | null;
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
                    pod_id: data.pod_id ?? r.pod_id,
                    hourly_rate: data.hourly_rate ?? r.hourly_rate,
                    cost_usd: data.cost_usd ?? r.cost_usd,
                    gpu_type: data.gpu_type ?? r.gpu_type,
                  }
                : r,
            ),
          );
          if (
            data.status === "COMPLETE" ||
            data.status === "FAILED" ||
            data.status === "KILLED_BY_WATCHDOG" ||
            data.status === "ABORTED"
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
      const steps = Math.floor(Number(form.timesteps));
      if (!Number.isFinite(steps) || steps < 1 || steps > 10_000_000) {
        throw new Error("Timesteps must be an integer between 1 and 10,000,000");
      }
      const nSteps = steps <= 2048 ? 512 : steps <= 5000 ? 1024 : 2048;
      const isRunpod = form.compute === "runpod";
      const archValue = form.arch;
      const cfg = archCfgFor(archValue, archDefaults, savedArchs);
      const isAvinash = archValue === "avinash_wall";
      const algo: "ppo" | "q_learning" = isAvinash
        ? "q_learning"
        : form.algo === "q_learning"
          ? "q_learning"
          : "ppo";
      const isQl = algo === "q_learning";
      // Neural FA Q-learning: PDF wall-follow task knobs; ε scales with total_timesteps.
      const qlSchedule = {
        algorithm: "q_learning",
        epsilon_start: 1.0,
        epsilon_end: 0.05,
        epsilon_decay: 0.05,
        explore_episodes: 200,
        episode_max_steps: 1200,
        linear_vel: 0.3,
        angular_vel: 0.7,
        near_max: 0.7,
        medium_max: 0.9,
        gamma: 1.0,
      };
      const mergedCfg =
        isQl && !isAvinash
          ? { ...qlSchedule, ...cfg, gamma: typeof cfg.gamma === "number" ? cfg.gamma : 1.0 }
          : cfg;
      const lrHint =
        typeof mergedCfg.lr_default === "number"
          ? Number(mergedCfg.lr_default)
          : isAvinash
            ? 0.1
            : isQl
              ? 0.0003
              : 0.0003;
      const gammaHint = isQl
        ? typeof mergedCfg.gamma === "number"
          ? Number(mergedCfg.gamma)
          : 1.0
        : 0.99;
      const body = {
        name: `${form.task}-${form.arch.replace(/^custom:/, "")}-${algo}-${form.compute}`,
        sim: form.sim,
        task: form.task,
        robot: form.robot,
        arch: archValue,
        arch_cfg: mergedCfg,
        trainer: {
          algo,
          timesteps: steps,
          lr: lrHint,
          batch_size: 64,
          n_steps: nSteps,
          n_envs: 1,
          gamma: gammaHint,
          device: isRunpod ? "cuda" : "cpu",
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
        gpu_type: isRunpod ? form.gpu_type : null,
        budget_usd: isRunpod ? Number(form.budget_usd) || 0 : 0,
        transfer_to: form.transfer_to.length > 0 ? form.transfer_to : null,
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

  function builderCfgFromForm(): Record<string, unknown> {
    const hidden = parseHiddenSizes(builderHidden);
    const base: Record<string, unknown> = {
      hidden_sizes: hidden,
      lr_default: builderLr,
    };
    if (builderBase === "kaf") {
      return {
        ...base,
        num_grids: builderNumGrids,
        activation_expectation: builderActExpect,
        use_layernorm: builderLayernorm,
        spline_dropout: 0.0,
      };
    }
    if (builderBase === "gpkan") {
      return {
        ...base,
        num_basis: builderNumBasis,
        init_bandwidth: builderBandwidth,
        base_activation: "gelu",
      };
    }
    if (builderBase === "fan") {
      return {
        ...base,
        p_ratio: builderPRatio,
        activation: builderActivation,
      };
    }
    if (builderBase === "kan") {
      return {
        ...base,
        grid_size: builderGridSize,
        spline_order: builderSplineOrder,
      };
    }
    return { ...base, activation: builderActivation === "gelu" ? "tanh" : builderActivation };
  }

  function applyBuilderBase(base: string) {
    setBuilderBase(base);
    const d = archDefaults[base] ?? archCfgFor(base);
    const hs = (d.hidden_sizes as number[] | undefined) ?? [64, 64];
    setBuilderHidden(hs.join(", "));
    if (typeof d.num_grids === "number") setBuilderNumGrids(d.num_grids);
    if (typeof d.num_basis === "number") setBuilderNumBasis(d.num_basis);
    if (typeof d.p_ratio === "number") setBuilderPRatio(d.p_ratio);
    if (typeof d.use_layernorm === "boolean") setBuilderLayernorm(d.use_layernorm);
    if (typeof d.activation_expectation === "number")
      setBuilderActExpect(d.activation_expectation);
    if (typeof d.init_bandwidth === "number") setBuilderBandwidth(d.init_bandwidth);
    if (typeof d.grid_size === "number") setBuilderGridSize(d.grid_size);
    if (typeof d.spline_order === "number") setBuilderSplineOrder(d.spline_order);
    if (typeof d.activation === "string") setBuilderActivation(d.activation);
    if (typeof d.lr_default === "number") setBuilderLr(d.lr_default);
    if (!builderName || builderName.endsWith("_wall_follow") || archs.includes(builderName.split("_")[0] ?? "")) {
      setBuilderName(`${base}_wall_follow`);
    }
  }

  async function saveArchitecture() {
    setBuilderSaving(true);
    setError(null);
    try {
      const res = await fetch("/api/architectures", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: builderName.trim(),
          base_arch: builderBase,
          cfg: builderCfgFromForm(),
          notes: builderNotes,
        }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`POST /api/architectures failed (${res.status}): ${text}`);
      }
      const saved = (await res.json()) as SavedArch;
      setForm((f) => ({ ...f, arch: `custom:${saved.id}` }));
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save architecture");
    } finally {
      setBuilderSaving(false);
    }
  }

  async function deleteArchitecture(id: string) {
    if (!window.confirm("Delete this saved architecture?")) return;
    setError(null);
    try {
      const res = await fetch(`/api/architectures/${id}`, { method: "DELETE" });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`DELETE failed (${res.status}): ${text}`);
      }
      if (form.arch === `custom:${id}`) {
        setForm((f) => ({ ...f, arch: "mlp" }));
      }
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete architecture");
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

  async function loadTransferReport(runId: string) {
    if (!runId) {
      setTransferReport(null);
      return;
    }
    setTransferLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/runs/${runId}/transfer`);
      if (res.status === 404) {
        setTransferReport(null);
        return;
      }
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`GET transfer failed (${res.status}): ${text}`);
      }
      const data = (await res.json()) as TransferReport;
      setTransferReport(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Transfer load failed");
    } finally {
      setTransferLoading(false);
    }
  }

  async function startTransferEval() {
    if (!transferRunId) {
      setError("Pick a COMPLETE run for transfer");
      return;
    }
    setTransferRunning(true);
    setError(null);
    try {
      const run = runs.find((r) => r.id === transferRunId);
      const defaultTarget = (run?.sim || "mujoco") === "mujoco" ? "pybullet" : "mujoco";
      const res = await fetch(`/api/runs/${transferRunId}/transfer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ targets: [defaultTarget], n_episodes: 3 }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`POST transfer failed (${res.status}): ${text}`);
      }
      for (let i = 0; i < 90; i++) {
        await new Promise((r) => setTimeout(r, 2000));
        const tr = await fetch(`/api/runs/${transferRunId}/transfer`);
        if (tr.status === 404) continue;
        if (!tr.ok) continue;
        const data = (await tr.json()) as TransferReport;
        if (data.status === "RUNNING") continue;
        setTransferReport(data);
        await refresh();
        break;
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Transfer failed");
    } finally {
      setTransferRunning(false);
    }
  }

  async function renderVideo(runId: string) {
    setRenderingId(runId);
    openRunDetail(runId);
    setError(null);
    try {
      const res = await fetch(`/api/runs/${runId}/render`, { method: "POST" });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`POST /api/runs/${runId}/render failed (${res.status}): ${text}`);
      }
      const data = (await res.json()) as RunRow;
      setRuns((prev) => prev.map((r) => (r.id === runId ? { ...r, ...data } : r)));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Render failed");
      setRenderingId(null);
    }
  }

  async function abortRun(run: RunRow) {
    const ok = window.confirm(
      `Abort run ${run.name}?\n\nThis terminates any RunPod pod / local process and sets status to ABORTED.`,
    );
    if (!ok) return;
    setAbortingId(run.id);
    setError(null);
    try {
      const res = await fetch(`/api/runs/${run.id}/abort`, { method: "POST" });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`POST /api/runs/${run.id}/abort failed (${res.status}): ${text}`);
      }
      const data = (await res.json()) as { run?: RunRow; status?: string };
      if (data.run) {
        setRuns((prev) => prev.map((r) => (r.id === run.id ? { ...r, ...data.run } : r)));
      } else {
        setRuns((prev) =>
          prev.map((r) =>
            r.id === run.id
              ? { ...r, status: data.status || "ABORTED", error: "aborted by user" }
              : r,
          ),
        );
      }
      void refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Abort failed");
    } finally {
      setAbortingId(null);
    }
  }

  async function flagExperimentFailure(run: RunRow) {
    const reason = window.prompt(
      `Flag experiment failure for ${run.name}\n\nWhat went wrong (science/engineering)?`,
      run.error || "",
    );
    if (reason == null) return;
    const trimmed = reason.trim();
    if (!trimmed) {
      setError("Experiment failure reason is required");
      return;
    }
    setFlaggingId(run.id);
    setError(null);
    try {
      const res = await fetch("/api/failures", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          category: "experiment",
          run_id: run.id,
          title: `${run.name}: experiment failure`,
          reason: trimmed,
        }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`POST /api/failures failed (${res.status}): ${text}`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Flag failed");
    } finally {
      setFlaggingId(null);
    }
  }

  // Poll while any run is RENDERING video
  useEffect(() => {
    const busy = runs.some((r) => r.video_status === "RENDERING");
    if (!busy) {
      setRenderingId(null);
      return;
    }
    const t = setInterval(() => void refresh(), 2000);
    return () => clearInterval(t);
  }, [runs, refresh]);

  const chartData = useMemo(
    () => (compareData ? buildChartData(compareData.runs.filter((r) => r.history.length)) : []),
    [compareData],
  );

  const displayedRuns = useMemo(
    () => queryExperiments(runs, debouncedSearch, filters, sortKey),
    [runs, debouncedSearch, filters, sortKey],
  );

  const expTotalPages = Math.max(1, Math.ceil(displayedRuns.length / expPageSize));
  const expPageClamped = Math.min(expPage, expTotalPages - 1);
  const pagedRuns = useMemo(
    () =>
      displayedRuns.slice(
        expPageClamped * expPageSize,
        expPageClamped * expPageSize + expPageSize,
      ),
    [displayedRuns, expPageClamped, expPageSize],
  );
  const expFrom =
    displayedRuns.length === 0 ? 0 : expPageClamped * expPageSize + 1;
  const expTo = Math.min(
    displayedRuns.length,
    (expPageClamped + 1) * expPageSize,
  );

  useEffect(() => {
    setExpPage(0);
  }, [debouncedSearch, filters, sortKey, expPageSize]);

  useEffect(() => {
    if (expPage > expTotalPages - 1) {
      setExpPage(Math.max(0, expTotalPages - 1));
    }
  }, [expPage, expTotalPages]);

  const filterOptions = useMemo(
    () => ({
      sim: uniqueValues(runs, "sim"),
      arch: uniqueValues(runs, "arch"),
      status: uniqueValues(runs, "status"),
      compute: uniqueValues(runs, "compute"),
      group: uniqueValues(runs, "group"),
    }),
    [runs],
  );

  const hasActiveFilters = filtersActive(filters);
  const hasQueryOrFilters = Boolean(debouncedSearch.trim()) || hasActiveFilters;

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
    count === null
      ? "…"
      : hasQueryOrFilters
        ? `${displayedRuns.length} of ${count} experiment${count === 1 ? "" : "s"}`
        : `${count} experiment${count === 1 ? "" : "s"}`;

  const selectable = runs.filter((r) => r.wandb_url);
  const detailRun = detailId ? runs.find((r) => r.id === detailId) ?? null : null;

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
        <div className="flex items-center gap-6">
          <a
            href="http://localhost:8000/failures"
            target="_blank"
            rel="noopener noreferrer"
            data-testid="failures-link"
            className="text-sm font-medium text-[var(--accent)] underline-offset-2 hover:underline"
          >
            Failures
          </a>
          <a
            href="http://localhost:8000/spend"
            target="_blank"
            rel="noopener noreferrer"
            data-testid="spend-link"
            className="text-sm font-medium text-[var(--accent)] underline-offset-2 hover:underline"
          >
            Spend
          </a>
          <a
            href="http://localhost:8000/tokens"
            target="_blank"
            rel="noopener noreferrer"
            data-testid="tokens-link"
            className="text-sm font-medium text-[var(--accent)] underline-offset-2 hover:underline"
          >
            Tokens
          </a>
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
            data-testid="nav-architectures"
            className={`rounded px-3 py-1.5 text-sm font-medium ${
              view === "architectures"
                ? "bg-[var(--accent)] text-white"
                : "border border-[var(--border)] bg-[var(--surface)]"
            }`}
            onClick={() => {
              setArchDetailKey(null);
              setView("architectures");
            }}
          >
            Architectures
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
          <button
            type="button"
            data-testid="nav-transfer"
            className={`rounded px-3 py-1.5 text-sm font-medium ${
              view === "transfer"
                ? "bg-[var(--accent)] text-white"
                : "border border-[var(--border)] bg-[var(--surface)]"
            }`}
            onClick={() => setView("transfer")}
          >
            Transfer
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
                      data-testid="compute-select"
                    >
                      <option value="local">local</option>
                      <option value="runpod">runpod</option>
                    </select>
                  </label>
                  <label className="text-sm">
                    <span className="mb-1 block text-[var(--muted)]">Arch</span>
                    <select
                      className="w-full rounded border border-[var(--border)] bg-white px-3 py-2"
                      value={form.arch}
                      onChange={(e) => {
                        const next = e.target.value;
                        const paperish = ["kaf", "gpkan", "fan"].includes(next) ||
                          savedArchs.some(
                            (s) =>
                              `custom:${s.id}` === next &&
                              ["kaf", "gpkan", "fan"].includes(s.base_arch),
                          );
                        const forceQl = next === "avinash_wall";
                        setForm({
                          ...form,
                          arch: next,
                          algo: forceQl ? "q_learning" : form.algo,
                          timesteps:
                            paperish && form.timesteps < 100_000 && form.algo === "ppo"
                              ? 100_000
                              : form.timesteps,
                        });
                        if (paperish && form.timesteps < 100_000 && form.algo === "ppo") {
                          setTimestepMode("preset");
                        }
                      }}
                      data-testid="arch-select"
                    >
                      <optgroup label="Builtins">
                        {archs.map((a) => {
                          const m = builtinMeta(a);
                          return (
                            <option
                              key={a}
                              value={a}
                              title={m ? `${m.label} — ${m.short}` : a}
                            >
                              {m ? `${m.label} (${a})` : a}
                            </option>
                          );
                        })}
                      </optgroup>
                      {savedArchs.length > 0 && (
                        <optgroup label="Saved">
                          {savedArchs.map((s) => (
                            <option
                              key={s.id}
                              value={`custom:${s.id}`}
                              title={
                                s.notes?.trim()
                                  ? s.notes
                                  : `${s.name} (${s.base_arch})`
                              }
                            >
                              {s.name} ({s.base_arch})
                            </option>
                          ))}
                        </optgroup>
                      )}
                    </select>
                  </label>
                  <label className="text-sm">
                    <span className="mb-1 block text-[var(--muted)]">Algo</span>
                    <select
                      className="w-full rounded border border-[var(--border)] bg-white px-3 py-2"
                      value={form.arch === "avinash_wall" ? "q_learning" : form.algo}
                      disabled={form.arch === "avinash_wall"}
                      onChange={(e) =>
                        setForm({
                          ...form,
                          algo: e.target.value as "ppo" | "q_learning",
                        })
                      }
                      data-testid="algo-select"
                      title={
                        form.arch === "avinash_wall"
                          ? "Tabular avinash_wall always uses Q-learning"
                          : "PPO (SB3) or PDF wall-follow Q-learning with this arch as Q-network"
                      }
                    >
                      <option value="ppo">PPO (SB3)</option>
                      <option value="q_learning">
                        Q-learning (PDF wall-follow)
                      </option>
                    </select>
                  </label>
                  <label className="text-sm">
                    <span className="mb-1 block text-[var(--muted)]">Task</span>
                    <select
                      className="w-full rounded border border-[var(--border)] bg-white px-3 py-2"
                      value={form.task}
                      onChange={(e) => setForm({ ...form, task: e.target.value })}
                      data-testid="task-select"
                    >
                      {tasks.map((t) => (
                        <option key={t} value={t}>
                          {t}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="text-sm">
                    <span className="mb-1 block text-[var(--muted)]">Sim</span>
                    <select
                      className="w-full rounded border border-[var(--border)] bg-white px-3 py-2"
                      value={form.sim}
                      onChange={(e) =>
                        setForm({
                          ...form,
                          sim: e.target.value,
                          transfer_to: form.transfer_to.filter((t) => t !== e.target.value),
                        })
                      }
                      data-testid="sim-select"
                    >
                      {sims.map((s) => {
                        const caps = simMeta[s]?.capabilities ?? [];
                        const stub = caps.includes("stub");
                        const needsInstall = caps.includes("requires_install");
                        const label = stub
                          ? `${s} (stub — needs NVIDIA)`
                          : needsInstall
                            ? `${s} (RunPod — no Mac install)`
                            : s;
                        return (
                          <option key={s} value={s}>
                            {label}
                          </option>
                        );
                      })}
                    </select>
                  </label>
                  <div className="text-sm sm:col-span-2" data-testid="transfer-to-field">
                    <span className="mb-1 block text-[var(--muted)]">
                      Transfer to (Phase 6, after train)
                    </span>
                    <div className="flex flex-wrap gap-3">
                      {["mujoco", "pybullet"]
                        .filter((s) => s !== form.sim)
                        .map((s) => (
                          <label key={s} className="inline-flex items-center gap-2">
                            <input
                              type="checkbox"
                              checked={form.transfer_to.includes(s)}
                              onChange={(e) => {
                                const next = e.target.checked
                                  ? [...form.transfer_to, s]
                                  : form.transfer_to.filter((x) => x !== s);
                                setForm({ ...form, transfer_to: next });
                              }}
                              data-testid={`transfer-to-${s}`}
                            />
                            {s}
                          </label>
                        ))}
                    </div>
                    <p className="mt-1 text-xs text-[var(--muted)]">
                      Optional. Zero-shot eval in selected sims after checkpoint save.
                    </p>
                  </div>
                  <label className="text-sm sm:col-span-2">
                    <span className="mb-1 block text-[var(--muted)]">Timesteps</span>
                    <div className="flex flex-wrap gap-2">
                      <select
                        className="min-w-[10rem] flex-1 rounded border border-[var(--border)] bg-white px-3 py-2"
                        value={
                          timestepMode === "custom"
                            ? "custom"
                            : String(form.timesteps)
                        }
                        onChange={(e) => {
                          if (e.target.value === "custom") {
                            setTimestepMode("custom");
                            return;
                          }
                          setTimestepMode("preset");
                          setForm({
                            ...form,
                            timesteps: Number(e.target.value),
                          });
                        }}
                        data-testid="timesteps-select"
                      >
                        <option value={100_000}>100k</option>
                        <option value={50_000}>50k</option>
                        <option value={5_000}>5k (smoke)</option>
                        <option value={2_048}>2k (quick)</option>
                        <option value="custom">Custom…</option>
                      </select>
                      {timestepMode === "custom" && (
                        <input
                          type="number"
                          min={1}
                          max={10_000_000}
                          step={1}
                          className="w-40 rounded border border-[var(--border)] bg-white px-3 py-2"
                          value={form.timesteps}
                          onChange={(e) =>
                            setForm({
                              ...form,
                              timesteps: Number(e.target.value),
                            })
                          }
                          data-testid="timesteps-custom"
                          aria-label="Custom timesteps"
                        />
                      )}
                    </div>
                    <span className="mt-1 block text-xs text-[var(--muted)]">
                      Any integer 1–10,000,000. Prefer mujoco/pybullet for new tasks;
                      genesis needs optional install; isaac* are stubs.
                    </span>
                  </label>
                  {form.compute === "runpod" && (
                    <>
                      <label className="text-sm">
                        <span className="mb-1 block text-[var(--muted)]">GPU type</span>
                        <select
                          className="w-full rounded border border-[var(--border)] bg-white px-3 py-2"
                          value={form.gpu_type}
                          onChange={(e) =>
                            setForm({ ...form, gpu_type: e.target.value })
                          }
                          data-testid="gpu-type-select"
                        >
                          <option value="best">
                            Best available (4090→3090→3070→A4000…)
                          </option>
                          <option value="NVIDIA GeForce RTX 4090">
                            NVIDIA GeForce RTX 4090 (Secure ~$0.34/hr)
                          </option>
                          <option value="NVIDIA GeForce RTX 3090">
                            NVIDIA GeForce RTX 3090
                          </option>
                          <option value="NVIDIA GeForce RTX 3070">
                            NVIDIA GeForce RTX 3070 (cheap smoke)
                          </option>
                          <option value="NVIDIA RTX A4000">NVIDIA RTX A4000</option>
                        </select>
                      </label>
                      <label className="text-sm">
                        <span className="mb-1 block text-[var(--muted)]">
                          Budget USD (watchdog kill when exceeded; 0 = no per-run cap)
                        </span>
                        <input
                          type="number"
                          min={0}
                          step={0.01}
                          className="w-full rounded border border-[var(--border)] bg-white px-3 py-2"
                          value={form.budget_usd}
                          onChange={(e) =>
                            setForm({
                              ...form,
                              budget_usd: Number(e.target.value),
                            })
                          }
                          data-testid="budget-usd-input"
                        />
                      </label>
                    </>
                  )}
                </div>
                {form.arch === "kan" && form.compute === "local" && (
                  <p className="mt-3 text-xs text-[var(--muted)]">
                    KAN on Mac CPU is slow — use 5k smoke for a quick gate, or 50k overnight.
                  </p>
                )}
                {["kaf", "gpkan", "fan"].includes(form.arch) && (
                  <p className="mt-3 text-xs text-[var(--muted)]">
                    Paper arches (arXiv:2502.06018 family): prefer RunPod + 100k wall_follow for
                    proof; use 2k/5k smoke first.
                  </p>
                )}
                {form.arch === "avinash_wall" && (
                  <p className="mt-3 text-xs text-[var(--muted)]">
                    Tabular Q-learning (course P2_D3): PDF reward + 27×3 Q-table. Algo is fixed to
                    Q-learning. Prefer wall_follow + 2k/5k smoke first; full PDF schedule ~200
                    episodes × 1200 steps (~240k). Ask before launching long 100k+ trains.
                  </p>
                )}
                {form.algo === "q_learning" && form.arch !== "avinash_wall" && (
                  <p className="mt-3 text-xs text-[var(--muted)]">
                    Function-approx Q-learning: same PDF 3 actions / reward / ε schedule as
                    avinash_wall, with <code>{form.arch}</code> as the Q-network tower (not PPO).
                    Prefer 2k/5k smoke first.
                  </p>
                )}
                {form.arch.startsWith("custom:") && (
                  <p className="mt-3 text-xs text-[var(--muted)]">
                    Using saved architecture from the Architectures tab.
                  </p>
                )}
                {form.compute === "runpod" && (
                  <p className="mt-3 text-xs text-[var(--muted)]">
                    Requires RUNPOD_API_KEY, RUNPOD_NETWORK_VOLUME_ID, ROBOLAB_WORKER_IMAGE,
                    ROBOLAB_GIT_URL, BACKEND_PUBLIC_URL in .env. Uses a pushed git SHA
                    (dirty tree falls back to origin/main). See docs/PHASE_3_TEST.md.
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

            <div
              className="mb-3 flex flex-wrap items-end gap-3"
              data-testid="experiments-toolbar"
            >
              <label className="min-w-[14rem] flex-1 text-sm">
                <span className="mb-1 block text-[var(--muted)]">Search</span>
                <input
                  type="search"
                  value={searchInput}
                  onChange={(e) => setSearchInput(e.target.value)}
                  placeholder="Name, id, sim, arch…"
                  className="w-full rounded border border-[var(--border)] bg-white px-3 py-2"
                  data-testid="experiments-search"
                  autoComplete="off"
                />
              </label>
              <button
                type="button"
                data-testid="experiments-filters-toggle"
                onClick={() => setFiltersOpen((o) => !o)}
                className={`rounded border px-3 py-2 text-sm font-medium ${
                  hasActiveFilters
                    ? "border-[var(--accent)] bg-[#e8f4ef] text-[var(--accent)]"
                    : "border-[var(--border)] bg-[var(--surface)]"
                }`}
              >
                Filters{hasActiveFilters ? " · on" : ""}
              </button>
              <label className="text-sm">
                <span className="mb-1 block text-[var(--muted)]">Sort</span>
                <select
                  className="rounded border border-[var(--border)] bg-white px-3 py-2"
                  value={sortKey}
                  onChange={(e) => setSortKey(e.target.value as SortKey)}
                  data-testid="experiments-sort"
                >
                  {SORT_OPTIONS.map((o) => (
                    <option key={o.value} value={o.value}>
                      {o.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="text-sm">
                <span className="mb-1 block text-[var(--muted)]">Per page</span>
                <select
                  className="rounded border border-[var(--border)] bg-white px-3 py-2"
                  value={expPageSize}
                  onChange={(e) =>
                    setExpPageSize(Number(e.target.value) as PageSize)
                  }
                  data-testid="experiments-page-size"
                >
                  {PAGE_SIZES.map((n) => (
                    <option key={n} value={n}>
                      {n}
                    </option>
                  ))}
                </select>
              </label>
              {hasQueryOrFilters ? (
                <button
                  type="button"
                  data-testid="experiments-clear"
                  className="rounded border border-[var(--border)] px-3 py-2 text-sm text-[var(--muted)]"
                  onClick={() => {
                    setSearchInput("");
                    setDebouncedSearch("");
                    setFilters(EMPTY_FILTERS);
                  }}
                >
                  Clear
                </button>
              ) : null}
            </div>

            {filtersOpen && (
              <div
                className="mb-4 grid gap-3 rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4 sm:grid-cols-2 lg:grid-cols-3"
                data-testid="experiments-filters"
              >
                <label className="text-sm">
                  <span className="mb-1 block text-[var(--muted)]">Sim</span>
                  <select
                    className="w-full rounded border border-[var(--border)] bg-white px-3 py-2"
                    value={filters.sim}
                    onChange={(e) => setFilters({ ...filters, sim: e.target.value })}
                    data-testid="filter-sim"
                  >
                    <option value="">Any</option>
                    {filterOptions.sim.map((v) => (
                      <option key={v} value={v}>
                        {v}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="text-sm">
                  <span className="mb-1 block text-[var(--muted)]">Arch</span>
                  <select
                    className="w-full rounded border border-[var(--border)] bg-white px-3 py-2"
                    value={filters.arch}
                    onChange={(e) => setFilters({ ...filters, arch: e.target.value })}
                    data-testid="filter-arch"
                  >
                    <option value="">Any</option>
                    {filterOptions.arch.map((v) => (
                      <option key={v} value={v}>
                        {v}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="text-sm">
                  <span className="mb-1 block text-[var(--muted)]">Status</span>
                  <select
                    className="w-full rounded border border-[var(--border)] bg-white px-3 py-2"
                    value={filters.status}
                    onChange={(e) => setFilters({ ...filters, status: e.target.value })}
                    data-testid="filter-status"
                  >
                    <option value="">Any</option>
                    {filterOptions.status.map((v) => (
                      <option key={v} value={v}>
                        {v}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="text-sm">
                  <span className="mb-1 block text-[var(--muted)]">Compute</span>
                  <select
                    className="w-full rounded border border-[var(--border)] bg-white px-3 py-2"
                    value={filters.compute}
                    onChange={(e) => setFilters({ ...filters, compute: e.target.value })}
                    data-testid="filter-compute"
                  >
                    <option value="">Any</option>
                    {filterOptions.compute.map((v) => (
                      <option key={v} value={v}>
                        {v}
                      </option>
                    ))}
                  </select>
                </label>
                {filterOptions.group.length > 0 ? (
                  <label className="text-sm">
                    <span className="mb-1 block text-[var(--muted)]">Group</span>
                    <select
                      className="w-full rounded border border-[var(--border)] bg-white px-3 py-2"
                      value={filters.group}
                      onChange={(e) => setFilters({ ...filters, group: e.target.value })}
                      data-testid="filter-group"
                    >
                      <option value="">Any</option>
                      {filterOptions.group.map((v) => (
                        <option key={v} value={v}>
                          {v}
                        </option>
                      ))}
                    </select>
                  </label>
                ) : null}
                <label className="text-sm">
                  <span className="mb-1 block text-[var(--muted)]">From</span>
                  <input
                    type="date"
                    className="w-full rounded border border-[var(--border)] bg-white px-3 py-2"
                    value={filters.dateFrom}
                    onChange={(e) => setFilters({ ...filters, dateFrom: e.target.value })}
                    data-testid="filter-date-from"
                  />
                </label>
                <label className="text-sm">
                  <span className="mb-1 block text-[var(--muted)]">To</span>
                  <input
                    type="date"
                    className="w-full rounded border border-[var(--border)] bg-white px-3 py-2"
                    value={filters.dateTo}
                    onChange={(e) => setFilters({ ...filters, dateTo: e.target.value })}
                    data-testid="filter-date-to"
                  />
                </label>
              </div>
            )}

            <div className="overflow-x-auto rounded-lg border border-[var(--border)] bg-[var(--surface)]">
              <table className="w-full min-w-[960px] text-left text-sm">
                <thead className="border-b border-[var(--border)] bg-[#f0f4f7] text-[var(--muted)]">
                  <tr>
                    <th className="px-4 py-3 font-medium">Pick</th>
                    <th className="px-4 py-3 font-medium">Name</th>
                    <th className="px-4 py-3 font-medium">Sim</th>
                    <th className="px-4 py-3 font-medium">Arch</th>
                    <th className="px-4 py-3 font-medium">Status</th>
                    <th className="px-4 py-3 font-medium">Progress</th>
                    <th className="px-4 py-3 font-medium">Mean return</th>
                    <th className="px-4 py-3 font-medium">Cost</th>
                    <th className="px-4 py-3 font-medium">W&amp;B</th>
                    <th className="px-4 py-3 font-medium">Video</th>
                  </tr>
                </thead>
                <tbody>
                  {count === 0 ? (
                    <tr>
                      <td
                        colSpan={10}
                        className="px-4 py-16 text-center text-[var(--muted)]"
                        data-testid="empty-experiments"
                      >
                        0 experiments
                      </td>
                    </tr>
                  ) : count === null ? (
                    <tr>
                      <td
                        colSpan={10}
                        className={`px-4 py-16 text-center ${
                          error ? "text-red-800" : "text-[var(--muted)]"
                        }`}
                        data-testid={
                          error ? "experiments-load-error" : "experiments-loading"
                        }
                      >
                        {error
                          ? `Can't load experiments — ${error}`
                          : "Loading…"}
                      </td>
                    </tr>
                  ) : displayedRuns.length === 0 ? (
                    <tr>
                      <td
                        colSpan={10}
                        className="px-4 py-16 text-center text-[var(--muted)]"
                        data-testid="empty-search"
                      >
                        No experiments match your search or filters.
                      </td>
                    </tr>
                  ) : (
                    pagedRuns.map((r) => (
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
                        <td className="px-4 py-3 font-medium">
                          <button
                            type="button"
                            className="text-left hover:underline"
                            onClick={() => openRunDetail(r.id)}
                            data-testid={`open-run-${r.id}`}
                          >
                            {r.name}
                          </button>
                          {r.compute === "runpod" ? (
                            <span className="ml-2 text-xs font-normal text-[var(--muted)]">
                              runpod
                            </span>
                          ) : null}
                        </td>
                        <td className="px-4 py-3">{r.sim}</td>
                        <td className="relative px-4 py-3">
                          <ArchHoverLabel
                            arch={r.arch}
                            savedArchs={savedArchs}
                            testId={`arch-label-${r.id}`}
                            onOpenArchitectures={(detailKey) => {
                              setArchDetailKey(detailKey ?? null);
                              setView("architectures");
                            }}
                          />
                        </td>
                        <td className="px-4 py-3">
                          <span
                            className={`inline-block rounded px-2 py-0.5 text-xs font-medium ${statusClass(r.status)}`}
                            data-testid="status-chip"
                          >
                            {r.status}
                          </span>
                          {r.pod_id ? (
                            <p
                              className="mt-1 max-w-xs font-mono text-xs text-[var(--muted)]"
                              data-testid="pod-id"
                              title={r.pod_id}
                            >
                              pod {r.pod_id.slice(0, 12)}
                              {r.pod_id.length > 12 ? "…" : ""}
                            </p>
                          ) : null}
                          {r.error ? (
                            <p
                              className="mt-1 max-w-md whitespace-pre-wrap text-xs text-red-700"
                              title={r.error}
                              data-testid="run-error"
                            >
                              {r.error.slice(0, 280)}
                              {r.error.length > 280 ? "…" : ""}
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
                        <td className="px-4 py-3 text-xs tabular-nums" data-testid="run-cost">
                          {r.compute === "runpod" ? (
                            <>
                              <div>
                                {r.hourly_rate != null
                                  ? `${formatUsd(r.hourly_rate)}/hr`
                                  : "—/hr"}
                              </div>
                              <div className="text-[var(--muted)]">
                                accrued{" "}
                                {r.cost_usd != null ? formatUsd(r.cost_usd) : "$0.00"}
                              </div>
                            </>
                          ) : (
                            <span className="text-[var(--muted)]">local</span>
                          )}
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
                        <td className="px-4 py-3">
                          {r.status === "COMPLETE" ? (
                            <div className="flex flex-col gap-1">
                              <button
                                type="button"
                                data-testid={`render-video-${r.id}`}
                                disabled={
                                  r.video_status === "RENDERING" ||
                                  renderingId === r.id
                                }
                                onClick={() => void renderVideo(r.id)}
                                className="rounded border border-[var(--border)] bg-white px-2 py-1 text-xs font-medium hover:bg-slate-50 disabled:opacity-40"
                              >
                                {r.video_status === "RENDERING" || renderingId === r.id
                                  ? "Rendering…"
                                  : r.video_status === "READY"
                                    ? "Re-render video"
                                    : "Render video"}
                              </button>
                              {r.video_status === "READY" ? (
                                <button
                                  type="button"
                                  className="relative z-10 text-left text-xs text-[var(--accent)] underline"
                                  onClick={() => openRunDetail(r.id)}
                                  data-testid={`watch-video-${r.id}`}
                                >
                                  Watch
                                </button>
                              ) : null}
                              {r.video_status === "FAILED" && r.video_error ? (
                                <span
                                  className="max-w-[10rem] truncate text-xs text-red-700"
                                  title={r.video_error}
                                >
                                  failed
                                </span>
                              ) : null}
                            </div>
                          ) : r.status === "RUNNING" ||
                            r.status === "PROVISIONING" ||
                            r.status === "QUEUED" ? (
                            <button
                              type="button"
                              data-testid={`abort-run-${r.id}`}
                              disabled={abortingId === r.id}
                              onClick={() => void abortRun(r)}
                              className="rounded border border-amber-300 bg-amber-50 px-2 py-1 text-xs font-medium text-amber-900 hover:bg-amber-100 disabled:opacity-40"
                            >
                              {abortingId === r.id ? "Aborting…" : "Abort"}
                            </button>
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

            {displayedRuns.length > 0 ? (
              <div
                className="mt-3 flex flex-wrap items-center justify-between gap-3 text-sm text-[var(--muted)]"
                data-testid="experiments-pager"
              >
                <span>
                  Showing {expFrom}–{expTo} of {displayedRuns.length}
                </span>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    className="rounded border border-[var(--border)] px-2.5 py-1 disabled:opacity-40"
                    disabled={expPageClamped <= 0}
                    onClick={() => setExpPage((p) => Math.max(0, p - 1))}
                    data-testid="experiments-page-prev"
                  >
                    Prev
                  </button>
                  <span>
                    Page {expPageClamped + 1} / {expTotalPages}
                  </span>
                  <button
                    type="button"
                    className="rounded border border-[var(--border)] px-2.5 py-1 disabled:opacity-40"
                    disabled={expPageClamped >= expTotalPages - 1}
                    onClick={() =>
                      setExpPage((p) => Math.min(expTotalPages - 1, p + 1))
                    }
                    data-testid="experiments-page-next"
                  >
                    Next
                  </button>
                </div>
              </div>
            ) : null}

            {detailRun && (
              <div
                ref={detailRef}
                className="mt-6 rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5"
                data-testid="run-detail"
              >
                <div className="mb-3 flex items-baseline justify-between gap-4">
                  <div>
                    <h3 className="text-base font-medium">{detailRun.name}</h3>
                    <p className="text-xs text-[var(--muted)]">
                      {detailRun.id} · {detailRun.status}
                      {detailRun.video_status
                        ? ` · video ${detailRun.video_status}`
                        : ""}
                    </p>
                    <p
                      className="mt-1 text-sm text-[var(--muted)]"
                      data-testid="run-spaces"
                    >
                      Sim: <code className="text-xs">{detailRun.sim}</code>
                      {" · "}
                      obs_dim:{" "}
                      <code className="text-xs" data-testid="obs-dim">
                        {detailRun.obs_dim ?? "—"}
                      </code>
                      {" · "}
                      act_dim:{" "}
                      <code className="text-xs" data-testid="act-dim">
                        {detailRun.act_dim ?? "—"}
                      </code>
                      {detailRun.control_hz != null && (
                        <>
                          {" · "}
                          control_hz:{" "}
                          <code className="text-xs">{detailRun.control_hz}</code>
                        </>
                      )}
                      {detailRun.physics_substeps != null && (
                        <>
                          {" · "}
                          physics_substeps:{" "}
                          <code className="text-xs">
                            {detailRun.physics_substeps}
                          </code>
                        </>
                      )}
                    </p>
                  </div>
                  <button
                    type="button"
                    className="text-sm text-[var(--muted)] underline"
                    onClick={() => setDetailId(null)}
                  >
                    Close
                  </button>
                </div>
                {detailRun.video_status === "RENDERING" && (
                  <p className="mb-3 text-sm text-[var(--muted)]" data-testid="video-rendering">
                    Rendering playback… usually under a couple of minutes.
                  </p>
                )}
                {detailRun.video_status === "FAILED" && detailRun.video_error && (
                  <p className="mb-3 text-sm text-red-700" data-testid="video-error">
                    {detailRun.video_error}
                  </p>
                )}
                {detailRun.video_status === "READY" && (
                  <video
                    key={detailRun.video_url ?? detailRun.id}
                    controls
                    playsInline
                    preload="metadata"
                    className="max-h-[360px] w-full rounded bg-black"
                    src={
                      detailRun.video_url?.startsWith("/")
                        ? detailRun.video_url
                        : `/api/runs/${detailRun.id}/video`
                    }
                    data-testid="video-player"
                  >
                    Your browser does not support video.
                  </video>
                )}
                {(detailRun.checkpoints?.length || detailRun.checkpoint_artifact) && (
                  <div className="mt-4" data-testid="checkpoints-list">
                    <h4 className="mb-1 text-sm font-medium">Checkpoints</h4>
                    <ul className="space-y-1 text-sm text-[var(--muted)]">
                      {(detailRun.checkpoints ?? []).map((c) => (
                        <li key={`${c.kind}-${c.name}`}>
                          {c.kind === "wandb_artifact" ? (
                            <>
                              W&amp;B artifact: <code className="text-xs">{c.name}</code>
                            </>
                          ) : (
                            <>
                              Local: <code className="text-xs">{c.path ?? c.name}</code>
                            </>
                          )}
                        </li>
                      ))}
                      {!detailRun.checkpoints?.length && detailRun.checkpoint_artifact ? (
                        <li>
                          W&amp;B artifact:{" "}
                          <code className="text-xs">{detailRun.checkpoint_artifact}</code>
                        </li>
                      ) : null}
                    </ul>
                  </div>
                )}
                {detailRun.status === "COMPLETE" &&
                  detailRun.video_status !== "READY" &&
                  detailRun.video_status !== "RENDERING" && (
                    <button
                      type="button"
                      data-testid="render-video-detail"
                      disabled={renderingId === detailRun.id}
                      onClick={() => void renderVideo(detailRun.id)}
                      className="mt-4 rounded bg-[var(--accent)] px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
                    >
                      Render video
                    </button>
                  )}
                <button
                  type="button"
                  data-testid="flag-experiment-failure"
                  disabled={flaggingId === detailRun.id}
                  onClick={() => void flagExperimentFailure(detailRun)}
                  className="mt-4 ml-2 rounded border border-[var(--border)] bg-white px-4 py-2 text-sm font-medium hover:bg-slate-50 disabled:opacity-50"
                >
                  {flaggingId === detailRun.id
                    ? "Flagging…"
                    : "Flag experiment failure"}
                </button>
              </div>
            )}

            {selectable.length === 0 && count !== null && count > 0 && (
              <p className="mt-3 text-sm text-[var(--muted)]">
                No runs with W&amp;B URLs yet — finish a training run first.
              </p>
            )}
          </>
        )}

        {view === "architectures" && (
          <div data-testid="architectures-view">
            <ArchitecturesList
              archs={archs}
              archDefaults={archDefaults}
              savedArchs={savedArchs}
              initialDetailKey={archDetailKey}
              onInitialDetailConsumed={() => setArchDetailKey(null)}
              onUse={(archValue) => {
                setForm((f) => ({ ...f, arch: archValue }));
                setShowNewRun(true);
                setArchDetailKey(null);
                setView("experiments");
              }}
              onDeleteSaved={(id) => void deleteArchitecture(id)}
            />

            <div className="mb-4 mt-10">
              <h2 className="text-lg font-medium text-[var(--text)]">Architecture Builder</h2>
              <p className="text-sm text-[var(--muted)]">
                Dial hyperparameters for KAF / GPKAN / FAN (arXiv:2502.06018 family), save a named
                arch, then pick it in New Run. Builtins mlp/kan unchanged.
              </p>
            </div>

            <div
              className="mb-6 rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5"
              data-testid="arch-builder-form"
            >
              <div className="grid gap-4 sm:grid-cols-2">
                <label className="text-sm">
                  <span className="mb-1 block text-[var(--muted)]">Base architecture</span>
                  <select
                    className="w-full rounded border border-[var(--border)] bg-white px-3 py-2"
                    value={builderBase}
                    onChange={(e) => applyBuilderBase(e.target.value)}
                    data-testid="builder-base"
                  >
                    {archs.map((a) => {
                      const m = builtinMeta(a);
                      return (
                        <option key={a} value={a} title={m?.short ?? a}>
                          {m ? `${m.label} (${a})` : a}
                        </option>
                      );
                    })}
                  </select>
                </label>
                <label className="text-sm">
                  <span className="mb-1 block text-[var(--muted)]">Name (slug)</span>
                  <input
                    className="w-full rounded border border-[var(--border)] bg-white px-3 py-2"
                    value={builderName}
                    onChange={(e) => setBuilderName(e.target.value)}
                    data-testid="builder-name"
                    placeholder="kaf_wall_follow"
                  />
                </label>
                <label className="text-sm sm:col-span-2">
                  <span className="mb-1 block text-[var(--muted)]">
                    Hidden widths (comma-separated)
                  </span>
                  <input
                    className="w-full rounded border border-[var(--border)] bg-white px-3 py-2"
                    value={builderHidden}
                    onChange={(e) => setBuilderHidden(e.target.value)}
                    data-testid="builder-hidden"
                  />
                </label>
                {builderBase === "kaf" && (
                  <>
                    <label className="text-sm">
                      <span className="mb-1 block text-[var(--muted)]">num_grids (RFF)</span>
                      <input
                        type="number"
                        min={2}
                        max={64}
                        className="w-full rounded border border-[var(--border)] bg-white px-3 py-2"
                        value={builderNumGrids}
                        onChange={(e) => setBuilderNumGrids(Number(e.target.value))}
                        data-testid="builder-num-grids"
                      />
                    </label>
                    <label className="text-sm">
                      <span className="mb-1 block text-[var(--muted)]">
                        activation_expectation (σ)
                      </span>
                      <input
                        type="number"
                        step={0.01}
                        min={0.1}
                        max={4}
                        className="w-full rounded border border-[var(--border)] bg-white px-3 py-2"
                        value={builderActExpect}
                        onChange={(e) => setBuilderActExpect(Number(e.target.value))}
                      />
                    </label>
                    <label className="flex items-center gap-2 text-sm sm:col-span-2">
                      <input
                        type="checkbox"
                        checked={builderLayernorm}
                        onChange={(e) => setBuilderLayernorm(e.target.checked)}
                      />
                      use_layernorm
                    </label>
                  </>
                )}
                {builderBase === "gpkan" && (
                  <>
                    <label className="text-sm">
                      <span className="mb-1 block text-[var(--muted)]">num_basis (RBF)</span>
                      <input
                        type="number"
                        min={2}
                        max={64}
                        className="w-full rounded border border-[var(--border)] bg-white px-3 py-2"
                        value={builderNumBasis}
                        onChange={(e) => setBuilderNumBasis(Number(e.target.value))}
                        data-testid="builder-num-basis"
                      />
                    </label>
                    <label className="text-sm">
                      <span className="mb-1 block text-[var(--muted)]">init_bandwidth</span>
                      <input
                        type="number"
                        step={0.1}
                        min={0.05}
                        max={10}
                        className="w-full rounded border border-[var(--border)] bg-white px-3 py-2"
                        value={builderBandwidth}
                        onChange={(e) => setBuilderBandwidth(Number(e.target.value))}
                      />
                    </label>
                  </>
                )}
                {builderBase === "fan" && (
                  <>
                    <label className="text-sm">
                      <span className="mb-1 block text-[var(--muted)]">p_ratio (0–0.5)</span>
                      <input
                        type="number"
                        step={0.05}
                        min={0.05}
                        max={0.45}
                        className="w-full rounded border border-[var(--border)] bg-white px-3 py-2"
                        value={builderPRatio}
                        onChange={(e) => setBuilderPRatio(Number(e.target.value))}
                        data-testid="builder-p-ratio"
                      />
                    </label>
                    <label className="text-sm">
                      <span className="mb-1 block text-[var(--muted)]">activation</span>
                      <select
                        className="w-full rounded border border-[var(--border)] bg-white px-3 py-2"
                        value={builderActivation}
                        onChange={(e) => setBuilderActivation(e.target.value)}
                      >
                        <option value="gelu">gelu</option>
                        <option value="relu">relu</option>
                        <option value="silu">silu</option>
                        <option value="tanh">tanh</option>
                      </select>
                    </label>
                  </>
                )}
                {builderBase === "kan" && (
                  <>
                    <label className="text-sm">
                      <span className="mb-1 block text-[var(--muted)]">grid_size</span>
                      <input
                        type="number"
                        min={1}
                        max={32}
                        className="w-full rounded border border-[var(--border)] bg-white px-3 py-2"
                        value={builderGridSize}
                        onChange={(e) => setBuilderGridSize(Number(e.target.value))}
                      />
                    </label>
                    <label className="text-sm">
                      <span className="mb-1 block text-[var(--muted)]">spline_order</span>
                      <input
                        type="number"
                        min={1}
                        max={5}
                        className="w-full rounded border border-[var(--border)] bg-white px-3 py-2"
                        value={builderSplineOrder}
                        onChange={(e) => setBuilderSplineOrder(Number(e.target.value))}
                      />
                    </label>
                  </>
                )}
                <label className="text-sm">
                  <span className="mb-1 block text-[var(--muted)]">lr_default (hint)</span>
                  <input
                    type="number"
                    step={0.0001}
                    min={1e-6}
                    max={0.1}
                    className="w-full rounded border border-[var(--border)] bg-white px-3 py-2"
                    value={builderLr}
                    onChange={(e) => setBuilderLr(Number(e.target.value))}
                  />
                </label>
                <label className="text-sm sm:col-span-2">
                  <span className="mb-1 block text-[var(--muted)]">Notes</span>
                  <input
                    className="w-full rounded border border-[var(--border)] bg-white px-3 py-2"
                    value={builderNotes}
                    onChange={(e) => setBuilderNotes(e.target.value)}
                    placeholder="optional"
                  />
                </label>
              </div>
              <div className="mt-4 flex gap-3">
                <button
                  type="button"
                  disabled={builderSaving}
                  onClick={() => void saveArchitecture()}
                  className="rounded bg-[var(--accent)] px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
                  data-testid="builder-save"
                >
                  {builderSaving ? "Saving…" : "Save architecture"}
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setShowNewRun(true);
                    setView("experiments");
                  }}
                  className="rounded border border-[var(--border)] px-4 py-2 text-sm"
                >
                  Use in New Run
                </button>
              </div>
            </div>
          </div>
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

        {view === "transfer" && (
          <div data-testid="transfer-view">
            <div className="mb-4 flex flex-wrap items-baseline justify-between gap-4">
              <div>
                <h2 className="text-lg font-medium text-[var(--text)]">Transfer</h2>
                <p className="text-sm text-[var(--muted)]">
                  Sim-to-sim zero-shot report: source vs target return, ratio, gap, and
                  perturbation curves
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

            <div className="mb-6 flex flex-wrap items-end gap-3">
              <label className="text-sm">
                <span className="mb-1 block text-[var(--muted)]">COMPLETE run</span>
                <select
                  className="min-w-[16rem] rounded border border-[var(--border)] bg-white px-3 py-2"
                  value={transferRunId}
                  data-testid="transfer-run-select"
                  onChange={(e) => {
                    const id = e.target.value;
                    setTransferRunId(id);
                    void loadTransferReport(id);
                  }}
                >
                  <option value="">Select…</option>
                  {runs
                    .filter((r) => r.status === "COMPLETE")
                    .map((r) => (
                      <option key={r.id} value={r.id}>
                        {r.name} ({r.sim}/{r.arch}) · {r.id.slice(0, 8)}
                        {r.transfer_status ? ` · ${r.transfer_status}` : ""}
                      </option>
                    ))}
                </select>
              </label>
              <button
                type="button"
                disabled={!transferRunId || transferRunning}
                onClick={() => void startTransferEval()}
                className="rounded bg-[var(--accent)] px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
                data-testid="transfer-run-btn"
              >
                {transferRunning ? "Running transfer…" : "Run transfer"}
              </button>
              <button
                type="button"
                disabled={!transferRunId || transferLoading}
                onClick={() => void loadTransferReport(transferRunId)}
                className="rounded border border-[var(--border)] px-3 py-2 text-sm"
              >
                Refresh
              </button>
            </div>

            {transferLoading && (
              <p className="text-sm text-[var(--muted)]">Loading report…</p>
            )}

            {!transferLoading && transferRunId && !transferReport && (
              <p className="text-sm text-[var(--muted)]">
                No report yet. Click <strong>Run transfer</strong> (needs local{" "}
                <code className="text-xs">policy.zip</code>).
              </p>
            )}

            {transferReport && transferReport.status === "FAILED" && (
              <p className="mb-4 text-sm text-red-700">
                Transfer failed: {transferReport.error || "unknown error"}
              </p>
            )}

            {transferReport && transferReport.status !== "FAILED" && (
              <>
                <div
                  className="mb-6 overflow-x-auto rounded-lg border border-[var(--border)] bg-[var(--surface)]"
                  data-testid="transfer-table"
                >
                  <table className="min-w-full text-left text-sm">
                    <thead className="border-b border-[var(--border)] text-[var(--muted)]">
                      <tr>
                        <th className="px-4 py-2">Sim</th>
                        <th className="px-4 py-2">Role</th>
                        <th className="px-4 py-2">Mean return</th>
                        <th className="px-4 py-2">CI</th>
                        <th className="px-4 py-2">Ratio</th>
                        <th className="px-4 py-2">Gap</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="border-b border-[var(--border)]">
                        <td className="px-4 py-2 font-medium">
                          {transferReport.source?.sim || transferReport.source_sim}
                        </td>
                        <td className="px-4 py-2">source</td>
                        <td className="px-4 py-2">
                          {transferReport.source?.mean?.toFixed(2) ?? "—"}
                        </td>
                        <td className="px-4 py-2 text-[var(--muted)]">
                          {transferReport.source?.ci_low != null &&
                          transferReport.source?.ci_high != null
                            ? `[${transferReport.source.ci_low.toFixed(1)}, ${transferReport.source.ci_high.toFixed(1)}]`
                            : "—"}
                        </td>
                        <td className="px-4 py-2">1.00</td>
                        <td className="px-4 py-2">0</td>
                      </tr>
                      {(transferReport.targets || []).map((t) => (
                        <tr key={t.sim} className="border-b border-[var(--border)]">
                          <td className="px-4 py-2 font-medium">{t.sim}</td>
                          <td className="px-4 py-2">
                            {t.status === "ok" ? "target" : t.status || "—"}
                          </td>
                          <td className="px-4 py-2">
                            {t.mean != null ? t.mean.toFixed(2) : t.reason || "—"}
                          </td>
                          <td className="px-4 py-2 text-[var(--muted)]">
                            {t.ci_low != null && t.ci_high != null
                              ? `[${t.ci_low.toFixed(1)}, ${t.ci_high.toFixed(1)}]`
                              : "—"}
                          </td>
                          <td className="px-4 py-2">
                            {t.transfer_ratio != null ? t.transfer_ratio.toFixed(3) : "—"}
                          </td>
                          <td className="px-4 py-2">
                            {t.gap != null ? t.gap.toFixed(2) : "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {(
                  [
                    ["friction", "Friction"],
                    ["mass_scale", "Mass scale"],
                    ["sensor_noise_std", "Sensor noise"],
                  ] as const
                ).map(([key, label]) => {
                  const rows = transferReport.robustness?.[key] || [];
                  const simsInChart = [...new Set(rows.map((r) => r.sim))];
                  const values = [...new Set(rows.map((r) => r.value))].sort(
                    (a, b) => a - b,
                  );
                  const chartRows = values.map((v) => {
                    const point: Record<string, number | null> = { value: v };
                    for (const s of simsInChart) {
                      const hit = rows.find((r) => r.sim === s && r.value === v);
                      point[s] = hit?.mean_return ?? null;
                    }
                    return point;
                  });
                  return (
                    <div
                      key={key}
                      className="mb-6 h-64 w-full rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4"
                      data-testid={`transfer-robust-${key}`}
                    >
                      <p className="mb-2 text-sm font-medium">{label} robustness</p>
                      {chartRows.length === 0 ? (
                        <p className="text-sm text-[var(--muted)]">No sweep data</p>
                      ) : (
                        <ResponsiveContainer width="100%" height="85%">
                          <LineChart data={chartRows}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                            <XAxis dataKey="value" tick={{ fontSize: 12 }} />
                            <YAxis tick={{ fontSize: 12 }} />
                            <Tooltip />
                            <Legend />
                            {simsInChart.map((s, i) => (
                              <Line
                                key={s}
                                type="monotone"
                                dataKey={s}
                                stroke={SERIES_COLORS[i % SERIES_COLORS.length]}
                                dot
                                connectNulls
                                strokeWidth={2}
                              />
                            ))}
                          </LineChart>
                        </ResponsiveContainer>
                      )}
                    </div>
                  );
                })}
              </>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
