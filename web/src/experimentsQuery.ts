/** Client-side search, filter, and sort for the experiments table. */

export type ExperimentRow = {
  id: string;
  name: string;
  sim: string;
  arch: string;
  task?: string;
  compute?: string;
  status: string;
  progress: number;
  mean_return: number | null;
  cost_usd?: number | null;
  created_at?: string | null;
  group?: string | null;
};

export type ExperimentFilters = {
  sim: string;
  arch: string;
  status: string;
  compute: string;
  group: string;
  dateFrom: string;
  dateTo: string;
};

export type SortKey =
  | "created_desc"
  | "created_asc"
  | "name_asc"
  | "name_desc"
  | "status_asc"
  | "progress_desc"
  | "cost_desc"
  | "return_desc"
  | "return_asc";

export const EMPTY_FILTERS: ExperimentFilters = {
  sim: "",
  arch: "",
  status: "",
  compute: "",
  group: "",
  dateFrom: "",
  dateTo: "",
};

export const SORT_OPTIONS: { value: SortKey; label: string }[] = [
  { value: "created_desc", label: "Created (newest)" },
  { value: "created_asc", label: "Created (oldest)" },
  { value: "name_asc", label: "Name (A–Z)" },
  { value: "name_desc", label: "Name (Z–A)" },
  { value: "status_asc", label: "Status" },
  { value: "progress_desc", label: "Progress" },
  { value: "cost_desc", label: "Cost" },
  { value: "return_desc", label: "Mean return (high)" },
  { value: "return_asc", label: "Mean return (low)" },
];

const FIELD_KEYS = ["sim", "arch", "status", "compute", "task", "group"] as const;

function norm(s: string | null | undefined): string {
  return (s ?? "").trim().toLowerCase();
}

/** Levenshtein distance (small strings). */
function editDistance(a: string, b: string): number {
  if (a === b) return 0;
  if (!a.length) return b.length;
  if (!b.length) return a.length;
  const rows = a.length + 1;
  const cols = b.length + 1;
  const prev = new Array<number>(cols);
  const cur = new Array<number>(cols);
  for (let j = 0; j < cols; j++) prev[j] = j;
  for (let i = 1; i < rows; i++) {
    cur[0] = i;
    for (let j = 1; j < cols; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
    }
    for (let j = 0; j < cols; j++) prev[j] = cur[j];
  }
  return prev[b.length];
}

/** Soft token score against a haystack string (higher = better). */
function tokenScore(token: string, hay: string): number {
  if (!token || !hay) return 0;
  if (hay === token) return 100;
  if (hay.startsWith(token)) return 80;
  if (hay.includes(token)) return 60;

  // Contiguous subsequence (e.g. "wlfw" in "wall_follow")
  let ti = 0;
  for (let i = 0; i < hay.length && ti < token.length; i++) {
    if (hay[i] === token[ti]) ti++;
  }
  if (ti === token.length) return 35;

  // Typo tolerance: compare token to each word / id chunk
  const parts = hay.split(/[^a-z0-9]+/).filter(Boolean);
  let best = 0;
  for (const p of parts) {
    if (!p) continue;
    const maxLen = Math.max(token.length, p.length);
    if (maxLen === 0) continue;
    const d = editDistance(token, p);
    const ratio = 1 - d / maxLen;
    if (ratio >= 0.6) best = Math.max(best, Math.round(ratio * 50));
  }
  // Whole-string edit distance for short queries
  if (token.length >= 2 && hay.length <= 48) {
    const d = editDistance(token, hay.slice(0, token.length + 2));
    const maxLen = Math.max(token.length, Math.min(hay.length, token.length + 2));
    const ratio = 1 - d / maxLen;
    if (ratio >= 0.55) best = Math.max(best, Math.round(ratio * 45));
  }
  return best;
}

function searchableText(run: ExperimentRow): string {
  return [
    run.name,
    run.id,
    run.sim,
    run.arch,
    run.status,
    run.compute,
    run.task,
    run.group,
  ]
    .map(norm)
    .filter(Boolean)
    .join(" ");
}

/** Case-insensitive exact on name/id, or clear equality on discrete fields. */
export function isExactMatch(run: ExperimentRow, query: string): boolean {
  const q = norm(query);
  if (!q) return false;
  if (norm(run.name) === q) return true;
  if (norm(run.id) === q) return true;
  for (const key of FIELD_KEYS) {
    const v = norm(run[key] as string | undefined);
    if (v && v === q) return true;
  }
  return false;
}

export function fuzzyRelevance(run: ExperimentRow, query: string): number {
  const q = norm(query);
  if (!q) return 0;
  const tokens = q.split(/\s+/).filter(Boolean);
  const hay = searchableText(run);
  const name = norm(run.name);
  const id = norm(run.id);

  let total = 0;
  for (const token of tokens) {
    const scores = [
      tokenScore(token, name) * 1.4,
      tokenScore(token, id) * 1.2,
      tokenScore(token, hay),
    ];
    const best = Math.max(...scores);
    if (best <= 0) return 0; // every token must hit something
    total += best;
  }
  return total / tokens.length;
}

export function searchExperiments<T extends ExperimentRow>(
  runs: T[],
  query: string,
): T[] {
  const q = norm(query);
  if (!q) return runs;

  const exact = runs.filter((r) => isExactMatch(r, q));
  if (exact.length > 0) return exact;

  return runs
    .map((r) => ({ r, score: fuzzyRelevance(r, q) }))
    .filter((x) => x.score > 0)
    .sort((a, b) => b.score - a.score || a.r.name.localeCompare(b.r.name))
    .map((x) => x.r);
}

function parseDay(isoOrDate: string): number | null {
  if (!isoOrDate) return null;
  // date input is YYYY-MM-DD; created_at is ISO
  const d = new Date(isoOrDate.length <= 10 ? `${isoOrDate}T00:00:00` : isoOrDate);
  const t = d.getTime();
  return Number.isFinite(t) ? t : null;
}

export function applyFilters<T extends ExperimentRow>(
  runs: T[],
  filters: ExperimentFilters,
): T[] {
  const fromTs = parseDay(filters.dateFrom);
  const toRaw = parseDay(filters.dateTo);
  // Inclusive end-of-day for dateTo
  const toTs =
    toRaw == null
      ? null
      : filters.dateTo.length <= 10
        ? toRaw + 24 * 60 * 60 * 1000 - 1
        : toRaw;

  return runs.filter((r) => {
    if (filters.sim && norm(r.sim) !== norm(filters.sim)) return false;
    if (filters.arch && norm(r.arch) !== norm(filters.arch)) return false;
    if (filters.status && norm(r.status) !== norm(filters.status)) return false;
    if (filters.compute && norm(r.compute) !== norm(filters.compute)) return false;
    if (filters.group && norm(r.group) !== norm(filters.group)) return false;
    if (fromTs != null || toTs != null) {
      const created = parseDay(r.created_at ?? "");
      if (created == null) return false;
      if (fromTs != null && created < fromTs) return false;
      if (toTs != null && created > toTs) return false;
    }
    return true;
  });
}

function cmpNullableNumber(
  a: number | null | undefined,
  b: number | null | undefined,
  desc: boolean,
): number {
  const av = a == null || Number.isNaN(a) ? null : a;
  const bv = b == null || Number.isNaN(b) ? null : b;
  if (av == null && bv == null) return 0;
  if (av == null) return 1;
  if (bv == null) return -1;
  return desc ? bv - av : av - bv;
}

export function sortExperiments<T extends ExperimentRow>(
  runs: T[],
  sort: SortKey,
): T[] {
  const out = [...runs];
  out.sort((a, b) => {
    switch (sort) {
      case "created_desc":
      case "created_asc": {
        const at = parseDay(a.created_at ?? "") ?? 0;
        const bt = parseDay(b.created_at ?? "") ?? 0;
        return sort === "created_desc" ? bt - at : at - bt;
      }
      case "name_asc":
        return a.name.localeCompare(b.name) || a.id.localeCompare(b.id);
      case "name_desc":
        return b.name.localeCompare(a.name) || a.id.localeCompare(b.id);
      case "status_asc":
        return a.status.localeCompare(b.status) || a.name.localeCompare(b.name);
      case "progress_desc":
        return (
          cmpNullableNumber(a.progress, b.progress, true) ||
          a.name.localeCompare(b.name)
        );
      case "cost_desc":
        return (
          cmpNullableNumber(a.cost_usd, b.cost_usd, true) ||
          a.name.localeCompare(b.name)
        );
      case "return_desc":
        return (
          cmpNullableNumber(a.mean_return, b.mean_return, true) ||
          a.name.localeCompare(b.name)
        );
      case "return_asc":
        return (
          cmpNullableNumber(a.mean_return, b.mean_return, false) ||
          a.name.localeCompare(b.name)
        );
      default:
        return 0;
    }
  });
  return out;
}

export function uniqueValues(
  runs: ExperimentRow[],
  key: keyof ExperimentRow,
): string[] {
  const set = new Set<string>();
  for (const r of runs) {
    const v = r[key];
    if (typeof v === "string" && v.trim()) set.add(v);
  }
  return [...set].sort((a, b) => a.localeCompare(b));
}

export function queryExperiments<T extends ExperimentRow>(
  runs: T[],
  query: string,
  filters: ExperimentFilters,
  sort: SortKey,
): T[] {
  const filtered = applyFilters(runs, filters);
  const searched = searchExperiments(filtered, query);
  // Fuzzy search already ranks by relevance; only re-sort when no active query
  // or when we have exact-only results / empty query.
  if (norm(query) && searched.length && !searched.every((r) => isExactMatch(r, query))) {
    // Keep fuzzy ranking; still allow explicit sort override when user picks non-default
    if (sort !== "created_desc") return sortExperiments(searched, sort);
    return searched;
  }
  return sortExperiments(searched, sort);
}

export function filtersActive(filters: ExperimentFilters): boolean {
  return Object.values(filters).some((v) => Boolean(v));
}
