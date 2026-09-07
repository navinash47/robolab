import { useEffect, useMemo, useState } from "react";
import { ArchHoverCard, resolveArchDisplay } from "./ArchHover";
import {
  ARCH_PAGE_SIZES,
  BUILTIN_DEFAULT_CFG,
  builtinMeta,
  docsHrefForBase,
  formatCfgSummary,
  type ArchPageSize,
} from "./archMeta";

export type SavedArchRow = {
  id: string;
  name: string;
  base_arch: string;
  cfg: Record<string, unknown>;
  notes?: string;
};

type ListItem =
  | {
      kind: "builtin";
      key: string;
      name: string;
      base_arch: string;
      cfg: Record<string, unknown>;
    }
  | {
      kind: "saved";
      key: string;
      name: string;
      base_arch: string;
      cfg: Record<string, unknown>;
      id: string;
      notes: string;
    };

type Props = {
  archs: string[];
  archDefaults: Record<string, Record<string, unknown>>;
  savedArchs: SavedArchRow[];
  onUse: (archValue: string) => void;
  onDeleteSaved: (id: string) => void;
  /** Open a specific arch detail when navigating from Experiments hover. */
  initialDetailKey?: string | null;
  onInitialDetailConsumed?: () => void;
};

function hyperparamsEntries(cfg: Record<string, unknown>): [string, string][] {
  return Object.entries(cfg).map(([k, v]) => [
    k,
    Array.isArray(v) ? v.join(", ") : String(v),
  ]);
}

export function ArchitecturesList({
  archs,
  archDefaults,
  savedArchs,
  onUse,
  onDeleteSaved,
  initialDetailKey = null,
  onInitialDetailConsumed,
}: Props) {
  const [pageSize, setPageSize] = useState<ArchPageSize>(10);
  const [page, setPage] = useState(0);
  const [detailKey, setDetailKey] = useState<string | null>(initialDetailKey);

  useEffect(() => {
    if (initialDetailKey) {
      setDetailKey(initialDetailKey);
      onInitialDetailConsumed?.();
    }
  }, [initialDetailKey, onInitialDetailConsumed]);

  const items: ListItem[] = useMemo(() => {
    const builtins: ListItem[] = [...archs]
      .sort((a, b) => a.localeCompare(b))
      .map((name) => ({
        kind: "builtin" as const,
        key: `builtin:${name}`,
        name,
        base_arch: name,
        cfg: {
          ...(BUILTIN_DEFAULT_CFG[name] ?? {}),
          ...(archDefaults[name] ?? {}),
        },
      }));
    const saved: ListItem[] = savedArchs.map((s) => ({
      kind: "saved" as const,
      key: `saved:${s.id}`,
      name: s.name,
      base_arch: s.base_arch,
      cfg: s.cfg,
      id: s.id,
      notes: s.notes ?? "",
    }));
    return [...builtins, ...saved];
  }, [archs, archDefaults, savedArchs]);

  const totalPages = Math.max(1, Math.ceil(items.length / pageSize));

  useEffect(() => {
    setPage((p) => Math.min(p, totalPages - 1));
  }, [totalPages, pageSize]);

  useEffect(() => {
    setPage(0);
  }, [pageSize]);

  const pageItems = items.slice(page * pageSize, page * pageSize + pageSize);
  const detail = detailKey ? items.find((i) => i.key === detailKey) ?? null : null;

  if (detail) {
    const m = builtinMeta(detail.base_arch);
    const docs = docsHrefForBase(detail.base_arch);
    const title =
      detail.kind === "builtin"
        ? (m?.label ?? detail.name)
        : detail.name;
    const long =
      detail.kind === "builtin"
        ? (m?.long ?? m?.short ?? "")
        : [
            detail.notes.trim() || "Saved custom architecture from the Builder.",
            m ? `Based on ${m.label}: ${m.short}` : `Based on ${detail.base_arch}.`,
            formatCfgSummary(detail.cfg),
          ].join(" ");
    const rows = hyperparamsEntries(detail.cfg);

    return (
      <div
        className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5"
        data-testid="arch-detail-view"
      >
        <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="text-xs uppercase tracking-wide text-[var(--muted)]">
              {detail.kind === "builtin" ? "Builtin" : "Saved architecture"}
            </p>
            <h3 className="text-lg font-medium text-[var(--text)]">{title}</h3>
            <p className="mt-1 font-mono text-xs text-[var(--muted)]">
              {detail.kind === "builtin"
                ? detail.name
                : `${detail.name} · id ${detail.id}`}
            </p>
          </div>
          <button
            type="button"
            className="rounded border border-[var(--border)] px-3 py-1.5 text-sm"
            onClick={() => setDetailKey(null)}
            data-testid="arch-detail-back"
          >
            Back to list
          </button>
        </div>

        <p className="mb-4 max-w-3xl text-sm leading-relaxed text-[var(--text)]">
          {long}
        </p>

        <div className="mb-4 flex flex-wrap gap-3 text-sm">
          <a
            href={docs.url}
            target="_blank"
            rel="noopener noreferrer"
            className="font-medium text-[var(--accent)] underline-offset-2 hover:underline"
            data-testid="arch-detail-docs"
          >
            {docs.label}
          </a>
          {detail.base_arch === "kaf" && docs.url !== "https://arxiv.org/abs/2502.06018" ? (
            <a
              href="https://arxiv.org/abs/2502.06018"
              target="_blank"
              rel="noopener noreferrer"
              className="font-medium text-[var(--accent)] underline-offset-2 hover:underline"
            >
              arXiv:2502.06018
            </a>
          ) : null}
          <a
            href="https://github.com/navinash47/robolab/blob/main/docs/PHASE_ARCH_BUILDER_APIS.md"
            target="_blank"
            rel="noopener noreferrer"
            className="text-[var(--muted)] underline-offset-2 hover:underline"
          >
            PHASE_ARCH_BUILDER_APIS.md
          </a>
        </div>

        <h4 className="mb-2 text-sm font-medium text-[var(--text)]">Hyperparameters</h4>
        {rows.length === 0 ? (
          <p className="text-sm text-[var(--muted)]">No cfg keys.</p>
        ) : (
          <table className="mb-4 w-full max-w-xl text-left text-sm" data-testid="arch-detail-hparams">
            <thead>
              <tr className="border-b border-[var(--border)] text-[var(--muted)]">
                <th className="py-1.5 pr-3 font-medium">Key</th>
                <th className="py-1.5 font-medium">Value</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(([k, v]) => (
                <tr key={k} className="border-b border-[var(--border)]">
                  <td className="py-1.5 pr-3 font-mono text-xs">{k}</td>
                  <td className="py-1.5 font-mono text-xs">{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            className="rounded bg-[var(--accent)] px-3 py-1.5 text-sm font-medium text-white"
            onClick={() =>
              onUse(
                detail.kind === "builtin" ? detail.name : `custom:${detail.id}`,
              )
            }
          >
            Use in New Run
          </button>
          {detail.kind === "saved" ? (
            <button
              type="button"
              className="rounded border border-red-300 px-3 py-1.5 text-sm text-red-800"
              onClick={() => onDeleteSaved(detail.id)}
            >
              Delete
            </button>
          ) : null}
        </div>
      </div>
    );
  }

  const from = items.length === 0 ? 0 : page * pageSize + 1;
  const to = Math.min(items.length, (page + 1) * pageSize);

  return (
    <div data-testid="arch-list-panel">
      <div className="mb-3 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h3 className="text-base font-medium">Architectures</h3>
          <p className="text-sm text-[var(--muted)]">
            Builtins plus saved Builder configs. Hover a name for a quick overview;
            open Details for docs and hyperparams.
          </p>
        </div>
        <label className="flex items-center gap-2 text-sm text-[var(--muted)]">
          Per page
          <select
            className="rounded border border-[var(--border)] bg-white px-2 py-1 text-[var(--text)]"
            value={pageSize}
            onChange={(e) => setPageSize(Number(e.target.value) as ArchPageSize)}
            data-testid="arch-page-size"
          >
            {ARCH_PAGE_SIZES.map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </label>
      </div>

      {items.length === 0 ? (
        <p className="text-sm text-[var(--muted)]" data-testid="saved-archs-empty">
          No architectures loaded.
        </p>
      ) : (
        <>
          <table className="w-full text-left text-sm" data-testid="saved-archs-table">
            <thead>
              <tr className="border-b border-[var(--border)] text-[var(--muted)]">
                <th className="py-2 pr-3 font-medium">Name</th>
                <th className="py-2 pr-3 font-medium">Kind</th>
                <th className="py-2 pr-3 font-medium">Base / id</th>
                <th className="py-2 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {pageItems.map((item) => {
                const tip = resolveArchDisplay(
                  item.kind === "builtin" ? item.name : `custom:${item.id}`,
                  savedArchs,
                );
                const display =
                  item.kind === "builtin"
                    ? (builtinMeta(item.name)?.label ?? item.name)
                    : item.name;
                return (
                  <tr
                    key={item.key}
                    className="border-b border-[var(--border)]"
                    data-testid={
                      item.kind === "saved"
                        ? `saved-arch-${item.id}`
                        : `builtin-arch-${item.name}`
                    }
                  >
                    <td className="relative py-2 pr-3">
                      <span className="arch-hover-target font-medium">
                        {display}
                        <span className="ml-1 font-mono text-xs font-normal text-[var(--muted)]">
                          ({item.kind === "builtin" ? item.name : item.base_arch})
                        </span>
                        <ArchHoverCard
                          title={tip.title}
                          subtitle={tip.subtitle}
                          body={tip.body}
                        />
                      </span>
                    </td>
                    <td className="py-2 pr-3 text-[var(--muted)]">
                      {item.kind === "builtin" ? "builtin" : "saved"}
                    </td>
                    <td className="py-2 pr-3 font-mono text-xs text-[var(--muted)]">
                      {item.kind === "builtin" ? item.name : item.id}
                    </td>
                    <td className="py-2">
                      <button
                        type="button"
                        className="mr-2 text-[var(--accent)] underline"
                        onClick={() => setDetailKey(item.key)}
                        data-testid={`arch-detail-${item.key}`}
                      >
                        Details
                      </button>
                      <button
                        type="button"
                        className="mr-2 text-[var(--accent)] underline"
                        onClick={() =>
                          onUse(
                            item.kind === "builtin"
                              ? item.name
                              : `custom:${item.id}`,
                          )
                        }
                      >
                        Use
                      </button>
                      {item.kind === "saved" ? (
                        <button
                          type="button"
                          className="text-red-700 underline"
                          onClick={() => onDeleteSaved(item.id)}
                        >
                          Delete
                        </button>
                      ) : null}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          <div
            className="mt-3 flex flex-wrap items-center justify-between gap-3 text-sm text-[var(--muted)]"
            data-testid="arch-pager"
          >
            <span>
              Showing {from}–{to} of {items.length}
            </span>
            <div className="flex items-center gap-2">
              <button
                type="button"
                className="rounded border border-[var(--border)] px-2.5 py-1 disabled:opacity-40"
                disabled={page <= 0}
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                data-testid="arch-page-prev"
              >
                Prev
              </button>
              <span>
                Page {page + 1} / {totalPages}
              </span>
              <button
                type="button"
                className="rounded border border-[var(--border)] px-2.5 py-1 disabled:opacity-40"
                disabled={page >= totalPages - 1}
                onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                data-testid="arch-page-next"
              >
                Next
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
