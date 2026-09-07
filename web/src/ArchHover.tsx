import type { ReactNode } from "react";
import {
  builtinMeta,
  formatCfgSummary,
  type ArchMeta,
} from "./archMeta";

export type SavedArchRef = {
  id: string;
  name: string;
  base_arch: string;
  cfg?: Record<string, unknown>;
  notes?: string;
};

export type ArchDisplayInfo = {
  label: string;
  slug: string;
  title: string;
  subtitle?: string;
  body: string;
  /** Architectures list detail key, when known. */
  detailKey?: string;
};

export function resolveArchDisplay(
  arch: string,
  saved?: SavedArchRef[],
): ArchDisplayInfo {
  if (arch.startsWith("custom:")) {
    const id = arch.slice("custom:".length);
    const hit = saved?.find((s) => s.id === id);
    if (hit) return displayForSaved(hit);
    return {
      label: id,
      slug: id,
      title: id,
      subtitle: "Saved architecture",
      body: "Custom architecture id — details not loaded in this session.",
      detailKey: `saved:${id}`,
    };
  }

  const byName = saved?.find((s) => s.name === arch);
  if (byName) return displayForSaved(byName);

  const m = builtinMeta(arch);
  return displayForBuiltin(arch, m);
}

function displayForBuiltin(name: string, m: ArchMeta | null): ArchDisplayInfo {
  return {
    label: m?.label ?? name,
    slug: name,
    title: m?.label ?? name,
    subtitle: m?.paper,
    body: m?.short ?? "Builtin architecture.",
    detailKey: `builtin:${name}`,
  };
}

function displayForSaved(hit: SavedArchRef): ArchDisplayInfo {
  const base = builtinMeta(hit.base_arch);
  const notes = (hit.notes ?? "").trim();
  const knobs = hit.cfg ? formatCfgSummary(hit.cfg) : "";
  const knobSuffix = knobs ? ` — ${knobs}` : "";
  return {
    label: hit.name,
    slug: hit.base_arch,
    title: hit.name,
    subtitle: `Saved · base ${base?.label ?? hit.base_arch}`,
    body: notes
      ? `${notes}${knobSuffix}`
      : `Custom ${base?.label ?? hit.base_arch} config${knobSuffix}`,
    detailKey: `saved:${hit.id}`,
  };
}

export function ArchHoverCard({
  title,
  subtitle,
  body,
  footer,
}: {
  title: string;
  subtitle?: string;
  body: string;
  footer?: ReactNode;
}) {
  return (
    <div
      className={
        footer ? "arch-hover-card arch-hover-card--interactive" : "arch-hover-card"
      }
      role="tooltip"
    >
      <div className="arch-hover-card__inner">
        <div className="arch-hover-card__title">{title}</div>
        {subtitle ? <div className="arch-hover-card__sub">{subtitle}</div> : null}
        <p className="arch-hover-card__body">{body}</p>
        {footer ? <div className="arch-hover-card__footer">{footer}</div> : null}
      </div>
    </div>
  );
}

type ArchHoverLabelProps = {
  arch: string;
  savedArchs?: SavedArchRef[];
  /** Optional link / action under the overview blurb. */
  onOpenArchitectures?: (detailKey?: string) => void;
  className?: string;
  testId?: string;
};

/** Friendly arch name + muted slug + green-tint overview on hover. */
export function ArchHoverLabel({
  arch,
  savedArchs,
  onOpenArchitectures,
  className,
  testId,
}: ArchHoverLabelProps) {
  const info = resolveArchDisplay(arch, savedArchs);
  const footer =
    onOpenArchitectures != null ? (
      <button
        type="button"
        className="arch-hover-card__link"
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          onOpenArchitectures(info.detailKey);
        }}
      >
        Open in Architectures →
      </button>
    ) : null;

  return (
    <span
      className={`arch-hover-target font-medium ${className ?? ""}`}
      data-testid={testId}
      tabIndex={0}
    >
      {info.label}
      <span className="ml-1 font-mono text-xs font-normal text-[var(--muted)]">
        ({info.slug})
      </span>
      <ArchHoverCard
        title={info.title}
        subtitle={info.subtitle}
        body={info.body}
        footer={footer}
      />
    </span>
  );
}
