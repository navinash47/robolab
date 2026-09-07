/** Display names, blurbs, and docs for Architectures UI. */

export type BuiltinArchKey = "mlp" | "kan" | "kaf" | "gpkan" | "fan" | "avinash_wall";

export type ArchMeta = {
  key: string;
  label: string;
  short: string;
  long: string;
  docsLabel: string;
  docsUrl: string;
  paper?: string;
};

const PHASE_ARCH_DOCS =
  "https://github.com/navinash47/robolab/blob/main/docs/PHASE_ARCH_BUILDER_APIS.md";
const AVINASH_WALL_DOCS =
  "https://github.com/navinash47/robolab/blob/main/docs/AVINASH_WALL.md";

export const BUILTIN_ARCH_META: Record<BuiltinArchKey, ArchMeta> = {
  mlp: {
    key: "mlp",
    label: "MLP",
    short: "Classic multilayer perceptron — dense layers with a fixed activation.",
    long: "Standard multilayer perceptron used as the RoboLab baseline policy backbone. Stacks fully connected layers with a shared activation (default tanh). Simple, fast, and a reliable control for comparing Kolmogorov–Arnold and Fourier-style arches.",
    docsLabel: "PHASE_ARCH_BUILDER_APIS.md",
    docsUrl: PHASE_ARCH_DOCS,
    paper: "baseline",
  },
  kan: {
    key: "kan",
    label: "KAN (B-spline)",
    short: "Kolmogorov–Arnold network with learnable B-spline edge functions.",
    long: "Kolmogorov–Arnold Network (Liu et al. / efficient-kan). Replaces fixed activations with learnable univariate B-splines on edges. RoboLab exposes grid_size and spline_order; mlp-style hidden widths still control layer sizes.",
    docsLabel: "PHASE_ARCH_BUILDER_APIS.md",
    docsUrl: PHASE_ARCH_DOCS,
    paper: "Liu et al. / efficient-kan",
  },
  kaf: {
    key: "kaf",
    label: "KAF (Kolmogorov–Arnold Fourier)",
    short: "RFF + GELU hybrid from Kolmogorov–Arnold Fourier Networks (arXiv:2502.06018).",
    long: "Kolmogorov–Arnold Fourier Network (Zhang et al., arXiv:2502.06018) — the main paper contribution. Replaces KAN B-splines with trainable Random Fourier Features (RFF) and a hybrid GELU–Fourier activation with learnable base/spectral scales. Key knobs: num_grids (RFF count), activation_expectation (σ), use_layernorm.",
    docsLabel: "arXiv:2502.06018",
    docsUrl: "https://arxiv.org/abs/2502.06018",
    paper: "arXiv:2502.06018",
  },
  gpkan: {
    key: "gpkan",
    label: "GPKAN (Gaussian RBF–KAN)",
    short: "Gaussian RBF–KAN baseline — GELU base plus learnable RBF centers per edge.",
    long: "Gaussian-basis KAN-style baseline in the spirit of GP-KAN (Yang & Wang; cited in the KAF paper). RoboLab uses a deterministic RBF–KAN (not full GP uncertainty): GELU base activation plus learnable Gaussian RBF centers, bandwidths, and coefficients per edge. Key knobs: num_basis, init_bandwidth.",
    docsLabel: "PHASE_ARCH_BUILDER_APIS.md",
    docsUrl: PHASE_ARCH_DOCS,
    paper: "GP-KAN spirit arXiv:2407.18397",
  },
  fan: {
    key: "fan",
    label: "FAN (Fourier Analysis Network)",
    short: "Fourier Analysis Network–style layer: cos/sin path plus nonlinear σ path.",
    long: "Fourier Analysis Network (Dong et al., arXiv:2410.02675). Each layer concatenates cos/sin of a learned projection with a nonlinear σ path; p_ratio controls the Fourier fraction (paper default 0.25). Used as a Fourier-style baseline alongside KAF.",
    docsLabel: "arXiv:2410.02675",
    docsUrl: "https://arxiv.org/abs/2410.02675",
    paper: "Dong et al. arXiv:2410.02675",
  },
  avinash_wall: {
    key: "avinash_wall",
    label: "Avinash Wall Follow",
    short: "Tabular Q-learning wall follower from the Robotics course project (P2_D3).",
    long: "Course-project wall follower: 3 lidar sectors × 3 distance bins (27 states), 3 actions (turn left / forward / turn right) at 0.3 m/s, and the PDF piecewise reward (+20 optimal right distance, +15/−8 front-near turn logic, −5 far, −1 near). Trains with tabular Q-learning (α=0.1, γ=1.0, ε 1.0→0.1 over 200 episodes) — not SB3 PPO. For the same PDF task with a neural Q-network, pick Arch=kaf|kan|gpkan|fan|mlp and Algo=Q-learning. SARSA is available via arch_cfg.algorithm=sarsa on the tabular path.",
    docsLabel: "AVINASH_WALL.md",
    docsUrl: AVINASH_WALL_DOCS,
    paper: "Course P2_D3 — Q-learning",
  },
};

/** Shared list page sizes (Experiments + Architectures). */
export const PAGE_SIZES = [10, 20, 50] as const;
export type PageSize = (typeof PAGE_SIZES)[number];
/** @deprecated Prefer PAGE_SIZES — kept for ArchitecturesList imports. */
export const ARCH_PAGE_SIZES = PAGE_SIZES;
export type ArchPageSize = PageSize;

/** Fallback hyperparams when /api/archs defaults are not yet loaded. */
export const BUILTIN_DEFAULT_CFG: Record<string, Record<string, unknown>> = {
  mlp: { hidden_sizes: [64, 64], activation: "tanh" },
  kan: { hidden_sizes: [32, 32], grid_size: 5, spline_order: 3 },
  kaf: {
    hidden_sizes: [64, 64],
    num_grids: 8,
    activation_expectation: 1.64,
    use_layernorm: true,
    spline_dropout: 0.0,
    lr_default: 3.0e-4,
  },
  gpkan: {
    hidden_sizes: [32, 32],
    num_basis: 8,
    init_bandwidth: 1.0,
    base_activation: "gelu",
    lr_default: 3.0e-4,
  },
  fan: {
    hidden_sizes: [64, 64],
    p_ratio: 0.25,
    activation: "gelu",
    lr_default: 3.0e-4,
  },
  avinash_wall: {
    algorithm: "q_learning",
    alpha: 0.1,
    gamma: 1.0,
    epsilon_start: 1.0,
    epsilon_end: 0.1,
    epsilon_decay: 0.05,
    explore_episodes: 200,
    episode_max_steps: 1200,
    linear_vel: 0.3,
    angular_vel: 0.7,
    near_max: 0.7,
    medium_max: 0.9,
    lr_default: 0.1,
  },
};

export function builtinMeta(name: string): ArchMeta | null {
  if (name in BUILTIN_ARCH_META) {
    return BUILTIN_ARCH_META[name as BuiltinArchKey];
  }
  return null;
}

export function formatCfgSummary(cfg: Record<string, unknown>): string {
  const parts: string[] = [];
  const hs = cfg.hidden_sizes;
  if (Array.isArray(hs)) parts.push(`widths ${hs.join("×")}`);
  if (typeof cfg.algorithm === "string") parts.push(String(cfg.algorithm));
  if (typeof cfg.alpha === "number") parts.push(`α ${cfg.alpha}`);
  if (typeof cfg.num_grids === "number") parts.push(`grids ${cfg.num_grids}`);
  if (typeof cfg.num_basis === "number") parts.push(`basis ${cfg.num_basis}`);
  if (typeof cfg.p_ratio === "number") parts.push(`p_ratio ${cfg.p_ratio}`);
  if (typeof cfg.grid_size === "number") parts.push(`grid ${cfg.grid_size}`);
  if (typeof cfg.spline_order === "number") parts.push(`order ${cfg.spline_order}`);
  if (typeof cfg.activation === "string") parts.push(cfg.activation);
  if (typeof cfg.base_activation === "string") parts.push(cfg.base_activation);
  if (typeof cfg.use_layernorm === "boolean" && cfg.use_layernorm) parts.push("LN");
  if (typeof cfg.init_bandwidth === "number") parts.push(`bw ${cfg.init_bandwidth}`);
  if (typeof cfg.activation_expectation === "number")
    parts.push(`σ ${cfg.activation_expectation}`);
  if (typeof cfg.lr_default === "number") parts.push(`lr ${cfg.lr_default}`);
  return parts.length ? parts.join(" · ") : "default knobs";
}

export function docsHrefForBase(base: string): { label: string; url: string } {
  const m = builtinMeta(base);
  if (m) return { label: m.docsLabel, url: m.docsUrl };
  return { label: "PHASE_ARCH_BUILDER_APIS.md", url: PHASE_ARCH_DOCS };
}
