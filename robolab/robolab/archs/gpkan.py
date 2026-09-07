"""GPKAN policy/value backbone — Gaussian RBF KAN-style (KAF paper baseline).

arXiv:2502.06018 compares GPKAN (Yang & Wang 2024 / GP-KAN spirit, arXiv:2407.18397).
Full probabilistic GP neurons are impractical inside SB3 PPO; this module uses
deterministic Gaussian RBF edge functions + GELU base (paper: GELU-based init).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from robolab.core.arch import Architecture, register_arch


class GaussianKANLinear(nn.Module):
    """Edge-wise: φ(x) = w_b·GELU(x) + Σ_k c_k · exp(-((x-μ_k)/σ_k)²) then Linear."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_basis: int = 8,
        init_bandwidth: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_basis = num_basis
        # Per-input-feature RBF bank, then mix to out_features
        self.centers = nn.Parameter(torch.empty(in_features, num_basis))
        self.log_bandwidth = nn.Parameter(
            torch.full((in_features, num_basis), math.log(max(init_bandwidth, 1e-3)))
        )
        self.coeffs = nn.Parameter(torch.empty(in_features, num_basis, out_features))
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.base_bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.centers, -2.0, 2.0)
        nn.init.normal_(self.coeffs, std=0.1 / math.sqrt(self.num_basis))
        nn.init.xavier_uniform_(self.base_weight)
        nn.init.zeros_(self.base_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in)
        bandwidth = self.log_bandwidth.exp().clamp(min=1e-3)
        # (B, in, K)
        diff = x.unsqueeze(-1) - self.centers.unsqueeze(0)
        rbf = torch.exp(-((diff / bandwidth.unsqueeze(0)) ** 2))
        # Mix RBFs: (B, out) = einsum over in,K
        spectral = torch.einsum("bik,iko->bo", rbf, self.coeffs)
        base = F.linear(F.gelu(x), self.base_weight, self.base_bias)
        return base + spectral


class GPKANStack(nn.Module):
    def __init__(
        self,
        layers_hidden: list[int],
        num_basis: int = 8,
        init_bandwidth: float = 1.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GaussianKANLinear(
                    in_dim,
                    out_dim,
                    num_basis=num_basis,
                    init_bandwidth=init_bandwidth,
                )
                for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


@register_arch("gpkan")
class GPKANPolicyNet(Architecture):
    """Gaussian RBF KAN-style feature extractor for SB3 ActorCriticPolicy."""

    def __init__(self, obs_dim: int, act_dim: int, cfg: dict | None = None):
        super().__init__(obs_dim, act_dim, cfg)
        hidden = list(self.cfg.get("hidden_sizes", [32, 32]))
        if not hidden:
            hidden = [32, 32]
        num_basis = int(self.cfg.get("num_basis", 8))
        init_bandwidth = float(self.cfg.get("init_bandwidth", 1.0))
        latent = int(hidden[-1])
        self._latent_pi = int(self.cfg.get("latent_dim_pi", latent))
        self._latent_vf = int(self.cfg.get("latent_dim_vf", latent))

        def build(out_dim: int) -> GPKANStack:
            layers_hidden = [obs_dim, *hidden]
            if layers_hidden[-1] != out_dim:
                layers_hidden = [*layers_hidden, out_dim]
            return GPKANStack(
                layers_hidden=layers_hidden,
                num_basis=num_basis,
                init_bandwidth=init_bandwidth,
            )

        self.pi_net = build(self._latent_pi)
        self.vf_net = build(self._latent_vf)

    @property
    def latent_dim_pi(self) -> int:
        return self._latent_pi

    @property
    def latent_dim_vf(self) -> int:
        return self._latent_vf

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pi_net(obs), self.vf_net(obs)
