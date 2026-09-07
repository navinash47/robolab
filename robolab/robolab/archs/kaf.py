"""KAF policy/value backbone — Kolmogorov-Arnold Fourier Network (arXiv:2502.06018).

Port of official FastKAFLayer / RandomFourierFeatures (kolmogorovArnoldFourierNetwork/KAF)
adapted for RoboLab Architecture ABC (pi/vf dual stacks).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from robolab.core.arch import Architecture, register_arch


class RandomFourierFeatures(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_grids: int,
        dropout: float = 0.0,
        activation_expectation: float = 1.64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_grids = num_grids
        self.dropout = nn.Dropout(dropout)
        var_w = 1.0 / (input_dim * activation_expectation)
        self.weight = nn.Parameter(torch.randn(input_dim, num_grids) * math.sqrt(var_w))
        self.bias = nn.Parameter(torch.empty(num_grids))
        nn.init.uniform_(self.bias, 0, 2 * math.pi)
        self.combination = nn.Linear(2 * num_grids, input_dim)
        nn.init.xavier_uniform_(self.combination.weight)
        if self.combination.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.combination.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.combination.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = torch.matmul(x, self.weight) + self.bias
        fourier_features = torch.cat([torch.cos(projection), torch.sin(projection)], dim=-1)
        fourier_features = self.dropout(fourier_features)
        return self.combination(fourier_features)


class FastKAFLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_grids: int = 8,
        use_layernorm: bool = True,
        spline_dropout: float = 0.0,
        activation_expectation: float = 1.64,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layernorm = (
            nn.LayerNorm(input_dim) if use_layernorm and input_dim > 1 else None
        )
        self.feature_transform = RandomFourierFeatures(
            input_dim=input_dim,
            num_grids=num_grids,
            dropout=spline_dropout,
            activation_expectation=activation_expectation,
        )
        self.base_scale = nn.Parameter(torch.tensor(1.0))
        self.spline_scale = nn.Parameter(torch.tensor(1e-2))
        self.final_linear = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.final_linear.weight)
        if self.final_linear.bias is not None:
            nn.init.zeros_(self.final_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.layernorm(x) if self.layernorm is not None else x
        b = F.gelu(x)
        s = self.feature_transform(x_norm)
        return self.final_linear(self.base_scale * b + self.spline_scale * s)


class KAFStack(nn.Module):
    def __init__(
        self,
        layers_hidden: list[int],
        num_grids: int = 8,
        spline_dropout: float = 0.0,
        use_layernorm: bool = True,
        activation_expectation: float = 1.64,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                FastKAFLayer(
                    in_dim,
                    out_dim,
                    num_grids=num_grids,
                    spline_dropout=spline_dropout,
                    use_layernorm=use_layernorm,
                    activation_expectation=activation_expectation,
                )
                for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


@register_arch("kaf")
class KAFPolicyNet(Architecture):
    """Kolmogorov–Arnold Fourier feature extractor for SB3 ActorCriticPolicy."""

    def __init__(self, obs_dim: int, act_dim: int, cfg: dict | None = None):
        super().__init__(obs_dim, act_dim, cfg)
        hidden = list(self.cfg.get("hidden_sizes", [64, 64]))
        if not hidden:
            hidden = [64, 64]
        num_grids = int(self.cfg.get("num_grids", 8))
        use_layernorm = bool(self.cfg.get("use_layernorm", True))
        spline_dropout = float(self.cfg.get("spline_dropout", 0.0))
        activation_expectation = float(self.cfg.get("activation_expectation", 1.64))
        latent = int(hidden[-1])
        self._latent_pi = int(self.cfg.get("latent_dim_pi", latent))
        self._latent_vf = int(self.cfg.get("latent_dim_vf", latent))

        def build(out_dim: int) -> KAFStack:
            layers_hidden = [obs_dim, *hidden]
            if layers_hidden[-1] != out_dim:
                layers_hidden = [*layers_hidden, out_dim]
            return KAFStack(
                layers_hidden=layers_hidden,
                num_grids=num_grids,
                spline_dropout=spline_dropout,
                use_layernorm=use_layernorm,
                activation_expectation=activation_expectation,
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
