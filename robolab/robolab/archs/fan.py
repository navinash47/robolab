"""FAN policy/value backbone — Fourier Analysis Network (Dong et al., arXiv:2410.02675).

Port of YihongDong/FAN FANLayer for RoboLab Architecture ABC.
Cited as a baseline in arXiv:2502.06018 (KAF paper).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from robolab.core.arch import Architecture, register_arch


class FANLayer(nn.Module):
    """φ(x) = [cos(W_p x) ‖ sin(W_p x) ‖ σ(B + W_g x)]."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        p_ratio: float = 0.25,
        activation: str = "gelu",
        use_p_bias: bool = True,
    ):
        super().__init__()
        if not 0.0 < p_ratio < 0.5:
            raise ValueError("p_ratio must be in (0, 0.5)")
        p_output_dim = int(output_dim * p_ratio)
        g_output_dim = output_dim - p_output_dim * 2
        if p_output_dim < 1 or g_output_dim < 1:
            raise ValueError(
                f"output_dim={output_dim} with p_ratio={p_ratio} yields invalid "
                f"p={p_output_dim} g={g_output_dim}; widen hidden sizes"
            )
        self.input_linear_p = nn.Linear(input_dim, p_output_dim, bias=use_p_bias)
        self.input_linear_g = nn.Linear(input_dim, g_output_dim)
        act = activation.lower()
        if act == "gelu":
            self.activation = F.gelu
        elif act == "relu":
            self.activation = F.relu
        elif act == "silu" or act == "swish":
            self.activation = F.silu
        elif act == "tanh":
            self.activation = torch.tanh
        else:
            self.activation = F.gelu

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        g = self.activation(self.input_linear_g(src))
        p = self.input_linear_p(src)
        return torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)


class FANStack(nn.Module):
    def __init__(
        self,
        layers_hidden: list[int],
        p_ratio: float = 0.25,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                FANLayer(in_dim, out_dim, p_ratio=p_ratio, activation=activation)
                for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


@register_arch("fan")
class FANPolicyNet(Architecture):
    """Fourier Analysis Network feature extractor for SB3 ActorCriticPolicy."""

    def __init__(self, obs_dim: int, act_dim: int, cfg: dict | None = None):
        super().__init__(obs_dim, act_dim, cfg)
        hidden = list(self.cfg.get("hidden_sizes", [64, 64]))
        if not hidden:
            hidden = [64, 64]
        # FAN needs output_dim large enough for p and g splits
        hidden = [max(int(h), 8) for h in hidden]
        p_ratio = float(self.cfg.get("p_ratio", 0.25))
        activation = str(self.cfg.get("activation", "gelu"))
        latent = int(hidden[-1])
        self._latent_pi = int(self.cfg.get("latent_dim_pi", latent))
        self._latent_vf = int(self.cfg.get("latent_dim_vf", latent))

        def build(out_dim: int) -> FANStack:
            out_dim = max(int(out_dim), 8)
            layers_hidden = [obs_dim, *hidden]
            if layers_hidden[-1] != out_dim:
                layers_hidden = [*layers_hidden, out_dim]
            return FANStack(
                layers_hidden=layers_hidden,
                p_ratio=p_ratio,
                activation=activation,
            )

        self.pi_net = build(self._latent_pi)
        self.vf_net = build(self._latent_vf)
        self._latent_pi = max(self._latent_pi, 8)
        self._latent_vf = max(self._latent_vf, 8)

    @property
    def latent_dim_pi(self) -> int:
        return self._latent_pi

    @property
    def latent_dim_vf(self) -> int:
        return self._latent_vf

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pi_net(obs), self.vf_net(obs)
