"""KAN policy/value backbone — efficient-kan (Blealtan) plug-in."""

from __future__ import annotations

import torch
from efficient_kan import KAN

from robolab.core.arch import Architecture, register_arch


@register_arch("kan")
class KANPolicyNet(Architecture):
    """Kolmogorov–Arnold feature extractor for SB3 ActorCriticPolicy.

    Uses Blealtan/efficient-kan ``KAN`` stacks for pi and vf branches.
    Default hidden sizes are smaller than MLP because B-spline layers are
    heavier on CPU.
    """

    def __init__(self, obs_dim: int, act_dim: int, cfg: dict | None = None):
        super().__init__(obs_dim, act_dim, cfg)
        hidden = list(self.cfg.get("hidden_sizes", [32, 32]))
        if not hidden:
            hidden = [32, 32]
        grid_size = int(self.cfg.get("grid_size", 5))
        spline_order = int(self.cfg.get("spline_order", 3))
        latent = int(hidden[-1])
        self._latent_pi = int(self.cfg.get("latent_dim_pi", latent))
        self._latent_vf = int(self.cfg.get("latent_dim_vf", latent))

        def build(out_dim: int) -> KAN:
            layers_hidden = [obs_dim, *hidden]
            if layers_hidden[-1] != out_dim:
                layers_hidden = [*layers_hidden, out_dim]
            return KAN(
                layers_hidden=layers_hidden,
                grid_size=grid_size,
                spline_order=spline_order,
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
        # update_grid=False — adaptive grids mutate weights; leave off for RL steps
        return self.pi_net(obs), self.vf_net(obs)
