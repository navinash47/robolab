"""RunConfig / RunStatus / RunRecord."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RunStatus(str, Enum):
    QUEUED = "QUEUED"
    PROVISIONING = "PROVISIONING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    KILLED_BY_WATCHDOG = "KILLED_BY_WATCHDOG"


class DomainParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    control_hz: float = 50.0
    physics_substeps: int = 5
    friction: float = 1.0
    mass_scale: float = 1.0
    sensor_noise_std: float = 0.0
    action_delay_steps: int = 0


class TrainerCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    algo: Literal["ppo", "sac"] = "ppo"
    timesteps: int = Field(default=50_000, ge=1, le=10_000_000)
    lr: float = 3e-4
    batch_size: int = 64
    n_steps: int = 2048
    n_envs: int = 1
    gamma: float = 0.99
    device: str = "cpu"
    seed: int = 0

    @field_validator("timesteps")
    @classmethod
    def _timesteps_positive(cls, v: int) -> int:
        if v < 1 or v > 10_000_000:
            raise ValueError("timesteps must be an integer in [1, 10000000]")
        return int(v)


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    sim: str = "mujoco"
    task: str = "wall_follow"
    robot: str = "diffdrive_lidar"
    arch: str = "mlp"
    arch_cfg: dict[str, Any] = Field(default_factory=lambda: {"hidden_sizes": [64, 64]})
    trainer: TrainerCfg = Field(default_factory=TrainerCfg)
    seeds: list[int] = Field(default_factory=lambda: [0])
    eval_suite: list[str] = Field(default_factory=list)
    domain: DomainParams = Field(default_factory=DomainParams)
    compute: Literal["local", "runpod"] = "local"
    gpu_type: str | None = None
    budget_usd: float = 0.0
    transfer_to: list[str] | None = None


class RunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    status: RunStatus = RunStatus.QUEUED
    config: RunConfig
    step: int = 0
    total_steps: int = 0
    mean_return: float | None = None
    wandb_url: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    pid: int | None = None
