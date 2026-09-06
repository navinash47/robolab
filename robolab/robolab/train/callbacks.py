"""Training callbacks: W&B logging + backend heartbeat."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback


def _post_json(url: str, payload: dict[str, Any], timeout: float = 15.0) -> None:
    """POST JSON to backend. Retries on transient tunnel/WAF failures."""
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        # Cloudflare Bot Fight often 403s bare urllib from datacenter IPs.
        "User-Agent": "RoboLabWorker/1.0 (+https://github.com/navinash47/robolab)",
        "Accept": "application/json",
    }
    last_exc: BaseException | None = None
    for attempt in range(1, 6):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp.read()
            return
        except urllib.error.HTTPError as exc:
            last_exc = exc
            # 403/502/530 are often CF tunnel/WAF flaps — retry.
            if exc.code not in {403, 408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 530}:
                print(f"[robolab] heartbeat POST failed: {exc}")
                return
            print(f"[robolab] heartbeat POST HTTP {exc.code} (attempt {attempt}/5); retrying")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            print(f"[robolab] heartbeat POST failed (attempt {attempt}/5): {exc}")
        time.sleep(min(8.0, 0.5 * (2 ** (attempt - 1))))
    if last_exc is not None:
        print(f"[robolab] heartbeat POST gave up: {last_exc}")


class WandbAndHeartbeatCallback(BaseCallback):
    def __init__(
        self,
        run_id: str,
        total_timesteps: int,
        backend_url: str,
        wandb_run: Any | None = None,
        heartbeat_every_steps: int = 512,
        param_count: int | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.run_id = run_id
        self.total_timesteps = total_timesteps
        self.backend_url = backend_url.rstrip("/")
        self.wandb_run = wandb_run
        self.heartbeat_every_steps = max(1, heartbeat_every_steps)
        self.param_count = param_count
        self._last_hb_t = 0.0
        self._last_mean: float | None = None
        self._wandb_url: str | None = getattr(wandb_run, "url", None) if wandb_run else None

    def _mean_return(self) -> float | None:
        # Prefer SB3 logger value (requires Monitor / make_vec_env)
        if self.logger is not None:
            val = self.logger.name_to_value.get("rollout/ep_rew_mean")
            if val is not None:
                self._last_mean = float(val)
                return self._last_mean
        # Fallback: episode info buffer
        if len(self.model.ep_info_buffer) > 0:
            rewards = [ep["r"] for ep in self.model.ep_info_buffer if "r" in ep]
            if rewards:
                self._last_mean = float(sum(rewards) / len(rewards))
                return self._last_mean
        return self._last_mean

    def _heartbeat(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_hb_t) < 2.0 and self.num_timesteps % self.heartbeat_every_steps != 0:
            return
        mean_ret = self._mean_return()
        eta = None
        if self.num_timesteps > 0:
            # rough ETA unused by UI but included per prompt
            eta = None
        payload = {
            "step": int(self.num_timesteps),
            "total": int(self.total_timesteps),
            "mean_return": mean_ret,
            "eta": eta,
            "wandb_url": self._wandb_url,
            "status": "RUNNING",
            "param_count": self.param_count,
        }
        _post_json(f"{self.backend_url}/api/runs/{self.run_id}/heartbeat", payload)
        self._last_hb_t = now

        if self.wandb_run is not None and mean_ret is not None:
            self.wandb_run.log(
                {
                    "rollout/ep_rew_mean": mean_ret,
                    "train/step": int(self.num_timesteps),
                },
                step=int(self.num_timesteps),
            )

    def _on_training_start(self) -> None:
        self._heartbeat(force=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.heartbeat_every_steps == 0:
            self._heartbeat()
        return True

    def _on_training_end(self) -> None:
        self._heartbeat(force=True)
