# RoboLab worker image — RunPod GPU training (Phase 3)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl \
    PATH=/root/.local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      curl \
      ca-certificates \
      libgl1 \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv once in the image; entrypoint still re-checks.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /opt/robolab-worker
COPY robolab/robolab/compute/entrypoint.sh /opt/robolab-worker/entrypoint.sh
RUN chmod +x /opt/robolab-worker/entrypoint.sh

# Code is cloned at GIT_SHA into /workspace/robolab (network volume).
ENTRYPOINT ["/opt/robolab-worker/entrypoint.sh"]
