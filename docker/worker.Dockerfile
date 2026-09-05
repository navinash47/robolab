# RoboLab worker image — RunPod GPU training (Phase 3)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=osmesa \
    PATH=/root/.local/bin:$PATH \
    UV_CACHE_DIR=/workspace/.cache/uv \
    XDG_CACHE_HOME=/workspace/.cache

RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      curl \
      ca-certificates \
      libgl1 \
      libglib2.0-0 \
      libosmesa6 \
      libgl1-mesa-glx \
      libegl1 \
      libgles2 \
    && rm -rf /var/lib/apt/lists/*

# Install uv once in the image; entrypoint still re-checks.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /opt/robolab-worker
COPY docker/bootstrap.sh /opt/robolab-worker/bootstrap.sh
RUN chmod +x /opt/robolab-worker/bootstrap.sh

# Bootstrap clones GIT_SHA then execs repo entrypoint.sh (so script fixes ship via git).
ENTRYPOINT ["/opt/robolab-worker/bootstrap.sh"]
