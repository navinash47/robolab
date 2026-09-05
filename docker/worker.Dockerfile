# RoboLab worker image (Phase 0 stub — real build in Phase 3)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app
COPY . /app

# Placeholder: Phase 3 will install robolab packages and set the training entrypoint.
CMD ["echo", "robolab worker stub — implement in Phase 3"]
