.PHONY: dev worker-image test sync

# Ports:
#   FastAPI backend  http://127.0.0.1:8000
#   Vite dashboard   http://localhost:5173
dev: sync
	@test -f .env || cp .env.example .env
	@cd web && npm install
	@echo "Starting RoboLab API on :8000 and web on :5173"
	@trap 'kill 0' EXIT INT TERM; \
	uv run --env-file .env --package robolab-api \
	  uvicorn robolab_api.main:app --host 127.0.0.1 --port 8000 --reload & \
	cd web && npm run dev

sync:
	uv sync --all-packages --python 3.11

# Build the RunPod worker image. Tag must match ROBOLAB_WORKER_IMAGE in .env
# (registry that RunPod can pull). Requires Docker Desktop or Colima.
# Example:
#   make worker-image IMAGE=yourdockerhub/robolab-worker:phase3
#   docker login && docker push yourdockerhub/robolab-worker:phase3
# See docs/PHASE_3_SETUP.md for GHCR and full gate checklist.
IMAGE ?= yourdockerhub/robolab-worker:phase3
worker-image:
	@command -v docker >/dev/null || { echo "docker not found — install Docker Desktop or: brew install colima docker && colima start"; exit 1; }
	docker build -f docker/worker.Dockerfile -t $(IMAGE) .
	@echo "Built $(IMAGE)."
	@echo "Next: docker login && docker push $(IMAGE)"
	@echo "Then set ROBOLAB_WORKER_IMAGE=$(IMAGE) in .env"

test:
	@echo "See docs/PHASE_3_TEST.md for the Phase 3 human gate."
	uv run --package robolab-api python -m pytest robolab_api/tests -q --tb=short
