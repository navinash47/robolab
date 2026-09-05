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
# (registry that RunPod can pull). Example:
#   make worker-image IMAGE=yourdockerhub/robolab-worker:phase3
#   docker push yourdockerhub/robolab-worker:phase3
IMAGE ?= robolab/worker:phase3
worker-image:
	docker build -f docker/worker.Dockerfile -t $(IMAGE) .
	@echo "Built $(IMAGE). Push to a registry RunPod can pull, then set ROBOLAB_WORKER_IMAGE=$(IMAGE) in .env"

test:
	@echo "See docs/PHASE_3_TEST.md for the Phase 3 human gate."
	uv run --package robolab-api python -m pytest robolab_api/tests -q --tb=short
