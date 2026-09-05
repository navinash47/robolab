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

worker-image:
	@echo "Phase 0 stub: docker/worker.Dockerfile is a placeholder. Real image lands in Phase 3."
	@echo "Would run: docker build -f docker/worker.Dockerfile -t robolab-worker ."

test:
	@echo "See docs/PHASE_1_TEST.md for the Phase 1 human gate."
	@echo "Optional smoke: uv run --env-file .env --package robolab python -c \"import robolab.sims.mujoco, robolab.tasks, robolab.archs; print('ok')\""
