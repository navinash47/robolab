.PHONY: dev worker-image worker-image-genesis worker-image-isaac test sync

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

# Darwin + Xcode 16+/Apple Clang 17: pybullet's bundled zlib breaks unless
# TARGET_OS_MAC is not auto-defined (see docs/PHASE_5_APIS.md).
sync:
	@if [ "$$(uname -s)" = "Darwin" ]; then \
	  CFLAGS="-fno-define-target-os-macros" uv sync --all-packages --python 3.11; \
	else \
	  uv sync --all-packages --python 3.11; \
	fi

# Build the RunPod worker image. Tag must match ROBOLAB_WORKER_IMAGE in .env
# (registry that RunPod can pull). Requires Docker Desktop or Colima.
# Example:
#   make worker-image IMAGE=yourdockerhub/robolab-worker:phase3
#   docker login && docker push yourdockerhub/robolab-worker:phase3
# Variants (require local/Hub phase3 base):
#   make worker-image-genesis && docker push …:genesis
#   make worker-image-isaac && docker push …:isaac
IMAGE ?= avinashnandyala2/robolab-worker:phase3
IMAGE_GENESIS ?= avinashnandyala2/robolab-worker:genesis
IMAGE_ISAAC ?= avinashnandyala2/robolab-worker:isaac
# RunPod GPUs are amd64; force platform even on Apple Silicon / Colima aarch64.
PLATFORM ?= linux/amd64
worker-image:
	@command -v docker >/dev/null || { echo "docker not found — install Docker Desktop or: brew install colima docker && colima start"; exit 1; }
	docker build --platform $(PLATFORM) -f docker/worker.Dockerfile -t $(IMAGE) .
	@echo "Built $(IMAGE) ($(PLATFORM))."
	@echo "Next: docker login && docker push $(IMAGE)"
	@echo "Then set ROBOLAB_WORKER_IMAGE=$(IMAGE) in .env"

worker-image-genesis:
	@command -v docker >/dev/null || { echo "docker not found"; exit 1; }
	docker build --platform $(PLATFORM) -f docker/worker.genesis.Dockerfile -t $(IMAGE_GENESIS) .
	@echo "Built $(IMAGE_GENESIS). Push then set ROBOLAB_WORKER_IMAGE_GENESIS=$(IMAGE_GENESIS)"

worker-image-isaac:
	@command -v docker >/dev/null || { echo "docker not found"; exit 1; }
	docker build --platform $(PLATFORM) -f docker/worker.isaac.Dockerfile -t $(IMAGE_ISAAC) .
	@echo "Built $(IMAGE_ISAAC). Push then set ROBOLAB_WORKER_IMAGE_ISAAC=$(IMAGE_ISAAC)"

test:
	@echo "See docs/PHASE_5_TEST.md for the Phase 5 human gate."
	uv run --package robolab-api python -m pytest robolab_api/tests -q --tb=short
