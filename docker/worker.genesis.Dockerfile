# Genesis-capable worker — lean tag over phase3 (no Isaac).
# genesis-world installs at pod start into the project .venv (entrypoint); volume caches wheels.
FROM avinashnandyala2/robolab-worker:phase3

ENV ROBOLAB_INSTALL_GENESIS=1 \
    ROBOLAB_GENESIS_GPU=1 \
    PYOPENGL_PLATFORM=egl \
    NVIDIA_DRIVER_CAPABILITIES=all \
    PYTHONWARNINGS=ignore
