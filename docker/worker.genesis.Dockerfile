# Genesis-capable worker — same bootstrap as phase3 + wheel cache for genesis-world.
# Does NOT bake Isaac. Code + genesis install still come from git checkout + entrypoint.
FROM avinashnandyala2/robolab-worker:phase3

ENV ROBOLAB_INSTALL_GENESIS=1 \
    ROBOLAB_GENESIS_GPU=1 \
    ROBOLAB_GENESIS_WHEEL_DIR=/opt/robolab-wheels

# Pre-download wheels so pod start is not blocked on PyPI when possible.
RUN mkdir -p /opt/robolab-wheels \
    && python3 -m pip download --dest /opt/robolab-wheels genesis-world \
    || echo "WARN: genesis-world wheel download skipped (will pip on pod)"
