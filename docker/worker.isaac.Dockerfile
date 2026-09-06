# Isaac-capable worker — thin COMPLETE path (no multi-GB Omniverse bake).
# Sets ROBOLAB_ISAAC_MODE=thin so isaaclab/isaac_sim adapters train for real
# (shared MuJoCo corridor / same URDF). Mac stubs stay stubs without this env.
FROM avinashnandyala2/robolab-worker:phase3

ENV ROBOLAB_ISAAC_MODE=thin
