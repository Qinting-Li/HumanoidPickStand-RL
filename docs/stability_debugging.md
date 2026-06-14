# Stability Debugging

Humanoids are unstable under passive zero-action control. A fall during a
zero-action rollout is not automatically a simulator bug.

Numerical instability is different. Treat the following as fatal:

- NaN or Inf in `qpos`, `qvel`, `qacc`, observation, or reward
- explosive acceleration
- invalid actuator ranges or joint limits
- missing meshes or malformed XML

## Important Parameters

- `timestep`: smaller values usually improve contact stability but cost more.
- solver iterations: more iterations can stabilize contact-rich scenes.
- damping and armature: modest values reduce CAD-import chatter; excessive
  values can hide bad dynamics.
- contact friction: high enough for grasping and feet, but not unrealistically
  large.
- reset pose: feet should not start deeply penetrating the ground, and joints
  should avoid singular straight-leg starts.
- actuator target ranges: should match physical joint limits and action scaling.

## Commands

```powershell
python scripts/run_diagnostics.py
python scripts/run_pd_stand.py
```

Diagnostics write CSV traces to `diagnostics/` and summarize fatal instability
in `diagnostics/summary.json`.
