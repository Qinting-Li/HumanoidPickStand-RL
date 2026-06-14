# Humanoid Pick-and-Stand Benchmark

An open-source MuJoCo benchmark for validating humanoid simulation assets,
standing control, and pick-and-stand task protocols before training expensive
reinforcement learning policies.

The benchmark is based on a 26-DOF humanoid model exported from URDF and
compiled to MuJoCo. It focuses on reproducible simulator assets, dynamics
diagnostics, controller baselines, and evaluation scripts rather than a flashy
first RL demo.

## Why This Is Useful

Robotics teams need models that load reliably, reset deterministically, expose
diagnostic signals, and fail loudly on numerical instability. This repository is
structured to support those engineering workflows:

- URDF/MJCF mesh validation
- MuJoCo model sanity checks
- repeatable diagnostic rollouts
- classical PD standing baseline
- benchmark metrics for random, scripted, and RL policies
- tests that catch broken assets and NaN/Inf simulation failures

## Model

- Simulator: MuJoCo
- Environment API: Gymnasium
- RL framework: Stable-Baselines3 PPO
- Robot: 60 links, 59 joints, 26 controlled revolute joints, 33 fixed joints
- Observation dimension: 88
- Action dimension: 26 normalized joint-position targets

The raw `assets/humanoid.urdf` still contains the original exporter paths using
`../meshes/...`. The generated MuJoCo-facing files `assets/humanoid_mujoco.urdf`
and `assets/scene.xml` use valid mesh paths and are the files used by the
benchmark.

## Repository Structure

```text
assets/                  URDF, generated MJCF, and STL meshes
configs/                 Training configuration
controllers/             Classical baseline controllers
docs/                    Asset pipeline, protocol, and stability notes
envs/                    Gymnasium environment
scripts/                 Validation, diagnostics, training, evaluation, video export
tests/                   Pytest regression tests
diagnostics/             Generated CSV/JSON diagnostic outputs
videos/                  Generated rollout videos
```

## Installation

```powershell
cd E:\humanoid_squat_pick_rl
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Asset Validation

```powershell
python scripts/validate_assets.py
python scripts/validate_mujoco_model.py
```

`validate_assets.py` reports the raw URDF exporter paths as a warning, then
requires the generated MuJoCo files to pass.

## Simulation Diagnostics

```powershell
python scripts/run_diagnostics.py
```

This runs zero-action, small-random-action, and PD standing rollouts. It writes
CSV logs and `diagnostics/summary.json`. Falling is not automatically fatal for
a humanoid. NaN/Inf state or explosive acceleration is fatal.

## PD Standing Baseline

```powershell
python scripts/run_pd_stand.py
```

The PD baseline is intentionally simple. It checks whether a conservative
neutral pose can run without numerical instability and reports whether the robot
remained upright.

## RL Training

```powershell
python scripts/train.py --config configs/train_config.yaml
```

Training remains available, but simulator validation should be run first.

## Evaluation

```powershell
python scripts/evaluate_random_policy.py
python scripts/evaluate_pd_baseline.py
python scripts/evaluate_rl_policy.py --model logs/models/best/best_model.zip
python scripts/export_episode_video.py --policy pd
```

If no RL model path is provided, `evaluate_rl_policy.py` prints a clear message
and exits without error.

## Tests

```powershell
pytest tests -q
```

The tests verify asset references, MuJoCo loading, Gymnasium reset/step API
behavior, PD action shape, and a short PD rollout with no NaN/Inf state.

## Known Limitations

- The current PD standing controller is a baseline diagnostic controller, not a
  production whole-body balance controller.
- Zero-action falling is expected and is not treated as a simulator bug.
- The grasp/lift/stand-up task phases define a protocol and reward decomposition,
  but high-success policies still require training and tuning.
- Raw URDF paths are intentionally left unchanged as exporter provenance; use the
  generated MuJoCo assets for simulation.

## Roadmap

- Add center-of-pressure and foot contact diagnostics.
- Add scripted squat/reach baseline beyond standing.
- Add benchmark result tables and seed-controlled evaluation manifests.
- Add CI for asset validation and headless MuJoCo tests.
- Publish trained reference policies only after simulator stability criteria pass.

