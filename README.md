# Humanoid Pick-and-Stand Benchmark

An open-source MuJoCo/Gymnasium benchmark for validating humanoid simulation assets, standing control, and pick-and-stand task protocols before training expensive reinforcement learning policies.

This project is built around a 26-actuator humanoid model exported from URDF and compiled for MuJoCo. The focus is not a polished RL demo, but a reproducible engineering pipeline for asset validation, simulator stability diagnosis, controller baselines, and policy evaluation.

## Motivation

Humanoid RL is expensive to debug when the simulator itself is unstable. This repository provides a structured workflow to verify that the robot model loads correctly, resets deterministically, exposes useful diagnostics, and fails clearly when numerical instability occurs.

It supports:

* URDF/MJCF mesh validation
* MuJoCo model sanity checks
* repeatable diagnostic rollouts
* classical PD standing baseline
* evaluation scripts for random, scripted, and RL policies
* regression tests for broken assets, API errors, and NaN/Inf simulation failures

## Model

* **Simulator:** MuJoCo
* **Environment API:** Gymnasium
* **RL Framework:** Stable-Baselines3 PPO
* **Robot:** 60 links, 59 joints, 26 actuated revolute joints, 33 fixed joints
* **Observation dimension:** 88
* **Action dimension:** 26 normalized joint-position targets

The raw `assets/humanoid.urdf` preserves the original exporter paths for provenance. The benchmark uses the generated MuJoCo-compatible files:

* `assets/humanoid_mujoco.urdf`
* `assets/scene.xml`

## Repository Structure

```text
assets/          URDF, MJCF, and STL mesh assets
configs/         Training configuration files
controllers/     Classical baseline controllers
docs/            Asset pipeline, benchmark protocol, and stability notes
envs/            Gymnasium humanoid environment
scripts/         Validation, diagnostics, training, evaluation, and video export
tests/           Pytest regression tests
diagnostics/     Generated diagnostic logs
videos/          Generated rollout videos
```

## Installation

```bash
cd E:\humanoid_squat_pick_rl
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Asset Validation

```bash
python scripts/validate_assets.py
python scripts/validate_mujoco_model.py
```

`validate_assets.py` warns about raw exporter paths in the original URDF, but requires the generated MuJoCo-facing files to pass validation.

## Simulation Diagnostics

```bash
python scripts/run_diagnostics.py
```

This runs zero-action, small-random-action, and PD-standing rollouts, then writes diagnostic logs and `diagnostics/summary.json`.

Falling is not automatically treated as a simulator failure. NaN/Inf state values or explosive acceleration are treated as fatal instability.

## PD Standing Baseline

```bash
python scripts/run_pd_stand.py
```

The PD controller is a diagnostic baseline, not a production whole-body balance controller. It checks whether a conservative neutral pose can run without numerical instability and reports whether the humanoid remains upright.

## RL Training

```bash
python scripts/train.py --config configs/train_config.yaml
```

RL training is available, but simulator validation should be completed before launching expensive training runs.

## Evaluation

```bash
python scripts/evaluate_random_policy.py
python scripts/evaluate_pd_baseline.py
python scripts/evaluate_rl_policy.py --model logs/models/best/best_model.zip
python scripts/export_episode_video.py --policy pd
```

If no RL checkpoint is provided, `evaluate_rl_policy.py` exits cleanly with an explicit message.

## Tests

```bash
pytest tests -q
```

The test suite checks asset references, MuJoCo loading, Gymnasium reset/step behavior, PD action shape, and short-rollout stability without NaN/Inf states.

## Known Limitations

* The current PD controller is a diagnostic standing baseline, not a full humanoid balance controller.
* Zero-action falling is expected and is not considered a simulator bug.
* The pick-and-stand task defines the protocol and reward decomposition, but high-success policies require further training and tuning.
* The raw URDF keeps original exporter paths for provenance; simulation should use the generated MuJoCo assets.

## Roadmap

* Add center-of-pressure and foot-contact diagnostics.
* Add scripted squat and reach baselines.
* Add seed-controlled benchmark result tables.
* Add CI for asset validation and headless MuJoCo tests.
* Release trained reference policies after simulator stability criteria are satisfied.
