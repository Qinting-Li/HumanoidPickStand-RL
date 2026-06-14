"""Validate the compiled MuJoCo scene for benchmark use."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import mujoco
from envs.humanoid_pick_env import NUM_ACTUATED


def main() -> int:
    scene = PROJECT_DIR / "assets" / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    failures: list[str] = []
    warnings: list[str] = []

    print("MuJoCo model validation")
    print(f"Scene: {scene}")
    print(f"Bodies:    {model.nbody}")
    print(f"Joints:    {model.njnt}")
    print(f"Actuators: {model.nu}")
    print(f"Sensors:   {model.nsensor}")
    print(f"Geoms:     {model.ngeom}")
    print(f"Contacts:  {model.npair}")

    if model.nu != NUM_ACTUATED:
        failures.append(f"actuator count {model.nu} != action dimension {NUM_ACTUATED}")

    ctrl_ranges = model.actuator_ctrlrange[: model.nu]
    if not np.isfinite(ctrl_ranges).all():
        failures.append("actuator ctrlrange contains NaN/Inf")
    for i in range(model.nu):
        if model.actuator_ctrllimited[i] and ctrl_ranges[i, 0] >= ctrl_ranges[i, 1]:
            failures.append(f"invalid ctrlrange for actuator index {i}")

    gear = np.abs(model.actuator_gear[: model.nu])
    if np.any(gear > 1000.0):
        warnings.append("one or more actuator gear values exceed 1000")

    for i in range(model.njnt):
        if model.jnt_limited[i]:
            lo, hi = model.jnt_range[i]
            if not np.isfinite([lo, hi]).all() or lo >= hi:
                failures.append(f"invalid joint limit at joint index {i}")

    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        failures.append("initial qpos/qvel contains NaN/Inf")
    if not np.isfinite(model.body_mass).all() or np.any(model.body_mass < 0):
        failures.append("body mass contains non-finite or negative values")
    if not np.isfinite(model.body_inertia).all() or np.any(model.body_inertia < 0):
        failures.append("body inertia contains non-finite or negative values")
    zero_mass = int(np.sum(model.body_mass[1:] == 0))
    if zero_mass:
        warnings.append(f"{zero_mass} non-world bodies have zero mass, usually fixed helper bodies")

    total_mass = float(np.sum(model.body_mass))
    print(f"Total mass: {total_mass:.3f} kg")
    if total_mass < 10.0 or total_mass > 200.0:
        warnings.append(f"total mass {total_mass:.3f} kg is outside expected humanoid range")

    for msg in warnings:
        print(f"WARN: {msg}")
    for msg in failures:
        print(f"FAIL: {msg}")

    passed = not failures
    print(f"SUMMARY: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
