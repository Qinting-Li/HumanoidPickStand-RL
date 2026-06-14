"""Run the PD standing baseline for 10 seconds."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import envs  # noqa: F401
from controllers.pd_stand_controller import PDStandController


def main() -> int:
    out_dir = PROJECT_DIR / "diagnostics"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "pd_stand.csv"

    env = gym.make(
        "HumanoidSquatPick-v0",
        scene_xml=str(PROJECT_DIR / "assets" / "scene.xml"),
        max_episode_steps=500,
    )
    controller = PDStandController()
    obs, info = env.reset(seed=11)
    rows = []
    fatal = False
    upright = True

    for step in range(500):
        action = controller.predict(env)
        obs, reward, terminated, truncated, info = env.step(action)
        base = env.unwrapped
        finite = (
            np.isfinite(obs).all()
            and np.isfinite(reward)
            and np.isfinite(base.data.qpos).all()
            and np.isfinite(base.data.qvel).all()
            and np.isfinite(base.data.qacc).all()
        )
        fatal = fatal or (not finite) or bool(info.get("unstable", False))
        upright = upright and info.get("pelvis_height", 0.0) > 0.55 and info.get("torso_up", 0.0) > 0.65
        rows.append(
            {
                "step": step,
                "reward": float(reward),
                "pelvis_height": float(info.get("pelvis_height", 0.0)),
                "torso_up": float(info.get("torso_up", 0.0)),
                "qacc_norm": float(info.get("qacc_norm", 0.0)),
                "contact_count": int(info.get("contact_count", 0)),
                "terminated": bool(terminated),
                "finite": bool(finite),
            }
        )
        if fatal or terminated or truncated:
            break

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    env.close()
    print(f"Saved: {out_path}")
    print(f"Steps: {len(rows)}")
    print(f"Remained upright: {upright}")
    print(f"Fatal instability: {fatal}")
    return 1 if fatal else 0


if __name__ == "__main__":
    raise SystemExit(main())

