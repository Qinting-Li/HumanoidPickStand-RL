"""Evaluate the classical PD standing baseline."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import envs  # noqa: F401
from controllers.pd_stand_controller import PDStandController


def main() -> int:
    episodes = 20
    controller = PDStandController()
    returns, lengths = [], []
    falls = 0
    successes = 0

    for ep in range(episodes):
        env = gym.make(
            "HumanoidSquatPick-v0",
            scene_xml=str(PROJECT_DIR / "assets" / "scene.xml"),
            max_episode_steps=500,
        )
        obs, info = env.reset(seed=100 + ep)
        total = 0.0
        upright = True
        for step in range(500):
            obs, reward, terminated, truncated, info = env.step(controller.predict(env))
            total += float(reward)
            upright = upright and info.get("pelvis_height", 0.0) > 0.55 and info.get("torso_up", 0.0) > 0.65
            if terminated or truncated:
                falls += int(terminated)
                break
        successes += int(upright and not terminated)
        returns.append(total)
        lengths.append(step + 1)
        env.close()

    print(f"Episodes: {episodes}")
    print(f"Success rate: {successes / episodes:.3f}")
    print(f"Average return: {np.mean(returns):.3f}")
    print(f"Average episode length: {np.mean(lengths):.1f}")
    print(f"Fall rate: {falls / episodes:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

