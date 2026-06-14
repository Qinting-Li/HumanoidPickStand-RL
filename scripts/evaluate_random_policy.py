"""Evaluate a random policy with benchmark metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import envs  # noqa: F401


def main() -> int:
    episodes = 20
    rng = np.random.default_rng(42)
    returns, lengths = [], []
    falls = 0
    successes = 0

    for ep in range(episodes):
        env = gym.make(
            "HumanoidSquatPick-v0",
            scene_xml=str(PROJECT_DIR / "assets" / "scene.xml"),
            max_episode_steps=500,
        )
        obs, info = env.reset(seed=ep)
        total = 0.0
        final_phase = info.get("phase")
        for step in range(500):
            action = rng.uniform(-1.0, 1.0, size=env.action_space.shape)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            final_phase = info.get("phase", final_phase)
            if terminated or truncated:
                falls += int(terminated)
                break
        successes += int(final_phase == "stand_up" and not terminated)
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

