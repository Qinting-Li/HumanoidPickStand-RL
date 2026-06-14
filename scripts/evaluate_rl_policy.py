"""Evaluate a Stable-Baselines3 PPO policy if a model is available."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import envs  # noqa: F401


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    if not args.model:
        print("No --model path provided; skipping RL policy evaluation.")
        return 0
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return 0

    from stable_baselines3 import PPO

    env = gym.make(
        "HumanoidSquatPick-v0",
        scene_xml=str(PROJECT_DIR / "assets" / "scene.xml"),
        max_episode_steps=500,
    )
    model = PPO.load(str(model_path), env=env)
    returns, lengths = [], []
    falls = 0
    successes = 0

    for ep in range(args.episodes):
        obs, info = env.reset(seed=200 + ep)
        total = 0.0
        final_phase = info.get("phase")
        for step in range(500):
            action, _ = model.predict(obs, deterministic=True)
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
    print(f"Episodes: {args.episodes}")
    print(f"Success rate: {successes / args.episodes:.3f}")
    print(f"Average return: {np.mean(returns):.3f}")
    print(f"Average episode length: {np.mean(lengths):.1f}")
    print(f"Fall rate: {falls / args.episodes:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

