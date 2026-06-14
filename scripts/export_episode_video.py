"""Export one rollout video if MuJoCo rendering is available."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import envs  # noqa: F401
from controllers.pd_stand_controller import PDStandController


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["zero", "random", "pd"], default="pd")
    parser.add_argument("--steps", type=int, default=250)
    args = parser.parse_args()

    out_dir = PROJECT_DIR / "videos"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.policy}_episode.mp4"

    env = gym.make(
        "HumanoidSquatPick-v0",
        scene_xml=str(PROJECT_DIR / "assets" / "scene.xml"),
        render_mode="rgb_array",
        max_episode_steps=args.steps,
    )
    controller = PDStandController()
    rng = np.random.default_rng(9)
    obs, info = env.reset(seed=9)
    frames = []

    for step in range(args.steps):
        if args.policy == "pd":
            action = controller.predict(env)
        elif args.policy == "random":
            action = rng.uniform(-0.1, 0.1, 26)
        else:
            action = np.zeros(26)
        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        if terminated or truncated:
            break
    env.close()

    if not frames:
        print("No frames rendered. MuJoCo offscreen rendering may be unavailable.")
        return 0

    import imageio

    imageio.mimsave(out_path, frames, fps=50)
    print(f"Saved video: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
