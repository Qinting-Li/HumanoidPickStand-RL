"""
Evaluate / visualize a trained humanoid squat-pick-stand agent.

Usage:
    python scripts/evaluate.py --model logs/models/best/best_model.zip
    python scripts/evaluate.py --model logs/models/humanoid_ppo_final.zip --record
"""

import os
import sys
import argparse
import time
import numpy as np

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_DIR)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import envs
import gymnasium as gym


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained humanoid agent")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained model (.zip)",
    )
    parser.add_argument(
        "--vecnorm",
        type=str,
        default=None,
        help="Path to VecNormalize stats (.pkl). Auto-detected if not specified.",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record frames and save as video (requires imageio)",
    )
    parser.add_argument("--render", action="store_true", default=True)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    scene_xml = os.path.join(PROJECT_DIR, "assets", "scene.xml")
    if not os.path.exists(scene_xml):
        print(f"ERROR: Scene XML not found at {scene_xml}")
        sys.exit(1)

    # Create environment
    render_mode = "rgb_array" if args.record else None
    env = gym.make(
        "HumanoidSquatPick-v0",
        scene_xml=scene_xml,
        render_mode=render_mode,
        max_episode_steps=2000,
    )
    vec_env = DummyVecEnv([lambda: env])

    # Load VecNormalize if available
    vecnorm_path = args.vecnorm
    if vecnorm_path is None:
        # Auto-detect
        model_dir = os.path.dirname(args.model)
        candidates = [
            os.path.join(model_dir, "vecnormalize.pkl"),
            os.path.join(model_dir, "..", "vecnormalize.pkl"),
        ]
        for c in candidates:
            if os.path.exists(c):
                vecnorm_path = c
                break

    if vecnorm_path and os.path.exists(vecnorm_path):
        print(f"Loading VecNormalize from: {vecnorm_path}")
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    # Load model
    print(f"Loading model from: {args.model}")
    model = PPO.load(args.model, env=vec_env)

    # Run episodes
    print(f"\nRunning {args.episodes} evaluation episodes...\n")

    all_rewards = []
    all_lengths = []
    frames = []

    for ep in range(args.episodes):
        obs = vec_env.reset()
        done = False
        ep_reward = 0.0
        ep_length = 0
        phase_history = []

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, dones, infos = vec_env.step(action)

            ep_reward += reward[0]
            ep_length += 1
            done = dones[0]

            if infos[0].get("phase"):
                phase_history.append(infos[0]["phase"])

            if args.record:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            if not args.record:
                time.sleep(0.02)  # ~50fps real-time

        # Episode stats
        final_phase = phase_history[-1] if phase_history else "?"
        print(
            f"  Episode {ep + 1}/{args.episodes}: "
            f"reward={ep_reward:.1f}, "
            f"length={ep_length}, "
            f"final_phase={final_phase}"
        )
        all_rewards.append(ep_reward)
        all_lengths.append(ep_length)

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Evaluation Summary ({args.episodes} episodes):")
    print(f"  Mean reward: {np.mean(all_rewards):.1f} +/- {np.std(all_rewards):.1f}")
    print(f"  Mean length: {np.mean(all_lengths):.0f} +/- {np.std(all_lengths):.0f}")
    print(f"{'=' * 50}")

    # Save video
    if args.record and frames:
        try:
            import imageio

            video_path = os.path.join(PROJECT_DIR, "logs", "eval_video.mp4")
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            imageio.mimsave(video_path, frames, fps=50)
            print(f"\nVideo saved to: {video_path}")
        except ImportError:
            print("\nInstall imageio to save videos: pip install imageio[ffmpeg]")

    vec_env.close()


if __name__ == "__main__":
    main()
