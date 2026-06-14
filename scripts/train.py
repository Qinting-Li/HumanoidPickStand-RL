"""
Train a PPO agent for the humanoid squat-pick-stand task.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/train_config.yaml
    python scripts/train.py --total-timesteps 10000000
"""

import os
import sys
import argparse
import yaml
import numpy as np

# Add project root to path
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_DIR)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.utils import set_random_seed

import envs  # registers the gymnasium environment
import gymnasium as gym


def make_env(scene_xml: str, rank: int, seed: int = 0, max_steps: int = 2000):
    """Create a function that returns a new env instance."""
    def _init():
        env = gym.make(
            "HumanoidSquatPick-v0",
            scene_xml=scene_xml,
            max_episode_steps=max_steps,
        )
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed + rank)
    return _init


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train humanoid squat-pick agent")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_DIR, "configs", "train_config.yaml"),
        help="Path to training config YAML",
    )
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    ppo_cfg = cfg.get("ppo", {})
    env_cfg = cfg.get("env", {})
    log_cfg = cfg.get("logging", {})

    # Override with command line args
    total_timesteps = args.total_timesteps or ppo_cfg.get("total_timesteps", 50_000_000)
    n_envs = args.n_envs or ppo_cfg.get("n_envs", 8)

    # Paths
    scene_xml = os.path.join(PROJECT_DIR, env_cfg.get("scene_xml", "assets/scene.xml"))
    log_dir = os.path.join(PROJECT_DIR, log_cfg.get("log_dir", "logs"))
    tb_log = os.path.join(PROJECT_DIR, log_cfg.get("tensorboard_log", "logs/tensorboard"))
    model_dir = os.path.join(log_dir, "models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tb_log, exist_ok=True)

    # Check scene exists
    if not os.path.exists(scene_xml):
        print(f"ERROR: Scene XML not found at {scene_xml}")
        print("Run 'python scripts/build_scene.py' first!")
        sys.exit(1)

    print("=" * 60)
    print("Humanoid Squat-Pick-Stand RL Training")
    print("=" * 60)
    print(f"  Scene:          {scene_xml}")
    print(f"  N envs:         {n_envs}")
    print(f"  Total steps:    {total_timesteps:,}")
    print(f"  Log dir:        {log_dir}")
    print(f"  Seed:           {args.seed}")
    print()

    # Create vectorized environments
    max_steps = env_cfg.get("max_episode_steps", 2000)
    env_fns = [make_env(scene_xml, i, args.seed, max_steps) for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Create eval environment
    eval_env_fns = [make_env(scene_xml, 100 + i, args.seed, max_steps) for i in range(2)]
    eval_vec_env = SubprocVecEnv(eval_env_fns)
    eval_vec_env = VecMonitor(eval_vec_env)
    eval_vec_env = VecNormalize(
        eval_vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0
    )

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1),  # ~50K steps between saves
        save_path=model_dir,
        name_prefix="humanoid_ppo",
        save_vecnormalize=True,
    )
    eval_cb = EvalCallback(
        eval_vec_env,
        best_model_save_path=os.path.join(model_dir, "best"),
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=max(log_cfg.get("eval_freq", 100_000) // n_envs, 1),
        n_eval_episodes=log_cfg.get("eval_episodes", 5),
        deterministic=True,
    )
    callbacks = CallbackList([checkpoint_cb, eval_cb])

    # Create or load model
    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = PPO.load(
            args.resume,
            env=vec_env,
            tensorboard_log=tb_log,
        )
        # Calculate remaining steps from checkpoint filename
        import re
        m = re.search(r"(\d+)_steps", os.path.basename(args.resume))
        if m:
            done_steps = int(m.group(1))
            remaining = max(total_timesteps - done_steps, 0)
            print(f"  Checkpoint was at {done_steps:,} steps")
            print(f"  Remaining steps:  {remaining:,}")
            total_timesteps = remaining  # only train the remainder
    else:
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=ppo_cfg.get("learning_rate", 1e-4),
            n_steps=ppo_cfg.get("n_steps", 4096),
            batch_size=ppo_cfg.get("batch_size", 256),
            n_epochs=ppo_cfg.get("n_epochs", 10),
            gamma=ppo_cfg.get("gamma", 0.99),
            gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
            clip_range=ppo_cfg.get("clip_range", 0.1),
            ent_coef=ppo_cfg.get("ent_coef", 0.001),
            vf_coef=ppo_cfg.get("vf_coef", 0.5),
            max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
            target_kl=0.02,  # stop update early if KL divergence too high
            tensorboard_log=tb_log,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                log_std_init=-1.0,  # initial std≈0.37, safe exploration
            ),
            verbose=1,
            seed=args.seed,
        )

    # Train
    print("\nStarting training...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=not bool(args.resume),  # keep counter when resuming
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    # Save final model
    final_path = os.path.join(model_dir, "humanoid_ppo_final")
    model.save(final_path)
    vec_env.save(os.path.join(model_dir, "vecnormalize.pkl"))
    print(f"\nFinal model saved to: {final_path}")

    vec_env.close()
    eval_vec_env.close()
    print("Training complete!")


if __name__ == "__main__":
    main()
