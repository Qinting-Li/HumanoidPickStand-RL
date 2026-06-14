from pathlib import Path

import gymnasium as gym
import numpy as np

import envs  # noqa: F401


PROJECT_DIR = Path(__file__).resolve().parents[1]


def make_env():
    return gym.make(
        "HumanoidSquatPick-v0",
        scene_xml=str(PROJECT_DIR / "assets" / "scene.xml"),
        max_episode_steps=50,
    )


def test_reset_and_step_are_finite():
    env = make_env()
    obs, info = env.reset(seed=0)
    assert env.action_space.shape == (26,)
    assert env.observation_space.shape == (88,)
    assert np.isfinite(obs).all()
    assert "phase" in info

    obs, reward, terminated, truncated, info = env.step(np.zeros(26, dtype=np.float32))
    assert np.isfinite(obs).all()
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "alive_reward" in info
    assert "action_smoothness_penalty" in info
    env.close()

