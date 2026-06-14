from pathlib import Path

import gymnasium as gym
import numpy as np

import envs  # noqa: F401
from controllers.pd_stand_controller import PDStandController


PROJECT_DIR = Path(__file__).resolve().parents[1]


def make_env(max_steps=80):
    return gym.make(
        "HumanoidSquatPick-v0",
        scene_xml=str(PROJECT_DIR / "assets" / "scene.xml"),
        max_episode_steps=max_steps,
    )


def test_pd_controller_returns_valid_action():
    env = make_env()
    env.reset(seed=1)
    action = PDStandController().predict(env)
    assert action.shape == (26,)
    assert np.isfinite(action).all()
    assert np.all(action <= 1.0)
    assert np.all(action >= -1.0)
    env.close()


def test_short_pd_rollout_has_no_nan_or_inf():
    env = make_env()
    controller = PDStandController()
    obs, _ = env.reset(seed=2)
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(controller.predict(env))
        base = env.unwrapped
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
        assert np.isfinite(base.data.qpos).all()
        assert np.isfinite(base.data.qvel).all()
        assert np.isfinite(base.data.qacc).all()
        assert not info.get("unstable", False)
        if terminated or truncated:
            break
    env.close()
