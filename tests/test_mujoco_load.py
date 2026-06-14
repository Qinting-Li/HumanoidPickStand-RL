from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

import envs  # noqa: F401


PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_scene_xml_loads():
    model = mujoco.MjModel.from_xml_path(str(PROJECT_DIR / "assets" / "scene.xml"))
    assert model.nbody >= 60
    assert model.nu == 26
    assert model.nsensor >= 3


def test_reset_contacts_have_no_robot_self_collision():
    env = gym.make(
        "HumanoidSquatPick-v0",
        scene_xml=str(PROJECT_DIR / "assets" / "scene.xml"),
        max_episode_steps=50,
    )
    env.reset(seed=0)
    base = env.unwrapped
    foot_ground = 0
    object_ground = 0
    robot_self = 0

    for i in range(base.data.ncon):
        c = base.data.contact[i]
        bodies = {
            int(base.model.geom_bodyid[c.geom1]),
            int(base.model.geom_bodyid[c.geom2]),
        }
        if 0 in bodies and (base._l_foot_id in bodies or base._r_foot_id in bodies):
            foot_ground += 1
        if 0 in bodies and base._object_id in bodies:
            object_ground += 1
        if 0 not in bodies and base._object_id not in bodies:
            robot_self += 1

    assert foot_ground >= 4
    assert object_ground >= 1
    assert robot_self == 0
    assert np.isfinite(base.data.qpos).all()
    env.close()
