"""Diagnose why the robot falls: check mass, actuator gains, zero-action stability."""
import os, sys
import numpy as np
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_DIR)
import mujoco

scene = os.path.join(PROJECT_DIR, "assets", "scene.xml")
model = mujoco.MjModel.from_xml_path(scene)
data = mujoco.MjData(model)

# 1. Robot mass
total_mass = sum(model.body_mass)
print(f"Total robot mass: {total_mass:.2f} kg")
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    m = model.body_mass[i]
    if m > 0.1:
        print(f"  {name}: {m:.2f} kg")

# 2. Check actuator properties
print(f"\nActuators: {model.nu}")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    gain = model.actuator_gainprm[i, 0]
    bias = model.actuator_biasprm[i, 1]  # kp for position actuator
    frange = model.actuator_forcerange[i]
    print(f"  {name}: kp={gain:.0f}, forcerange=[{frange[0]:.0f}, {frange[1]:.0f}]")

# 3. Zero-action stability test
print("\n--- Zero-action test (hold standing for 2000 steps = 4 sec) ---")
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

# Set ctrl to initial joint positions (home pose)
from envs.humanoid_pick_env import ACTUATED_JOINTS
for i, jname in enumerate(ACTUATED_JOINTS):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    qadr = model.jnt_qposadr[jid]
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{jname}")
    data.ctrl[aid] = data.qpos[qadr]

pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
init_h = data.xpos[pelvis_id, 2]
print(f"Initial pelvis height: {init_h:.3f}")

for step in range(2000):
    mujoco.mj_step(model, data)
    if step % 200 == 0 or step == 1999:
        h = data.xpos[pelvis_id, 2]
        max_qvel = np.max(np.abs(data.qvel))
        quat = data.xquat[pelvis_id]
        up_z = 1.0 - 2.0 * (quat[1]**2 + quat[2]**2)
        print(f"  Step {step:5d}: h={h:.3f}, max_qvel={max_qvel:.2f}, up_z={up_z:.3f}")
    if not np.isfinite(data.qpos).all():
        print(f"  NaN at step {step}!")
        break

final_h = data.xpos[pelvis_id, 2]
print(f"\nFinal pelvis height: {final_h:.3f} (delta: {final_h - init_h:.3f})")
if final_h > 0.6:
    print("PASS: Robot stays standing with zero action")
else:
    print("FAIL: Robot falls with zero action - actuator gains insufficient!")

# 4. Check what happens with env's action mapping
print("\n--- Env action=0 test (through env wrapper) ---")
import envs
import gymnasium as gym
env = gym.make("HumanoidSquatPick-v0", scene_xml=scene, max_episode_steps=2000)
obs, _ = env.reset()
total_r = 0
for i in range(500):
    obs, r, term, trunc, info = env.step(np.zeros(26))
    total_r += r
    if i % 100 == 0:
        print(f"  Step {i}: h={info.get('pelvis_height',0):.3f}, r={r:.2f}, phase={info.get('phase','?')}")
    if term:
        print(f"  Terminated at step {i}, total_reward={total_r:.1f}")
        break
else:
    print(f"  Survived 500 steps! total_reward={total_r:.1f}")
env.close()
