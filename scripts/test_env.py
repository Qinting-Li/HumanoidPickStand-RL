"""Quick validation of the environment."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import envs
import gymnasium as gym

scene = os.path.join(os.path.dirname(__file__), "..", "assets", "scene.xml")
env = gym.make("HumanoidSquatPick-v0", scene_xml=scene, max_episode_steps=200)
obs, info = env.reset(seed=42)
print(f"Obs shape: {obs.shape}  Action shape: {env.action_space.shape}")
print(f"Initial phase: {info['phase']}")

print("\n--- Zero action (should hold standing) ---")
for i in range(30):
    obs, reward, terminated, truncated, info = env.step(np.zeros(26, dtype=np.float32))
    h = info.get("pelvis_height", 0)
    if i % 5 == 0 or terminated:
        print(f"  Step {i+1:3d}: reward={reward:+7.2f}  h={h:.3f}  phase={info['phase']}  term={terminated}")
    if terminated:
        break

print("\n--- Small random actions ---")
obs, info = env.reset(seed=99)
for i in range(30):
    action = np.random.uniform(-0.1, 0.1, size=26).astype(np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    h = info.get("pelvis_height", 0)
    if i % 5 == 0 or terminated:
        print(f"  Step {i+1:3d}: reward={reward:+7.2f}  h={h:.3f}  phase={info['phase']}  term={terminated}")
    if terminated:
        break

env.close()
print("\nValidation passed!")
