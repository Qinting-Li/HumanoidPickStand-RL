"""Stress test: run 10 episodes with fully random actions to check for NaN instabilities."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import envs
import gymnasium as gym

scene = os.path.join(os.path.dirname(__file__), "..", "assets", "scene.xml")
env = gym.make("HumanoidSquatPick-v0", scene_xml=scene, max_episode_steps=500)

unstable_count = 0
for ep in range(10):
    obs, info = env.reset(seed=ep)
    steps_done = 0
    was_unstable = False
    for step in range(500):
        action = np.random.uniform(-1, 1, size=26).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        steps_done = step + 1
        if info.get("unstable", False):
            unstable_count += 1
            was_unstable = True
            break
        if terminated or truncated:
            break
    print(f"  Ep {ep}: steps={steps_done}, term={terminated}, unstable={was_unstable}")

env.close()
print(f"\nUnstable episodes: {unstable_count}/10")
