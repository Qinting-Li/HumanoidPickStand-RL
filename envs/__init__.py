from gymnasium.envs.registration import register

register(
    id="HumanoidSquatPick-v0",
    entry_point="envs.humanoid_pick_env:HumanoidSquatPickEnv",
    max_episode_steps=2000,
)
