# Benchmark Protocol

The benchmark task is split into six phases:

- `stand`
- `squat`
- `reach`
- `grasp`
- `lift`
- `stand_up`

Each policy is evaluated with the same environment, reset distribution, action
space, and metrics.

## Metrics

- success rate
- average return
- average episode length
- fall rate
- fatal numerical instability count
- minimum pelvis height
- maximum qacc norm

## Reward Components

The environment `info` dictionary reports:

- `alive_reward`
- `posture_reward`
- `pelvis_height_reward`
- `hand_object_distance_reward`
- `grasp_contact_reward`
- `lift_reward`
- `action_smoothness_penalty`
- `fall_penalty`

These components make it easier to compare scripted baselines and RL policies
without reverse-engineering a single scalar reward.

## Baselines

`controllers/pd_stand_controller.py` provides a diagnostic standing controller.
It is not a full humanoid balance controller. It exists to verify that the model,
reset, actuator mapping, and diagnostic logging behave reproducibly.

