"""PD-style neutral standing controller.

The environment action is a normalized joint-position target. This controller
therefore outputs small corrective target offsets around the benchmark neutral
pose instead of raw torques.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from envs.humanoid_pick_env import ACTUATED_JOINTS, NEUTRAL_STAND_POSE, NUM_ACTUATED


@dataclass
class PDStandController:
    """Joint-space PD baseline that matches the 26 actuator controls."""

    kp: float = 0.35
    kd: float = 0.035
    max_action: float = 0.35

    def target_qpos(self, env) -> np.ndarray:
        """Return neutral joint targets in actuator order."""
        target = env.unwrapped._home_qpos.copy()
        for i, name in enumerate(ACTUATED_JOINTS):
            if name in NEUTRAL_STAND_POSE:
                target[i] = NEUTRAL_STAND_POSE[name]
        return np.clip(
            target,
            env.unwrapped._joint_range[:, 0],
            env.unwrapped._joint_range[:, 1],
        )

    def predict(self, env) -> np.ndarray:
        """Compute a finite normalized action for a Gymnasium env instance."""
        base = env.unwrapped
        current = base.data.qpos[base._joint_qpos_idx]
        velocity = base.data.qvel[base._joint_qvel_idx]
        target = self.target_qpos(env)

        desired = target + self.kp * (target - current) - self.kd * velocity
        desired = np.clip(desired, base._joint_range[:, 0], base._joint_range[:, 1])

        scale = np.where(base._action_scale > 1e-6, base._action_scale, 1.0)
        action = (desired - base._home_qpos) / scale
        action = np.clip(action, -self.max_action, self.max_action)
        if action.shape != (NUM_ACTUATED,):
            raise ValueError(f"PD action shape {action.shape} != {(NUM_ACTUATED,)}")
        return action.astype(np.float32)

