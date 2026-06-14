"""
Humanoid Squat-Pick-Stand Environment

A Gymnasium environment for training a humanoid robot to:
1. Stand stably
2. Squat down
3. Reach and grasp an object with both hands
4. Stand back up while holding the object

Uses MuJoCo for physics simulation.
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

# Phase definitions
PHASE_STAND = 0
PHASE_SQUAT = 1
PHASE_REACH = 2
PHASE_GRASP = 3
PHASE_LIFT = 4
PHASE_STANDUP = 5
NUM_PHASES = 6

PHASE_NAMES = ["stand", "squat", "reach", "grasp", "lift", "stand_up"]

# Actuated joint names (must match scene.xml)
ACTUATED_JOINTS = [
    "hip_roll_l_joint", "hip_yaw_l_joint", "hip_pitch_l_joint",
    "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
    "hip_roll_r_joint", "hip_yaw_r_joint", "hip_pitch_r_joint",
    "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
    "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
    "elbow_roll_l_joint", "elbow_yaw_l_joint",
    "wrist_roll_l_joint", "wrist_pitch_l_joint",
    "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint",
    "elbow_roll_r_joint", "elbow_yaw_r_joint",
    "wrist_roll_r_joint", "wrist_pitch_r_joint",
]
NUM_ACTUATED = len(ACTUATED_JOINTS)  # 26

NEUTRAL_STAND_POSE = {
    "hip_pitch_l_joint": -0.08,
    "knee_pitch_l_joint": 0.16,
    "ankle_pitch_l_joint": -0.08,
    "hip_pitch_r_joint": -0.08,
    "knee_pitch_r_joint": 0.16,
    "ankle_pitch_r_joint": -0.08,
    "shoulder_roll_l_joint": 0.18,
    "elbow_roll_l_joint": -0.25,
    "shoulder_roll_r_joint": -0.18,
    "elbow_roll_r_joint": 0.25,
}


class HumanoidSquatPickEnv(gym.Env):
    """Humanoid squat-pick-stand whole-body control environment."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        scene_xml: str = None,
        render_mode: str = None,
        max_episode_steps: int = 2000,
        control_dt: float = 0.02,
    ):
        super().__init__()
        self.render_mode = render_mode

        # Locate scene XML
        if scene_xml is None:
            project_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..")
            )
            scene_xml = os.path.join(project_dir, "assets", "scene.xml")

        if not os.path.exists(scene_xml):
            raise FileNotFoundError(
                f"Scene XML not found: {scene_xml}\n"
                "Run 'python scripts/build_scene.py' first."
            )

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)

        # Timing
        self.sim_dt = self.model.opt.timestep
        self.control_dt = control_dt
        self.n_substeps = max(1, int(round(control_dt / self.sim_dt)))

        # Max steps
        self._max_episode_steps = max_episode_steps
        self._step_count = 0

        # ---- Joint / actuator index mapping ----
        self._setup_indices()

        # ---- Action space: normalized [-1, 1] for each actuated joint ----
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(NUM_ACTUATED,), dtype=np.float32
        )

        # ---- Observation space ----
        obs_size = self._get_obs_size()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # ---- Phase state machine ----
        self._phase = PHASE_STAND
        self._phase_step = 0

        # Phase parameters
        self._stand_steps = 50
        self._squat_target_h = 0.50
        self._reach_threshold = 0.08
        self._grasp_hold_steps = 30
        self._lift_hold_steps = 30
        self._standup_target_h = 0.85

        # ---- Previous action for smoothness penalty ----
        self._prev_action = np.zeros(NUM_ACTUATED, dtype=np.float32)

        # ---- Renderer ----
        self._renderer = None
        if self.render_mode in ("human", "rgb_array"):
            self._init_renderer()

        # ---- Store initial qpos/qvel for reset ----
        mujoco.mj_forward(self.model, self.data)
        self._init_qpos = self.data.qpos.copy()
        self._init_qvel = self.data.qvel.copy()

        # ---- Home pose: the initial joint positions define the standing pose ----
        # action=0 should map to this pose, so the robot stays standing
        self._home_qpos = self._build_neutral_joint_pose()
        # Compute per-joint action scale: how far (in rad) a unit action moves
        # Use 0.3× half-range so initial random policy makes small, safe movements.
        # The agent can still reach full range via sustained actions over time.
        jrange = self._joint_range
        half_range = 0.5 * (jrange[:, 1] - jrange[:, 0])
        self._action_scale = 0.3 * np.clip(half_range, 0.05, np.inf)
        # Maximum velocity allowed (rad/s) – safety clamp
        self._max_qvel = 10.0
        self._fatal_qacc_norm = 1.0e5

    def _build_neutral_joint_pose(self) -> np.ndarray:
        """Return conservative neutral standing targets in actuator order."""
        pose = self.data.qpos[self._joint_qpos_idx].copy()
        for i, name in enumerate(ACTUATED_JOINTS):
            if name in NEUTRAL_STAND_POSE:
                pose[i] = NEUTRAL_STAND_POSE[name]
        return np.clip(pose, self._joint_range[:, 0], self._joint_range[:, 1])

    def _setup_indices(self):
        """Build index mappings for joints, bodies, sites."""
        # Actuated joint qpos/qvel indices
        self._joint_qpos_idx = []
        self._joint_qvel_idx = []
        self._joint_range = np.zeros((NUM_ACTUATED, 2))

        for i, jname in enumerate(ACTUATED_JOINTS):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise ValueError(f"Joint '{jname}' not found in MuJoCo model")
            qadr = self.model.jnt_qposadr[jid]
            vadr = self.model.jnt_dofadr[jid]
            self._joint_qpos_idx.append(qadr)
            self._joint_qvel_idx.append(vadr)
            self._joint_range[i] = self.model.jnt_range[jid]

        self._joint_qpos_idx = np.array(self._joint_qpos_idx)
        self._joint_qvel_idx = np.array(self._joint_qvel_idx)

        # Actuator indices
        self._act_idx = []
        for jname in ACTUATED_JOINTS:
            act_name = f"act_{jname}"
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            if aid < 0:
                raise ValueError(f"Actuator '{act_name}' not found in MuJoCo model")
            self._act_idx.append(aid)
        self._act_idx = np.array(self._act_idx)

        # Root (pelvis) freejoint index
        root_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        if root_jid >= 0:
            self._root_qpos_adr = self.model.jnt_qposadr[root_jid]
            self._root_qvel_adr = self.model.jnt_dofadr[root_jid]
        else:
            self._root_qpos_adr = 0
            self._root_qvel_adr = 0

        # Body indices
        self._pelvis_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )
        self._l_hand_id = self._find_body_id("L_hand_base_link")
        self._r_hand_id = self._find_body_id("R_hand_base_link")
        self._object_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pick_object"
        )

        # Object freejoint index
        obj_jid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_free"
        )
        if obj_jid >= 0:
            self._obj_qpos_adr = self.model.jnt_qposadr[obj_jid]
        else:
            self._obj_qpos_adr = -1

        # Site indices (for palm contact sensing)
        self._l_palm_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "l_palm_site"
        )
        self._r_palm_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "r_palm_site"
        )

        # Foot body indices for support polygon
        self._l_foot_id = self._find_body_id("ankle_roll_l_link")
        self._r_foot_id = self._find_body_id("ankle_roll_r_link")

    def _find_body_id(self, name: str) -> int:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return bid if bid >= 0 else 0

    def _get_obs_size(self) -> int:
        """Compute the observation vector dimension."""
        # joint_pos(26) + joint_vel(26) + pelvis_pos(3) + pelvis_quat(4)
        # + pelvis_linvel(3) + pelvis_angvel(3) + com_pos(3)
        # + object_pos(3) + object_quat(4) + l_hand_pos(3) + r_hand_pos(3)
        # + phase_onehot(6) + phase_timer(1)
        return 26 + 26 + 3 + 4 + 3 + 3 + 3 + 3 + 4 + 3 + 3 + 6 + 1  # = 88

    def _get_obs(self) -> np.ndarray:
        """Build the observation vector."""
        d = self.data

        # Joint positions and velocities (normalized to [-1, 1])
        joint_pos_raw = d.qpos[self._joint_qpos_idx]
        jrange = self._joint_range
        jmid = 0.5 * (jrange[:, 0] + jrange[:, 1])
        jscale = 0.5 * (jrange[:, 1] - jrange[:, 0])
        jscale = np.where(jscale > 1e-6, jscale, 1.0)
        joint_pos = (joint_pos_raw - jmid) / jscale

        joint_vel = d.qvel[self._joint_qvel_idx] * 0.1  # scale down velocities

        # Pelvis state
        pelvis_pos = d.qpos[self._root_qpos_adr : self._root_qpos_adr + 3].copy()
        pelvis_quat = d.qpos[self._root_qpos_adr + 3 : self._root_qpos_adr + 7].copy()
        pelvis_linvel = d.qvel[self._root_qvel_adr : self._root_qvel_adr + 3].copy() * 0.1
        pelvis_angvel = d.qvel[self._root_qvel_adr + 3 : self._root_qvel_adr + 6].copy() * 0.1

        # Center of mass
        com_pos = self._get_com().copy()

        # Object state
        if self._obj_qpos_adr >= 0:
            obj_pos = d.qpos[self._obj_qpos_adr : self._obj_qpos_adr + 3].copy()
            obj_quat = d.qpos[self._obj_qpos_adr + 3 : self._obj_qpos_adr + 7].copy()
        else:
            obj_pos = np.zeros(3)
            obj_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Hand positions (world frame)
        l_hand_pos = d.xpos[self._l_hand_id].copy()
        r_hand_pos = d.xpos[self._r_hand_id].copy()

        # Phase encoding
        phase_onehot = np.zeros(NUM_PHASES, dtype=np.float32)
        phase_onehot[self._phase] = 1.0
        phase_timer = np.array(
            [min(self._phase_step / 200.0, 1.0)], dtype=np.float32
        )

        obs = np.concatenate([
            joint_pos,       # 26
            joint_vel,       # 26
            pelvis_pos,      # 3
            pelvis_quat,     # 4
            pelvis_linvel,   # 3
            pelvis_angvel,   # 3
            com_pos,         # 3
            obj_pos,         # 3
            obj_quat,        # 4
            l_hand_pos,      # 3
            r_hand_pos,      # 3
            phase_onehot,    # 6
            phase_timer,     # 1
        ]).astype(np.float32)

        return obs

    def _get_com(self) -> np.ndarray:
        """Compute robot center of mass (excluding the pick object)."""
        total_mass = 0.0
        com = np.zeros(3)
        for i in range(self.model.nbody):
            # Skip world body (0) and pick_object
            if i == 0 or i == self._object_id:
                continue
            m = self.model.body_mass[i]
            total_mass += m
            com += m * self.data.xipos[i]
        if total_mass > 0:
            com /= total_mass
        return com

    def _action_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        """Map normalized action [-1, 1] to joint position targets.

        action=0 -> home (standing) pose
        action=+/-1 -> home +/- half joint range
        Targets are clamped to joint limits.
        """
        action = np.clip(action, -1.0, 1.0)
        targets = self._home_qpos + action * self._action_scale
        # Clamp to joint limits
        targets = np.clip(targets, self._joint_range[:, 0], self._joint_range[:, 1])
        return targets

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)

        # Set a reproducible, slightly flexed standing pose. This avoids
        # perfectly straight-leg starts while keeping the reset physically modest.
        self.data.qpos[:] = self._init_qpos[:]
        self.data.qpos[self._root_qpos_adr : self._root_qpos_adr + 3] = [
            0.0,
            0.0,
            0.90,
        ]
        self.data.qpos[self._root_qpos_adr + 3 : self._root_qpos_adr + 7] = [
            1.0,
            0.0,
            0.0,
            0.0,
        ]
        self.data.qpos[self._joint_qpos_idx] = self._home_qpos
        self.data.qvel[:] = 0.0

        # Add small random perturbation to joint positions
        if self.np_random is not None:
            noise = self.np_random.uniform(-0.005, 0.005, size=NUM_ACTUATED)
            self.data.qpos[self._joint_qpos_idx] += noise

        # Randomize object position slightly
        if self._obj_qpos_adr >= 0:
            if self.np_random is not None:
                self.data.qpos[self._obj_qpos_adr] = 0.35 + self.np_random.uniform(
                    -0.05, 0.05
                )
                self.data.qpos[self._obj_qpos_adr + 1] = self.np_random.uniform(
                    -0.05, 0.05
                )
            self.data.qpos[self._obj_qpos_adr + 2] = 0.04
            # Reset object orientation to identity
            self.data.qpos[self._obj_qpos_adr + 3] = 1.0
            self.data.qpos[self._obj_qpos_adr + 4 : self._obj_qpos_adr + 7] = 0.0

        mujoco.mj_forward(self.model, self.data)

        # Reset state
        self._phase = PHASE_STAND
        self._phase_step = 0
        self._step_count = 0
        self._prev_action = np.zeros(NUM_ACTUATED, dtype=np.float32)

        obs = self._get_obs()
        info = {
            "phase": PHASE_NAMES[self._phase],
            "pelvis_height": self._get_pelvis_height(),
        }
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).flatten()
        if action.size < NUM_ACTUATED:
            action = np.pad(action, (0, NUM_ACTUATED - action.size))
        action = action[:NUM_ACTUATED]

        # Convert normalized action to joint targets
        targets = self._action_to_ctrl(action)

        # Set actuator controls
        ctrl = np.zeros(self.model.nu)
        ctrl[self._act_idx] = targets
        self.data.ctrl[:] = ctrl

        # Clamp velocities before stepping to prevent explosion
        np.clip(self.data.qvel, -self._max_qvel, self._max_qvel,
                out=self.data.qvel)

        # Step simulation
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
            # Early break on NaN/Inf to avoid propagation
            if self._has_fatal_instability():
                break
            # Clamp velocities each sub-step
            np.clip(self.data.qvel, -self._max_qvel, self._max_qvel,
                    out=self.data.qvel)

        mujoco.mj_forward(self.model, self.data)

        self._step_count += 1
        self._phase_step += 1

        # Check for simulation instability (NaN/Inf)
        if self._has_fatal_instability():
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, -10.0, True, False, {
                "phase": PHASE_NAMES[self._phase],
                "unstable": True,
                "contact_count": int(self.data.ncon),
                "qacc_norm": self._qacc_norm(),
            }

        # Update phase
        self._update_phase()

        # Compute reward
        reward, reward_info = self._compute_reward(action)

        # Check termination
        terminated = self._check_termination()
        truncated = self._step_count >= self._max_episode_steps

        # Get observation
        obs = self._get_obs()
        # Clamp observation to avoid NaN propagation to network
        obs = np.clip(obs, -10.0, 10.0)

        info = {
            "phase": PHASE_NAMES[self._phase],
            "phase_step": self._phase_step,
            "step": self._step_count,
            "pelvis_height": self._get_pelvis_height(),
            "torso_up": self._torso_up(),
            "contact_count": int(self.data.ncon),
            "qpos_norm": float(np.linalg.norm(self.data.qpos)),
            "qvel_norm": float(np.linalg.norm(self.data.qvel)),
            "qacc_norm": self._qacc_norm(),
            **reward_info,
        }

        self._prev_action = action.copy()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ---- Phase state machine ----

    def _update_phase(self):
        """Transition between task phases."""
        if self._phase == PHASE_STAND:
            if self._phase_step >= self._stand_steps:
                self._phase = PHASE_SQUAT
                self._phase_step = 0

        elif self._phase == PHASE_SQUAT:
            pelvis_h = self._get_pelvis_height()
            if pelvis_h < self._squat_target_h + 0.05:
                self._phase = PHASE_REACH
                self._phase_step = 0

        elif self._phase == PHASE_REACH:
            l_dist, r_dist = self._hand_object_distances()
            avg_dist = 0.5 * (l_dist + r_dist)
            if avg_dist < self._reach_threshold:
                self._phase = PHASE_GRASP
                self._phase_step = 0

        elif self._phase == PHASE_GRASP:
            if self._phase_step >= self._grasp_hold_steps:
                if self._is_object_grasped():
                    self._phase = PHASE_LIFT
                    self._phase_step = 0

        elif self._phase == PHASE_LIFT:
            if self._phase_step >= self._lift_hold_steps:
                self._phase = PHASE_STANDUP
                self._phase_step = 0

        elif self._phase == PHASE_STANDUP:
            pass  # Terminal phase – keep going

    # ---- Reward computation ----

    def _compute_reward(self, action: np.ndarray):
        """Compute shaped reward based on current phase."""
        info = {}
        reward = 0.0

        # --- Alive bonus (must dominate penalties so standing > falling) ---
        r_alive = 5.0
        reward += r_alive
        info["alive_reward"] = r_alive

        # --- Balance reward ---
        posture_reward = self._reward_balance()
        reward += 3.0 * posture_reward
        info["posture_reward"] = posture_reward
        info["pelvis_height_reward"] = self._reward_stand()
        info["hand_object_distance_reward"] = self._reward_reach()
        info["grasp_contact_reward"] = self._reward_grasp()
        info["lift_reward"] = self._reward_lift()

        # --- Phase-specific rewards ---
        r_phase = 0.0

        if self._phase == PHASE_STAND:
            # Reward for maintaining stable standing posture
            r_phase = info["pelvis_height_reward"]

        elif self._phase == PHASE_SQUAT:
            r_phase = self._reward_squat()

        elif self._phase == PHASE_REACH:
            r_phase = info["hand_object_distance_reward"]

        elif self._phase == PHASE_GRASP:
            r_phase = info["grasp_contact_reward"]

        elif self._phase == PHASE_LIFT:
            r_phase = info["lift_reward"]

        elif self._phase == PHASE_STANDUP:
            r_phase = self._reward_standup()

        reward += 5.0 * r_phase
        info["phase_reward"] = r_phase

        # --- Penalties ---

        # Energy penalty: penalize large ACTIONS (not forces!) to keep scale bounded
        # action \in [-1,1], so max penalty = 0.005 * 26 = 0.13 per step (negligible)
        r_energy = -0.005 * np.sum(action ** 2)
        reward += r_energy
        info["energy_penalty"] = r_energy

        # Action rate penalty (smoothness)
        action_diff = action - self._prev_action
        r_smooth = -0.01 * np.sum(action_diff ** 2)
        reward += r_smooth
        info["action_smoothness_penalty"] = r_smooth

        # Joint limit penalty
        joint_pos = self.data.qpos[self._joint_qpos_idx]
        jrange = self._joint_range
        margin = 0.05
        lower_viol = np.maximum(jrange[:, 0] + margin - joint_pos, 0.0)
        upper_viol = np.maximum(joint_pos - jrange[:, 1] + margin, 0.0)
        r_jlimit = -0.5 * np.sum(lower_viol ** 2 + upper_viol ** 2)
        reward += r_jlimit
        info["joint_limit_penalty"] = r_jlimit
        info["fall_penalty"] = -10.0 if self._check_termination() else 0.0

        info["reward_total"] = reward
        return reward, info

    def _reward_balance(self) -> float:
        """Reward for upright pelvis orientation and stable CoM."""
        # Pelvis orientation: penalize tilt from upright
        up_z = self._torso_up()
        orientation_reward = np.clip(up_z, 0.0, 1.0)

        # CoM over support polygon (project CoM to ground and compare with feet)
        com = self._get_com()
        l_foot_pos = self.data.xpos[self._l_foot_id]
        r_foot_pos = self.data.xpos[self._r_foot_id]
        support_center = 0.5 * (l_foot_pos[:2] + r_foot_pos[:2])
        com_offset = np.linalg.norm(com[:2] - support_center)
        com_reward = np.exp(-5.0 * com_offset)

        # Penalize large angular velocity
        angvel = self.data.qvel[self._root_qvel_adr + 3 : self._root_qvel_adr + 6]
        angvel_penalty = np.exp(-0.5 * np.sum(angvel ** 2))

        return 0.4 * orientation_reward + 0.4 * com_reward + 0.2 * angvel_penalty

    def _reward_stand(self) -> float:
        """Reward for stable standing at initial height."""
        h = self._get_pelvis_height()
        target = self._standup_target_h
        return np.exp(-10.0 * (h - target) ** 2)

    def _reward_squat(self) -> float:
        """Reward for lowering pelvis to squat height."""
        h = self._get_pelvis_height()
        target = self._squat_target_h
        # Reward getting closer to squat height
        dist_to_target = abs(h - target)
        return np.exp(-10.0 * dist_to_target)

    def _reward_reach(self) -> float:
        """Reward for moving hands toward the object."""
        l_dist, r_dist = self._hand_object_distances()
        avg_dist = 0.5 * (l_dist + r_dist)
        return np.exp(-5.0 * avg_dist)

    def _reward_grasp(self) -> float:
        """Reward for maintaining contact between hands and object."""
        grasped = self._is_object_grasped()
        l_dist, r_dist = self._hand_object_distances()
        proximity = np.exp(-10.0 * (l_dist + r_dist))

        if grasped:
            return 1.0
        else:
            return 0.3 * proximity

    def _reward_standup(self) -> float:
        """Reward for standing up while holding the object."""
        h = self._get_pelvis_height()
        target = self._standup_target_h

        # Height reward
        h_reward = np.exp(-10.0 * (h - target) ** 2)

        # Keep holding the object
        grasped = self._is_object_grasped()
        grasp_reward = 1.0 if grasped else 0.0

        # Object height should rise with the robot
        obj_h = 0.0
        if self._obj_qpos_adr >= 0:
            obj_h = self.data.qpos[self._obj_qpos_adr + 2]
        obj_reward = np.clip(obj_h / 0.5, 0.0, 1.0)

        return 0.3 * h_reward + 0.4 * grasp_reward + 0.3 * obj_reward

    def _reward_lift(self) -> float:
        """Reward object lift progress after grasp."""
        if self._obj_qpos_adr < 0:
            return 0.0
        obj_h = self.data.qpos[self._obj_qpos_adr + 2]
        return float(np.clip((obj_h - 0.04) / 0.25, 0.0, 1.0))

    # ---- Helper methods ----

    def _get_pelvis_height(self) -> float:
        return self.data.qpos[self._root_qpos_adr + 2]

    def _torso_up(self) -> float:
        pelvis_quat = self.data.qpos[
            self._root_qpos_adr + 3 : self._root_qpos_adr + 7
        ]
        _, x, y, _ = pelvis_quat
        return float(1.0 - 2.0 * (x * x + y * y))

    def _qacc_norm(self) -> float:
        if self.data.qacc.size == 0:
            return 0.0
        return float(np.linalg.norm(self.data.qacc))

    def _has_fatal_instability(self) -> bool:
        if not np.isfinite(self.data.qpos).all():
            return True
        if not np.isfinite(self.data.qvel).all():
            return True
        if not np.isfinite(self.data.qacc).all():
            return True
        return self._qacc_norm() > self._fatal_qacc_norm

    def _hand_object_distances(self):
        """Return (left_hand_dist, right_hand_dist) to object center."""
        obj_pos = self.data.xpos[self._object_id]
        l_pos = self.data.xpos[self._l_hand_id]
        r_pos = self.data.xpos[self._r_hand_id]
        return np.linalg.norm(l_pos - obj_pos), np.linalg.norm(r_pos - obj_pos)

    def _is_object_grasped(self) -> bool:
        """Check if both hands are in contact with the object."""
        l_contact = False
        r_contact = False

        obj_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom"
        )

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2

            # Check if this contact involves the object
            if g1 == obj_geom_id or g2 == obj_geom_id:
                other = g2 if g1 == obj_geom_id else g1
                # Check which body this geom belongs to
                body_id = self.model.geom_bodyid[other]

                # Walk up the body tree to see if it's part of left or right hand
                bid = body_id
                while bid > 0:
                    if bid == self._l_hand_id:
                        l_contact = True
                        break
                    if bid == self._r_hand_id:
                        r_contact = True
                        break
                    bid = self.model.body_parentid[bid]

        return l_contact and r_contact

    def _check_termination(self) -> bool:
        """Check if the episode should terminate (robot fell)."""
        h = self._get_pelvis_height()

        # Fell too low (catch early before physics explode)
        if h < 0.35:
            return True

        # Excessive tilt
        pelvis_quat = self.data.qpos[
            self._root_qpos_adr + 3 : self._root_qpos_adr + 7
        ]
        w, x, y, z = pelvis_quat
        up_z = 1.0 - 2.0 * (x * x + y * y)
        if up_z < 0.5:  # ~60 degree tilt
            return True

        return False

    # ---- Rendering ----

    def _init_renderer(self):
        """Initialize MuJoCo renderer for visualization."""
        try:
            self._renderer = mujoco.Renderer(self.model, height=720, width=1280)
        except Exception:
            self._renderer = None

    def render(self):
        if self._renderer is None:
            return None

        self._renderer.update_scene(self.data, camera="track_cam")
        pixels = self._renderer.render()

        if self.render_mode == "rgb_array":
            return pixels
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
