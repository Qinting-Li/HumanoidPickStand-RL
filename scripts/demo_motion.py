"""
Scripted demo: visualize the humanoid performing stand -> squat -> reach -> grasp -> stand-up.
Uses keyframe interpolation with PD control to show the full motion sequence.
Renders to video file (MP4) and optionally launches MuJoCo interactive viewer.

Usage:
    python scripts/demo_motion.py              # render video
    python scripts/demo_motion.py --viewer     # launch interactive MuJoCo viewer
"""

import os
import sys
import argparse
import numpy as np

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_DIR)

import mujoco

# ── Joint name order (must match scene.xml actuator order) ──────────────────
JOINT_NAMES = [
    # Left leg (0-5)
    "hip_roll_l_joint", "hip_yaw_l_joint", "hip_pitch_l_joint",
    "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
    # Right leg (6-11)
    "hip_roll_r_joint", "hip_yaw_r_joint", "hip_pitch_r_joint",
    "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
    # Left arm (12-18)
    "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
    "elbow_roll_l_joint", "elbow_yaw_l_joint",
    "wrist_roll_l_joint", "wrist_pitch_l_joint",
    # Right arm (19-25)
    "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint",
    "elbow_roll_r_joint", "elbow_yaw_r_joint",
    "wrist_roll_r_joint", "wrist_pitch_r_joint",
]
NJ = len(JOINT_NAMES)  # 26

# ── Keyframe poses ──────────────────────────────────────────────────────────
# Each pose is a dict mapping a short key to a value (radians).
# Joints not listed default to 0 (the URDF home / standing pose).

def _pose(
    # legs
    hip_pitch_l=0, knee_l=0, ankle_pitch_l=0,
    hip_pitch_r=0, knee_r=0, ankle_pitch_r=0,
    hip_roll_l=0, hip_roll_r=0,
    # arms
    sh_pitch_l=0, sh_roll_l=0, sh_yaw_l=0, elb_roll_l=0, elb_yaw_l=0, wr_roll_l=0, wr_pitch_l=0,
    sh_pitch_r=0, sh_roll_r=0, sh_yaw_r=0, elb_roll_r=0, elb_yaw_r=0, wr_roll_r=0, wr_pitch_r=0,
):
    """Build a 26-dim numpy array from friendly keyword arguments."""
    q = np.zeros(NJ)
    # Left leg
    q[0] = hip_roll_l
    q[2] = hip_pitch_l
    q[3] = knee_l
    q[4] = ankle_pitch_l
    # Right leg
    q[6] = hip_roll_r
    q[8] = hip_pitch_r
    q[9] = knee_r
    q[10] = ankle_pitch_r
    # Left arm
    q[12] = sh_pitch_l;  q[13] = sh_roll_l;  q[14] = sh_yaw_l
    q[15] = elb_roll_l;  q[16] = elb_yaw_l
    q[17] = wr_roll_l;   q[18] = wr_pitch_l
    # Right arm
    q[19] = sh_pitch_r;  q[20] = sh_roll_r;  q[21] = sh_yaw_r
    q[22] = elb_roll_r;  q[23] = elb_yaw_r
    q[24] = wr_roll_r;   q[25] = wr_pitch_r
    return q


# Phase 0 – STAND (hold home pose, arms slightly at sides)
POSE_STAND = _pose(
    sh_pitch_l=-0.3, sh_roll_l=0.15,
    sh_pitch_r=0.3,  sh_roll_r=-0.15,
)

# Phase 1 – SQUAT (bend knees & hips, keep torso upright, arms forward)
POSE_SQUAT = _pose(
    hip_pitch_l=-1.05, knee_l=2.1, ankle_pitch_l=-1.05,
    hip_pitch_r=-1.05, knee_r=2.1, ankle_pitch_r=-1.05,
    sh_pitch_l=-1.2,  sh_roll_l=0.3, elb_roll_l=-0.8,
    sh_pitch_r=1.2,   sh_roll_r=-0.3, elb_roll_r=0.8,
)

# Phase 2 – REACH (deep squat + arms reaching forward/down toward object)
POSE_REACH = _pose(
    hip_pitch_l=-1.25, knee_l=2.3, ankle_pitch_l=-1.1,
    hip_pitch_r=-1.25, knee_r=2.3, ankle_pitch_r=-1.1,
    sh_pitch_l=-1.8,  sh_roll_l=0.2, sh_yaw_l=0.3, elb_roll_l=-1.2, elb_yaw_l=0.3,
    sh_pitch_r=1.8,   sh_roll_r=-0.2, sh_yaw_r=-0.3, elb_roll_r=1.2, elb_yaw_r=-0.3,
)

# Phase 3 – GRASP (same as reach but wrists curl inward to "grip")
POSE_GRASP = _pose(
    hip_pitch_l=-1.25, knee_l=2.3, ankle_pitch_l=-1.1,
    hip_pitch_r=-1.25, knee_r=2.3, ankle_pitch_r=-1.1,
    sh_pitch_l=-1.8,  sh_roll_l=0.15, sh_yaw_l=0.3, elb_roll_l=-1.2, elb_yaw_l=0.3,
    wr_roll_l=0.3, wr_pitch_l=0.5,
    sh_pitch_r=1.8,   sh_roll_r=-0.15, sh_yaw_r=-0.3, elb_roll_r=1.2, elb_yaw_r=-0.3,
    wr_roll_r=-0.3, wr_pitch_r=0.5,
)

# Phase 4 – STANDUP (return to standing while keeping arms in holding position)
POSE_STANDUP = _pose(
    hip_pitch_l=-0.1, knee_l=0.2, ankle_pitch_l=-0.1,
    hip_pitch_r=-0.1, knee_r=0.2, ankle_pitch_r=-0.1,
    sh_pitch_l=-1.0,  sh_roll_l=0.15, elb_roll_l=-0.9, elb_yaw_l=0.2,
    wr_roll_l=0.3, wr_pitch_l=0.5,
    sh_pitch_r=1.0,   sh_roll_r=-0.15, elb_roll_r=0.9, elb_yaw_r=-0.2,
    wr_roll_r=-0.3, wr_pitch_r=0.5,
)

# Phase 5 – FINAL STAND (stable hold with object)
POSE_FINAL = _pose(
    sh_pitch_l=-0.5,  sh_roll_l=0.15, elb_roll_l=-0.6,
    wr_roll_l=0.2, wr_pitch_l=0.3,
    sh_pitch_r=0.5,   sh_roll_r=-0.15, elb_roll_r=0.6,
    wr_roll_r=-0.2, wr_pitch_r=0.3,
)

# Keyframe schedule: (pose, duration_seconds)
KEYFRAMES = [
    (POSE_STAND,   1.5),   # Hold standing
    (POSE_SQUAT,   2.0),   # Squat down
    (POSE_REACH,   1.5),   # Reach for object
    (POSE_GRASP,   1.0),   # Grasp
    (POSE_STANDUP, 2.5),   # Stand up with object
    (POSE_FINAL,   2.0),   # Stable final pose
]
PHASE_LABELS = ["STAND", "SQUAT", "REACH", "GRASP", "STANDUP", "HOLD"]


def build_index_maps(model):
    """Return arrays for joint qpos addresses and actuator indices."""
    qpos_adr = []
    act_idx = []
    for jname in JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        qpos_adr.append(model.jnt_qposadr[jid])
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{jname}")
        act_idx.append(aid)
    return np.array(qpos_adr), np.array(act_idx)


def run_demo_render(scene_xml: str, output_path: str, width=1280, height=720):
    """Run the scripted demo and render each frame to a video file."""
    model = mujoco.MjModel.from_xml_path(scene_xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    qpos_adr, act_idx = build_index_maps(model)

    renderer = mujoco.Renderer(model, height=height, width=width)

    dt = model.opt.timestep  # 0.002
    ctrl_dt = 0.02           # 50 Hz control
    n_substeps = int(round(ctrl_dt / dt))
    fps = int(round(1.0 / ctrl_dt))

    # Collect frames
    frames = []
    prev_pose = KEYFRAMES[0][0].copy()

    total_time = sum(dur for _, dur in KEYFRAMES)
    print(f"Demo duration: {total_time:.1f}s  ({int(total_time * fps)} frames at {fps} fps)")

    for phase_idx, (target_pose, duration) in enumerate(KEYFRAMES):
        n_steps = int(round(duration / ctrl_dt))
        label = PHASE_LABELS[phase_idx]
        print(f"  Phase {phase_idx}: {label:8s}  ({duration:.1f}s, {n_steps} steps)")

        for step in range(n_steps):
            # Smooth interpolation (cosine ease-in-out)
            t = step / max(n_steps - 1, 1)
            alpha = 0.5 * (1.0 - np.cos(np.pi * t))
            current_target = prev_pose + alpha * (target_pose - prev_pose)

            # Set actuator controls to target positions
            data.ctrl[act_idx] = current_target

            # Step simulation
            for _ in range(n_substeps):
                mujoco.mj_step(model, data)

            # Render
            renderer.update_scene(data, camera="track_cam")
            frame = renderer.render()
            frames.append(frame.copy())

        prev_pose = target_pose.copy()

    renderer.close()

    # Save video
    print(f"\nSaving video to {output_path} ...")
    try:
        import imageio
        imageio.mimsave(output_path, frames, fps=fps, quality=8)
        print(f"  Video saved: {output_path}  ({len(frames)} frames)")
    except ImportError:
        print("  imageio not installed, saving as numpy frames instead...")
        np_path = output_path.replace(".mp4", "_frames.npz")
        np.savez_compressed(np_path, frames=np.array(frames))
        print(f"  Frames saved: {np_path}")

    return frames


def run_demo_viewer(scene_xml: str):
    """Run the scripted demo in MuJoCo's interactive viewer."""
    model = mujoco.MjModel.from_xml_path(scene_xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    qpos_adr, act_idx = build_index_maps(model)

    dt = model.opt.timestep
    ctrl_dt = 0.02
    n_substeps = int(round(ctrl_dt / dt))

    # Build full timeline
    timeline = []
    prev_pose = KEYFRAMES[0][0].copy()
    for phase_idx, (target_pose, duration) in enumerate(KEYFRAMES):
        n_steps = int(round(duration / ctrl_dt))
        for step in range(n_steps):
            t = step / max(n_steps - 1, 1)
            alpha = 0.5 * (1.0 - np.cos(np.pi * t))
            pose = prev_pose + alpha * (target_pose - prev_pose)
            timeline.append(pose)
        prev_pose = target_pose.copy()

    # Add a hold at the end
    for _ in range(100):
        timeline.append(timeline[-1].copy())

    step_idx = [0]

    def controller(model, data):
        idx = min(step_idx[0], len(timeline) - 1)
        data.ctrl[act_idx] = timeline[idx]
        step_idx[0] += 1

    print(f"Launching MuJoCo viewer ({len(timeline)} steps)...")
    print("  Close the viewer window when done.")
    mujoco.viewer.launch(model, data, controller=controller)


def main():
    parser = argparse.ArgumentParser(description="Demo: scripted squat-pick-stand motion")
    parser.add_argument("--viewer", action="store_true", help="Launch interactive MuJoCo viewer")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    scene_xml = os.path.join(PROJECT_DIR, "assets", "scene.xml")
    if not os.path.exists(scene_xml):
        print(f"ERROR: {scene_xml} not found. Run build_scene.py first.")
        sys.exit(1)

    if args.viewer:
        run_demo_viewer(scene_xml)
    else:
        output = args.output or os.path.join(PROJECT_DIR, "demo_squat_pick.mp4")
        run_demo_render(scene_xml, output, args.width, args.height)


if __name__ == "__main__":
    main()
