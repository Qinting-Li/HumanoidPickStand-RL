"""
Build MuJoCo scene XML from humanoid URDF.

Steps:
1. Pre-process URDF for MuJoCo compatibility
2. Compile with MuJoCo and save as MJCF
3. Post-process MJCF to add scene elements (ground, object, actuators, contacts)

Usage:
    python scripts/build_scene.py
"""

import os
import re
import sys
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ASSETS_DIR = os.path.join(PROJECT_DIR, "assets")

ACTUATED_JOINTS = [
    # Left leg (6)
    "hip_roll_l_joint",
    "hip_yaw_l_joint",
    "hip_pitch_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "ankle_roll_l_joint",
    # Right leg (6)
    "hip_roll_r_joint",
    "hip_yaw_r_joint",
    "hip_pitch_r_joint",
    "knee_pitch_r_joint",
    "ankle_pitch_r_joint",
    "ankle_roll_r_joint",
    # Left arm (7)
    "shoulder_pitch_l_joint",
    "shoulder_roll_l_joint",
    "shoulder_yaw_l_joint",
    "elbow_roll_l_joint",
    "elbow_yaw_l_joint",
    "wrist_roll_l_joint",
    "wrist_pitch_l_joint",
    # Right arm (7)
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_roll_r_joint",
    "elbow_yaw_r_joint",
    "wrist_roll_r_joint",
    "wrist_pitch_r_joint",
]

# kp/kv/forcerange per joint group.
# PD-controller: force = kp*(target - q) - kv*qdot, clamped by forcerange.
# kv ≈ 10% of kp gives good damping without over-damping.
JOINT_PARAMS = {
    "hip_roll":       {"kp": 3000, "kv": 300, "force": 500},
    "hip_yaw":        {"kp": 2000, "kv": 200, "force": 300},
    "hip_pitch":      {"kp": 3000, "kv": 300, "force": 500},
    "knee_pitch":     {"kp": 3000, "kv": 300, "force": 500},
    "ankle_pitch":    {"kp": 2000, "kv": 200, "force": 300},
    "ankle_roll":     {"kp": 1500, "kv": 150, "force": 200},
    "shoulder_pitch": {"kp": 600,  "kv": 60,  "force": 100},
    "shoulder_roll":  {"kp": 600,  "kv": 60,  "force": 100},
    "shoulder_yaw":   {"kp": 300,  "kv": 30,  "force": 60},
    "elbow_roll":     {"kp": 300,  "kv": 30,  "force": 60},
    "elbow_yaw":      {"kp": 150,  "kv": 15,  "force": 30},
    "wrist_roll":     {"kp": 150,  "kv": 15,  "force": 30},
    "wrist_pitch":    {"kp": 150,  "kv": 15,  "force": 30},
}


def _get_joint_group(joint_name: str) -> str:
    """Extract the joint group key from a joint name like 'hip_roll_l_joint'."""
    # Remove _l_joint / _r_joint suffix
    name = joint_name.replace("_l_joint", "").replace("_r_joint", "")
    return name


# ---------------------------------------------------------------------------
# Step 1 : Pre-process URDF
# ---------------------------------------------------------------------------
def preprocess_urdf() -> str:
    """Fix URDF for MuJoCo compatibility and return path to modified file."""
    src = os.path.join(ASSETS_DIR, "humanoid.urdf")
    dst = os.path.join(ASSETS_DIR, "humanoid_mujoco.urdf")

    with open(src, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Fix mesh paths: ../meshes/X -> meshes/X
    content = content.replace("../meshes/", "meshes/")

    # 2. Remove non-standard <property .../> elements
    content = re.sub(r"\s*<property\s+[^/]*/>\s*", "\n", content)

    # 3. Remove non-standard 'start_stop' attribute from <limit>
    content = re.sub(r'\s+start_stop="[^"]*"', "", content)

    # 4. Add MuJoCo compiler extension inside <robot> (if not present)
    if "<mujoco>" not in content:
        ext = (
            '\n  <mujoco>\n'
            '    <compiler meshdir="meshes/" balanceinertia="true" discardvisual="false" fusestatic="false"/>\n'
            '  </mujoco>\n'
        )
        content = content.replace("</robot>", ext + "</robot>")

    with open(dst, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"  Pre-processed URDF saved to {dst}")
    return dst


# ---------------------------------------------------------------------------
# Step 2 : Compile URDF -> MJCF via MuJoCo
# ---------------------------------------------------------------------------
def compile_urdf_to_mjcf(urdf_path: str) -> str:
    """Compile the URDF with MuJoCo and save the raw MJCF."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(urdf_path)
    mjcf_path = os.path.join(ASSETS_DIR, "humanoid_base.xml")
    mujoco.mj_saveLastXML(mjcf_path, model)
    print(f"  Base MJCF saved to {mjcf_path}")

    # Verify joint count
    joint_names = []
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            joint_names.append(name)
    print(f"  Found {model.njnt} joints: {joint_names[:10]}...")

    return mjcf_path


# ---------------------------------------------------------------------------
# Step 3 : Post-process – add scene elements
# ---------------------------------------------------------------------------
def _find_body(parent: ET.Element, name: str):
    """Recursively find a <body> element with the given name."""
    for body in parent.iter("body"):
        if body.get("name") == name:
            return body
    return None


def _collect_geom_names(body: ET.Element):
    """Return a list of geom names directly under *body*."""
    return [g.get("name") for g in body.findall("geom") if g.get("name")]


def postprocess_mjcf(mjcf_path: str) -> str:
    """Augment the raw MJCF with scene elements and save as scene.xml."""
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    # ---- compiler --------------------------------------------------------
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("angle", "radian")
    compiler.set("autolimits", "true")

    # ---- option (physics) ------------------------------------------------
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", "0.002")
    # Conservative solver settings for contact-rich humanoid diagnostics.
    # More Newton iterations are preferable to hiding instability with very high
    # damping or friction.
    option.set("iterations", "150")
    option.set("solver", "Newton")
    option.set("gravity", "0 0 -9.81")
    option.set("cone", "pyramidal")
    option.set("impratio", "10")
    option.set("noslip_iterations", "5")

    flag = option.find("flag")
    if flag is None:
        flag = ET.SubElement(option, "flag")
    flag.set("multiccd", "enable")

    # ---- size ------------------------------------------------------------
    size_elem = root.find("size")
    if size_elem is None:
        size_elem = ET.SubElement(root, "size")
    size_elem.set("nconmax", "500")
    size_elem.set("njmax", "2000")

    # ---- visual ----------------------------------------------------------
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    headlight.set("ambient", "0.4 0.4 0.4")
    headlight.set("diffuse", "0.8 0.8 0.8")
    glob = visual.find("global")
    if glob is None:
        glob = ET.SubElement(visual, "global")
    glob.set("offwidth", "1920")
    glob.set("offheight", "1080")

    # ---- default ---------------------------------------------------------
    default = root.find("default")
    if default is None:
        default = ET.SubElement(root, "default")
    joint_default = ET.SubElement(default, "joint")
    # Modest joint damping and armature reduce CAD-import chatter while keeping
    # the model responsive enough for controllers and RL policies.
    joint_default.set("damping", "3")
    joint_default.set("armature", "0.15")
    joint_default.set("frictionloss", "0.05")
    geom_default = ET.SubElement(default, "geom")
    geom_default.set("condim", "4")
    geom_default.set("friction", "1.0 0.005 0.0001")

    # ---- worldbody additions ---------------------------------------------
    worldbody = root.find("worldbody")

    # Ground plane
    ground = ET.SubElement(worldbody, "geom")
    ground.set("name", "ground")
    ground.set("type", "plane")
    ground.set("size", "10 10 0.1")
    ground.set("rgba", "0.8 0.9 0.8 1")
    ground.set("friction", "1.0 0.005 0.0001")
    ground.set("contype", "1")
    ground.set("conaffinity", "1")

    # Lights
    for lname, lpos, ldir in [
        ("light_top", "0 0 4", "0 0 -1"),
        ("light_front", "2 -2 3", "-1 1 -1"),
    ]:
        light = ET.SubElement(worldbody, "light")
        light.set("name", lname)
        light.set("pos", lpos)
        light.set("dir", ldir)
        light.set("diffuse", "0.7 0.7 0.7")
        light.set("specular", "0.3 0.3 0.3")

    # Camera
    cam = ET.SubElement(worldbody, "camera")
    cam.set("name", "track_cam")
    cam.set("pos", "2.0 -1.5 1.5")
    cam.set("xyaxes", "0.6 0.8 0 -0.3 0.2 0.9")

    # Pick-up object
    obj_body = ET.SubElement(worldbody, "body")
    obj_body.set("name", "pick_object")
    obj_body.set("pos", "0.35 0.0 0.04")
    fj = ET.SubElement(obj_body, "freejoint")
    fj.set("name", "object_free")
    obj_geom = ET.SubElement(obj_body, "geom")
    obj_geom.set("name", "object_geom")
    obj_geom.set("type", "box")
    obj_geom.set("size", "0.04 0.04 0.04")
    obj_geom.set("rgba", "0.9 0.2 0.2 1")
    obj_geom.set("mass", "0.5")
    obj_geom.set("friction", "1.0 0.005 0.0001")
    obj_geom.set("condim", "4")
    obj_geom.set("contype", "1")
    obj_geom.set("conaffinity", "1")
    obj_site = ET.SubElement(obj_body, "site")
    obj_site.set("name", "object_site")
    obj_site.set("size", "0.005")

    # ---- Modify root body (pelvis) – add freejoint + initial height ------
    pelvis = _find_body(worldbody, "pelvis")
    if pelvis is None:
        # MuJoCo URDF converter might wrap bodies; search deeper
        for b in worldbody.iter("body"):
            pelvis = b
            break  # first body is the root
    if pelvis is not None:
        # Insert freejoint as first child
        existing = list(pelvis)
        freejoint = ET.Element("freejoint")
        freejoint.set("name", "root")
        pelvis.insert(0, freejoint)
        # Set initial height so robot stands on ground
        cur_pos = pelvis.get("pos", "0 0 0")
        parts = cur_pos.split()
        parts[2] = "0.88"  # ~leg length + foot
        pelvis.set("pos", " ".join(parts))
        print(f"  Added freejoint to body '{pelvis.get('name')}'")
    else:
        print("  WARNING: Could not find root body to add freejoint!")

    # ---- Add sites on hand links for contact sensing ---------------------
    for hand_name, site_name in [
        ("L_hand_base_link", "l_palm_site"),
        ("R_hand_base_link", "r_palm_site"),
    ]:
        hand = _find_body(worldbody, hand_name)
        if hand is not None:
            site = ET.SubElement(hand, "site")
            site.set("name", site_name)
            site.set("pos", "0 -0.07 0")
            site.set("size", "0.03")
            site.set("rgba", "0 1 0 0.3")
            print(f"  Added site '{site_name}' on '{hand_name}'")

    # ---- Actuators -------------------------------------------------------
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")

    for jname in ACTUATED_JOINTS:
        group = _get_joint_group(jname)
        params = JOINT_PARAMS.get(group, {"kp": 100, "kv": 10, "force": 40})
        kp = params["kp"]
        kv = params["kv"]
        fmax = params["force"]

        # Use <general> actuator for PD control: force = kp*(ctrl-q) - kv*qdot
        act = ET.SubElement(actuator, "general")
        act.set("name", f"act_{jname}")
        act.set("joint", jname)
        act.set("gainprm", str(kp))
        act.set("biasprm", f"0 {-kp} {-kv}")
        act.set("biastype", "affine")
        act.set("gaintype", "fixed")
        act.set("forcerange", f"{-fmax} {fmax}")
        act.set("ctrlrange", "-3.14 3.14")
        act.set("ctrllimited", "true")

    print(f"  Added {len(ACTUATED_JOINTS)} position actuators")

    # ---- Contact pairs (high friction for palm-object) -------------------
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")

    # Find all geoms belonging to hand bodies and finger bodies.
    # MuJoCo's URDF converter creates unnamed geoms, so we name them first.
    hand_body_names = [
        "L_hand_base_link", "R_hand_base_link",
        "L_thumb_proximal_base", "L_thumb_proximal", "L_thumb_intermediate", "L_thumb_distal",
        "L_index_proximal", "L_index_intermediate",
        "L_middle_proximal", "L_middle_intermediate",
        "L_ring_proximal", "L_ring_intermediate",
        "L_pinky_proximal", "L_pinky_intermediate",
        "R_thumb_proximal_base", "R_thumb_proximal", "R_thumb_intermediate", "R_thumb_distal",
        "R_index_proximal", "R_index_intermediate",
        "R_middle_proximal", "R_middle_intermediate",
        "R_ring_proximal", "R_ring_intermediate",
        "R_pinky_proximal", "R_pinky_intermediate",
    ]
    hand_geoms = []
    geom_counter = 0
    for bname in hand_body_names:
        body = _find_body(worldbody, bname)
        if body is not None:
            for g in body.findall("geom"):
                gname = g.get("name")
                if not gname:
                    gname = f"hand_geom_{geom_counter}"
                    g.set("name", gname)
                    geom_counter += 1
                hand_geoms.append(gname)
                # Also set high friction directly on these geoms
                g.set("friction", "5.0 0.01 0.001")

    for gname in hand_geoms:
        pair = ET.SubElement(contact, "pair")
        pair.set("geom1", gname)
        pair.set("geom2", "object_geom")
        pair.set("friction", "5 5 0.005 0.0001 0.0001")
        pair.set("condim", "4")

    print(f"  Added {len(hand_geoms)} high-friction contact pairs")

    # ---- Sensors ---------------------------------------------------------
    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(root, "sensor")

    # Pelvis IMU
    if pelvis is not None:
        pname = pelvis.get("name", "pelvis")
        # Add a site for the IMU if not there
        imu_site = None
        for s in pelvis.findall("site"):
            if s.get("name") == "imu_site":
                imu_site = s
                break
        if imu_site is None:
            imu_site = ET.SubElement(pelvis, "site")
            imu_site.set("name", "imu_site")
            imu_site.set("pos", "0 0 0")
            imu_site.set("size", "0.01")

        acc = ET.SubElement(sensor, "accelerometer")
        acc.set("name", "pelvis_accel")
        acc.set("site", "imu_site")
        gyro = ET.SubElement(sensor, "gyro")
        gyro.set("name", "pelvis_gyro")
        gyro.set("site", "imu_site")
        framequat = ET.SubElement(sensor, "framequat")
        framequat.set("name", "pelvis_quat")
        framequat.set("objtype", "site")
        framequat.set("objname", "imu_site")

    # ---- Save final scene ------------------------------------------------
    scene_path = os.path.join(ASSETS_DIR, "scene.xml")

    # Pretty-print with indentation
    ET.indent(tree, space="  ")
    tree.write(scene_path, encoding="unicode", xml_declaration=True)
    print(f"  Scene saved to {scene_path}")
    return scene_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Building MuJoCo scene from humanoid URDF")
    print("=" * 60)

    print("\nStep 1: Pre-processing URDF...")
    urdf_path = preprocess_urdf()

    print("\nStep 2: Compiling URDF -> MJCF with MuJoCo...")
    try:
        mjcf_path = compile_urdf_to_mjcf(urdf_path)
    except Exception as e:
        print(f"\n  ERROR compiling URDF: {e}")
        print("  Make sure mujoco is installed: pip install mujoco")
        sys.exit(1)

    print("\nStep 3: Post-processing MJCF (adding scene elements)...")
    scene_path = postprocess_mjcf(mjcf_path)

    print("\n" + "=" * 60)
    print(f"Done! Scene XML ready at: {scene_path}")
    print("=" * 60)

    # Validate the final scene
    print("\nValidating scene.xml...")
    try:
        import mujoco
        model = mujoco.MjModel.from_xml_path(scene_path)
        print(f"  Bodies:    {model.nbody}")
        print(f"  Joints:    {model.njnt}")
        print(f"  Actuators: {model.nu}")
        print(f"  Geoms:     {model.ngeom}")
        print("  Validation passed!")
    except Exception as e:
        print(f"  Validation WARNING: {e}")
        print("  The scene may need manual fixes.")


if __name__ == "__main__":
    main()
