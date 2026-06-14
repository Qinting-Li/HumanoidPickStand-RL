from pathlib import Path

import mujoco


PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_scene_xml_loads():
    model = mujoco.MjModel.from_xml_path(str(PROJECT_DIR / "assets" / "scene.xml"))
    assert model.nbody >= 60
    assert model.nu == 26
    assert model.nsensor >= 3

