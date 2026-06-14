from pathlib import Path

from scripts.validate_assets import ASSETS_DIR, check_file


def test_generated_assets_have_valid_mesh_paths():
    assert check_file(ASSETS_DIR / "humanoid_mujoco.urdf", ASSETS_DIR)[0]
    assert check_file(ASSETS_DIR / "scene.xml", ASSETS_DIR / "meshes")[0]


def test_raw_urdf_reports_exporter_mesh_paths():
    raw = (ASSETS_DIR / "humanoid.urdf").read_text(encoding="utf-8")
    assert "../meshes/" in raw

