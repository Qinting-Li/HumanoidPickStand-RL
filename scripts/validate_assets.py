"""Validate URDF/MJCF mesh references for the benchmark assets."""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_DIR / "assets"


def _mesh_refs(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    refs: list[str] = []
    for elem in root.iter():
        if "filename" in elem.attrib:
            refs.append(elem.attrib["filename"])
        if elem.tag == "mesh" and "file" in elem.attrib:
            refs.append(elem.attrib["file"])
    return sorted(set(refs))


def _resolve(ref: str, base_dir: Path) -> Path:
    ref_path = Path(ref)
    if ref_path.is_absolute():
        return ref_path
    return (base_dir / ref_path).resolve()


def check_file(xml_path: Path, base_dir: Path) -> tuple[bool, list[str]]:
    missing = []
    for ref in _mesh_refs(xml_path):
        if not _resolve(ref, base_dir).exists():
            missing.append(ref)
    return len(missing) == 0, missing


def main() -> int:
    raw = ASSETS_DIR / "humanoid.urdf"
    generated_urdf = ASSETS_DIR / "humanoid_mujoco.urdf"
    scene = ASSETS_DIR / "scene.xml"

    print("Humanoid Pick-and-Stand Benchmark asset validation")
    print(f"Project: {PROJECT_DIR}")

    raw_refs = _mesh_refs(raw)
    raw_broken_style = [ref for ref in raw_refs if ref.startswith("../meshes/")]
    raw_ok, raw_missing = check_file(raw, ASSETS_DIR)
    if raw_broken_style:
        print(f"WARN raw URDF keeps exporter paths: {len(raw_broken_style)} '../meshes/...' refs")
    print(f"RAW  {raw.name}: {'OK' if raw_ok else 'WARN'} refs={len(raw_refs)} missing={len(raw_missing)}")

    generated_ok, generated_missing = check_file(generated_urdf, ASSETS_DIR)
    scene_ok, scene_missing = check_file(scene, ASSETS_DIR / "meshes")

    print(
        f"GEN  {generated_urdf.name}: {'PASS' if generated_ok else 'FAIL'} "
        f"missing={len(generated_missing)}"
    )
    print(f"MJCF {scene.name}: {'PASS' if scene_ok else 'FAIL'} missing={len(scene_missing)}")

    if generated_missing:
        print("Missing generated URDF refs:")
        for ref in generated_missing:
            print(f"  - {ref}")
    if scene_missing:
        print("Missing scene mesh refs:")
        for ref in scene_missing:
            print(f"  - {ref}")

    passed = generated_ok and scene_ok
    print(f"SUMMARY: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

