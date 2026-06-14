# Model Asset Pipeline

The source model is a SolidWorks-exported URDF with STL meshes. The raw file
`assets/humanoid.urdf` preserves exporter-relative mesh paths such as
`../meshes/pelvis.STL`. In this repository layout those paths are not valid when
the raw URDF is loaded directly from `assets/`.

The benchmark uses generated MuJoCo-facing assets:

1. `scripts/build_scene.py` rewrites `../meshes/` to `meshes/`.
2. It adds a MuJoCo compiler block with `meshdir="meshes/"`.
3. MuJoCo compiles the processed URDF to `assets/humanoid_base.xml`.
4. The script augments that MJCF into `assets/scene.xml` with ground, object,
   actuators, contact pairs, sensors, camera, and solver settings.

Run:

```powershell
python scripts/validate_assets.py
python scripts/validate_mujoco_model.py
```

The raw URDF path issue is reported as a warning. The generated URDF and MJCF
must pass mesh-reference validation.

