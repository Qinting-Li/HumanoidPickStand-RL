"""Run simulator diagnostics and write reproducible logs."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import envs  # noqa: F401
from controllers.pd_stand_controller import PDStandController


def _finite_env(env) -> bool:
    base = env.unwrapped
    return (
        np.isfinite(base.data.qpos).all()
        and np.isfinite(base.data.qvel).all()
        and np.isfinite(base.data.qacc).all()
    )


def _rollout(name: str, policy, steps: int, out_dir: Path) -> dict:
    env = gym.make(
        "HumanoidSquatPick-v0",
        scene_xml=str(PROJECT_DIR / "assets" / "scene.xml"),
        max_episode_steps=steps,
    )
    obs, info = env.reset(seed=7)
    rows = []
    fatal = False
    terminated = False
    truncated = False

    for step in range(steps):
        action = policy(env, obs, info, step)
        obs, reward, terminated, truncated, info = env.step(action)
        base = env.unwrapped
        finite = _finite_env(env) and np.isfinite(obs).all() and np.isfinite(reward)
        qacc_norm = float(np.linalg.norm(base.data.qacc))
        fatal = fatal or (not finite) or bool(info.get("unstable", False))
        rows.append(
            {
                "step": step,
                "reward": float(reward),
                "phase": info.get("phase", ""),
                "pelvis_height": float(info.get("pelvis_height", 0.0)),
                "torso_up": float(info.get("torso_up", 0.0)),
                "contact_count": int(info.get("contact_count", 0)),
                "qpos_norm": float(info.get("qpos_norm", 0.0)),
                "qvel_norm": float(info.get("qvel_norm", 0.0)),
                "qacc_norm": qacc_norm,
                "finite": finite,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        if fatal or terminated or truncated:
            break

    out_path = out_dir / f"{name}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    env.close()
    min_height = min(row["pelvis_height"] for row in rows)
    max_qacc = max(row["qacc_norm"] for row in rows)
    return {
        "name": name,
        "steps": len(rows),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "fatal_instability": bool(fatal),
        "min_pelvis_height": min_height,
        "max_qacc_norm": max_qacc,
        "csv": str(out_path),
    }


def main() -> int:
    out_dir = PROJECT_DIR / "diagnostics"
    out_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(123)
    pd = PDStandController()

    summaries = [
        _rollout("zero_action", lambda env, obs, info, step: np.zeros(26), 250, out_dir),
        _rollout(
            "small_random_action",
            lambda env, obs, info, step: rng.uniform(-0.05, 0.05, 26),
            250,
            out_dir,
        ),
        _rollout("pd_standing", lambda env, obs, info, step: pd.predict(env), 500, out_dir),
    ]

    summary = {
        "fatal_instability": any(item["fatal_instability"] for item in summaries),
        "rollouts": summaries,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for item in summaries:
        print(
            f"{item['name']}: steps={item['steps']} "
            f"fatal={item['fatal_instability']} min_h={item['min_pelvis_height']:.3f} "
            f"max_qacc={item['max_qacc_norm']:.1f}"
        )
    print(f"Saved summary: {summary_path}")
    print(f"SUMMARY: {'FAIL' if summary['fatal_instability'] else 'PASS'}")
    return 1 if summary["fatal_instability"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

