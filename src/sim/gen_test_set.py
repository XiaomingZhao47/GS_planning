"""Generate a held-out test set for benchmark evaluation.

Samples N navigable poses from a habitat scene, renders RGB-D + saves poses
in the same on-disk layout that habitat_runner.run_sortie produces, but
written to ``<out>/test/`` so the benchmark driver picks it up.

The poses are sampled via habitat's pathfinder so they all lie in free space
(no embedding inside walls). Yaws are uniform.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def main(scene: str, out_dir: Path, n: int, hfov_deg: float = 90.0,
         width: int = 640, height: int = 480, seed: int = 0):
    import habitat_sim
    import quaternion
    import imageio.v2 as imageio
    from src.sim.habitat_runner import CameraIntrinsics, HABITAT_TO_OPENCV

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)
    (out_dir / "depth").mkdir(exist_ok=True)

    intr = CameraIntrinsics(width, height, hfov_deg)

    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = scene
    cfg.enable_physics = False

    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"; rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [height, width]; rgb.hfov = hfov_deg

    depth = habitat_sim.CameraSensorSpec()
    depth.uuid = "depth"; depth.sensor_type = habitat_sim.SensorType.DEPTH
    depth.resolution = [height, width]; depth.hfov = hfov_deg

    ag = habitat_sim.agent.AgentConfiguration()
    ag.sensor_specifications = [rgb, depth]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [ag]))
    agent = sim.initialize_agent(0)

    rng = np.random.default_rng(seed)
    poses_cv = []
    try:
        for i in range(n):
            # Sample a navigable point on the navmesh
            for _ in range(50):
                p = sim.pathfinder.get_random_navigable_point()
                if np.all(np.isfinite(p)):
                    break
            # Lift to camera height (~0.6 m above floor)
            p = np.array(p, dtype=np.float64)
            yaw = float(rng.uniform(0, 2 * math.pi))
            # Build OpenGL agent rotation: yaw around world Y axis
            qy = math.cos(yaw / 2)
            qw_ = math.sin(yaw / 2)
            q = quaternion.quaternion(qy, 0.0, qw_, 0.0)
            state = habitat_sim.AgentState()
            state.position = p
            state.rotation = q
            agent.set_state(state, reset_sensors=True)
            obs = sim.get_sensor_observations()

            T_hab = np.eye(4)
            T_hab[:3, :3] = quaternion.as_rotation_matrix(q)
            T_hab[:3, 3] = p
            wfc_cv = T_hab @ HABITAT_TO_OPENCV
            poses_cv.append(wfc_cv)

            imageio.imwrite(out_dir / "images" / f"rgb_{i:06d}.png", obs["rgb"][..., :3])
            np.save(out_dir / "depth" / f"depth_{i:06d}.npy",
                    obs["depth"].astype(np.float32))
    finally:
        sim.close()

    poses_cv = np.stack(poses_cv, axis=0)
    np.save(out_dir / "poses.npy", poses_cv)
    (out_dir / "intrinsics.json").write_text(json.dumps({
        "width": width, "height": height, "hfov_deg": hfov_deg,
        "fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy,
    }, indent=2))
    print(f"wrote {n} test poses to {out_dir}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/xiaoming/GS_planning")
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    main_args = p.parse_args()
    main(main_args.scene, Path(main_args.out), main_args.n, seed=main_args.seed)
