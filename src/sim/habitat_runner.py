"""Sortie executor on top of habitat-sim.

Drives a virtual RGB-D camera through a Replica scene along a given list of
SE(3) waypoints and writes per-frame RGB, depth, and pose to disk in a form
the converter in sortie_to_nerf_synthetic.py can turn into Inria 3DGS input.

Conventions:
  - World frame: Habitat's y-up, right-handed (we keep it as-is).
  - Pose format on disk: 4x4 world-from-camera matrix (OpenCV/COLMAP camera
    convention, x-right y-down z-forward). We convert from Habitat (y-up,
    z-back) when saving.
  - One sortie = one output directory. Multi-sortie experiments live as
    sibling directories so the GS trainer can be pointed at any subset.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

# habitat-sim is imported lazily so that data-prep utilities can run without it
def _import_habitat():
    import habitat_sim  # type: ignore
    return habitat_sim


# Habitat (y-up, -z forward) -> OpenCV/COLMAP (y-down, +z forward)
HABITAT_TO_OPENCV = np.array(
    [[1, 0, 0, 0],
     [0, -1, 0, 0],
     [0, 0, -1, 0],
     [0, 0, 0, 1]],
    dtype=np.float64,
)


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    hfov_deg: float

    @property
    def fx(self) -> float:
        return 0.5 * self.width / math.tan(0.5 * math.radians(self.hfov_deg))

    @property
    def fy(self) -> float:
        # square pixels
        return self.fx

    @property
    def cx(self) -> float:
        return 0.5 * self.width

    @property
    def cy(self) -> float:
        return 0.5 * self.height


def _make_sim(scene_path: str, intr: CameraIntrinsics):
    hsim = _import_habitat()
    cfg = hsim.SimulatorConfiguration()
    cfg.scene_id = scene_path
    cfg.enable_physics = False

    rgb = hsim.CameraSensorSpec()
    rgb.uuid = "rgb"
    rgb.sensor_type = hsim.SensorType.COLOR
    rgb.resolution = [intr.height, intr.width]
    rgb.hfov = intr.hfov_deg
    rgb.position = [0.0, 0.0, 0.0]

    depth = hsim.CameraSensorSpec()
    depth.uuid = "depth"
    depth.sensor_type = hsim.SensorType.DEPTH
    depth.resolution = [intr.height, intr.width]
    depth.hfov = intr.hfov_deg
    depth.position = [0.0, 0.0, 0.0]

    agent_cfg = hsim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb, depth]

    return hsim.Simulator(hsim.Configuration(cfg, [agent_cfg]))


def _habitat_pose_to_world_from_cam_opencv(position: np.ndarray, rotation_quat) -> np.ndarray:
    """Build a 4x4 world-from-camera matrix in OpenCV convention from a
    Habitat agent position (xyz) and rotation (quaternion, magnum or numpy).
    """
    import quaternion  # type: ignore  # numpy-quaternion
    if not isinstance(rotation_quat, quaternion.quaternion):
        rotation_quat = quaternion.quaternion(*rotation_quat)
    R = quaternion.as_rotation_matrix(rotation_quat)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(position, dtype=np.float64)
    # Habitat agent frame is y-up, -z forward (OpenGL). Convert camera basis.
    return T @ HABITAT_TO_OPENCV


def run_sortie(
    scene_path: str,
    waypoints: Sequence[np.ndarray],   # list of 4x4 world-from-cam (OpenCV) poses
    out_dir: Path,
    intr: CameraIntrinsics,
) -> dict:
    """Execute one sortie: render an image at each waypoint and dump everything
    to ``out_dir`` in a NeRF-Synthetic-like layout::

        out_dir/
          images/rgb_000000.png ... rgb_NNNNNN.png
          depth/depth_000000.npy ... (float32 metric depth)
          poses.npy             # (N, 4, 4) world-from-camera, OpenCV
          intrinsics.json
    """
    import imageio.v2 as imageio  # type: ignore
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "depth").mkdir(parents=True, exist_ok=True)

    hsim = _import_habitat()
    sim = _make_sim(scene_path, intr)
    agent = sim.initialize_agent(0)

    poses = []
    try:
        for i, wp in enumerate(waypoints):
            # Set agent state from desired OpenCV world-from-cam pose by
            # inverting the camera-basis conversion.
            T_hab = wp @ np.linalg.inv(HABITAT_TO_OPENCV)
            state = hsim.AgentState()
            state.position = T_hab[:3, 3]
            import quaternion  # type: ignore
            state.rotation = quaternion.from_rotation_matrix(T_hab[:3, :3])
            agent.set_state(state, reset_sensors=True)

            obs = sim.get_sensor_observations()
            rgb = obs["rgb"][..., :3]
            depth = obs["depth"].astype(np.float32)

            imageio.imwrite(out_dir / "images" / f"rgb_{i:06d}.png", rgb)
            np.save(out_dir / "depth" / f"depth_{i:06d}.npy", depth)
            poses.append(wp)
    finally:
        sim.close()

    poses = np.stack(poses, axis=0)
    np.save(out_dir / "poses.npy", poses)
    with open(out_dir / "intrinsics.json", "w") as f:
        json.dump({
            "width": intr.width, "height": intr.height,
            "hfov_deg": intr.hfov_deg,
            "fx": intr.fx, "fy": intr.fy,
            "cx": intr.cx, "cy": intr.cy,
        }, f, indent=2)

    return {"n_frames": len(poses), "out_dir": str(out_dir)}


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, help="Path to a Replica scene .glb/.ply")
    p.add_argument("--waypoints", required=True, help=".npy file of (N, 4, 4) poses (OpenCV world-from-cam)")
    p.add_argument("--out", required=True, help="output directory for this sortie")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--hfov", type=float, default=90.0)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    intr = CameraIntrinsics(args.width, args.height, args.hfov)
    waypoints = np.load(args.waypoints)
    assert waypoints.ndim == 3 and waypoints.shape[1:] == (4, 4), waypoints.shape
    info = run_sortie(args.scene, waypoints, Path(args.out), intr)
    print(json.dumps(info, indent=2))
