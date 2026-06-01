#!/usr/bin/env python3
"""Path D — Habitat-Sim free-fly capture of the Herkules wreck (GLB).

Samples a multi-altitude orbit around the wreck centroid and dumps frames
in the sortie layout that src/sim/sortie_to_nerf_synthetic.py consumes
(images/rgb_XXXXXX.png, depth/depth_XXXXXX.npy, poses.npy, intrinsics.json).
This is the planning-free / dense-orbit baseline.

Run with: $HOME/miniconda3/envs/habitat/bin/python this_script.py
"""
from __future__ import annotations
import json, math, os
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import habitat_sim
import quaternion as qt
from scipy.spatial.transform import Rotation as Rsc


GLB = "/home/xiaoming/GS_planning_handoff/gazebo_sortie/wreck/herkules.glb"
OUT = Path("/home/xiaoming/GS_planning_handoff/data/wreck_habitat/sortie_00")

W, H = 640, 480
HFOV_DEG = 62.20

# Wreck bbox centroid from trimesh load: min=[-6.58,-11.22,-3.55] max=[5.83,11.40,10.07]
WRECK_CENTER = np.array([-0.37, 0.09, 3.26])
NUM_YAWS = 18           # orbit slices
NUM_HEIGHTS = 3         # altitudes (above / level / below centroid)
RADIUS = 22.0           # orbit radius (m); ~1.7x the bbox half-extent
ALT_OFFSETS = (-6.0, 0.0, 6.0)


def make_sensor(uuid: str, stype) -> habitat_sim.CameraSensorSpec:
    s = habitat_sim.CameraSensorSpec()
    s.uuid = uuid
    s.sensor_type = stype
    s.resolution = [H, W]
    s.position = [0.0, 0.0, 0.0]
    s.hfov = HFOV_DEG
    return s


def look_at_R_opengl(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a 3x3 rotation putting world -> agent (OpenGL: +y up, looks -z).
    Columns of R are the world-frame components of the camera's [x_right, y_up, z_back]."""
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    up = np.array([0.0, 1.0, 0.0])
    z = -fwd                                  # agent +z = backward
    x = np.cross(up, z); x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def opengl_to_opencv_world_from_cam(R_gl: np.ndarray, eye: np.ndarray) -> np.ndarray:
    """OpenGL cam basis (x-right, y-up, z-back) -> OpenCV (x-right, y-down, z-forward).
    Apply diag(1, -1, -1) to the columns of R."""
    R_cv = R_gl.copy()
    R_cv[:, 1] *= -1
    R_cv[:, 2] *= -1
    T = np.eye(4); T[:3, :3] = R_cv; T[:3, 3] = eye
    return T


def main():
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "depth").mkdir(parents=True, exist_ok=True)

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = GLB
    sim_cfg.enable_physics = False
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [
        make_sensor("rgb",   habitat_sim.SensorType.COLOR),
        make_sensor("depth", habitat_sim.SensorType.DEPTH),
    ]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    agent = sim.get_agent(0)
    print("sim ready; scene:", GLB)

    poses = []
    frame_idx = 0
    for h_off in ALT_OFFSETS:
        for k in range(NUM_YAWS):
            ang = 2.0 * math.pi * k / NUM_YAWS
            eye = WRECK_CENTER + np.array([RADIUS * math.cos(ang),
                                           h_off,
                                           RADIUS * math.sin(ang)])
            R = look_at_R_opengl(eye, WRECK_CENTER)
            qxyzw = Rsc.from_matrix(R).as_quat()
            state = habitat_sim.AgentState()
            state.position = eye
            state.rotation = qt.quaternion(qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2])
            agent.set_state(state)
            obs = sim.get_sensor_observations()
            rgb = obs["rgb"][..., :3]
            depth = obs["depth"].astype(np.float32)
            imageio.imwrite(OUT / "images" / f"rgb_{frame_idx:06d}.png", rgb)
            np.save(OUT / "depth" / f"depth_{frame_idx:06d}.npy", depth)
            poses.append(opengl_to_opencv_world_from_cam(R, eye))
            frame_idx += 1
            if frame_idx % 10 == 0:
                print(f"  captured {frame_idx} views")

    poses = np.stack(poses, axis=0)
    np.save(OUT / "poses.npy", poses)
    fx = (W / 2.0) / math.tan(math.radians(HFOV_DEG) / 2.0)
    intr = {"width": W, "height": H,
            "fx": fx, "fy": fx,
            "cx": W / 2.0, "cy": H / 2.0,
            "hfov_deg": HFOV_DEG}
    (OUT / "intrinsics.json").write_text(json.dumps(intr, indent=2))
    print(f"done: {frame_idx} frames -> {OUT}")
    sim.close()


if __name__ == "__main__":
    main()
