#!/usr/bin/env python3
"""DELIBERATE step: build the occupancy grid from sortie-0 perception, then run
the OctoMap+Frontier (Baseline A) and OctoMap+Volumetric (Baseline B) planners
from src/planner/baselines.py to choose sortie-1 waypoints.

Converts each planner's OpenCV world-from-cam poses (y-up: eye=[x,height,z]) to
sortie_executor waypoints {x,y,yaw_deg} in the gazebo/odom frame:
    gazebo_x = eye[0],  gazebo_y = eye[2],  yaw = atan2(R[0,2], R[2,2]).
Writes a YAML per planner. Runs in the gs_planning env.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
import yaml

ROOT = Path("/home/xiaoming/GS_planning_handoff")
sys.path.insert(0, str(ROOT))
from src.planner.occupancy_grid import OccupancyGrid3D            # noqa: E402
from src.planner.baselines import PlannerState, plan_frontier_A, plan_volumetric_B  # noqa: E402


def build_grid(sortie, voxel=0.10, frame_stride=4, stride=6, depth_max=5.0):
    intr = json.loads((sortie / "intrinsics.json").read_text())
    poses = np.load(sortie / "poses.npy")
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    # robust bounds from a coarse unprojection
    pts = []
    for i in range(0, poses.shape[0], frame_stride):
        d = np.load(sortie / "depth" / f"depth_{i:06d}.npy")
        H, W = d.shape
        yy, xx = np.meshgrid(np.arange(0, H, 12), np.arange(0, W, 12), indexing="ij")
        z = d[yy, xx]; m = (z > 0.1) & (z < depth_max) & np.isfinite(z)
        if not m.any():
            continue
        cam = np.stack([(xx[m]-cx)*z[m]/fx, (yy[m]-cy)*z[m]/fy, z[m]], 1)
        pts.append((poses[i][:3, :3] @ cam.T).T + poses[i][:3, 3])
    pts = np.concatenate(pts, 0)
    lo = np.percentile(pts, 2, 0) - 0.5
    hi = np.percentile(pts, 98, 0) + 0.5
    dims = np.ceil((hi - lo) / voxel).astype(int)
    g = OccupancyGrid3D(origin=lo, cell=voxel, dims=dims)
    for i in range(0, poses.shape[0], frame_stride):
        d = np.load(sortie / "depth" / f"depth_{i:06d}.npy")
        g.integrate_depth(poses[i], d, fx=fx, fy=fy, cx=cx, cy=cy,
                          depth_max=depth_max, stride=stride)
    return g, poses, lo, hi


def poses_to_waypoints(poses):
    wps = []
    for P in poses:
        eye = P[:3, 3]; R = P[:3, :3]
        yaw = math.degrees(math.atan2(R[0, 2], R[2, 2]))
        wps.append({"x": round(float(eye[0]), 3),
                    "y": round(float(eye[2]), 3),
                    "yaw_deg": round(float(yaw), 1)})
    return wps


def main():
    sortie = ROOT / "data/gazebo_house_wf/sortie_00"
    g, poses, lo, hi = build_grid(sortie)
    ch = float(np.median(poses[:, 1, 3]))   # camera height in world (comp1)
    print(f"grid dims={g.dims} camera_height={ch:.3f} "
          f"occ={int(g.occupied().sum())} free={int(g.free().sum())} unk={int(g.unknown().sum())}")
    state = PlannerState(grid=g, last_pose=poses[-1].copy(),
                         bounds_min=lo, bounds_max=hi, sortie_idx=1)

    out = ROOT / "data/gazebo_house_wf"
    for name, fn in [("frontier", lambda: plan_frontier_A(state, n_stops=3, yaws_per_stop=2, camera_height=ch)),
                     ("volumetric", lambda: plan_volumetric_B(state, n_waypoints=6, camera_height=ch))]:
        poses_out = fn()
        wps = poses_to_waypoints(poses_out)
        (out / f"sortie_1_{name}.yaml").write_text(yaml.safe_dump({"waypoints": wps}, sort_keys=False))
        print(f"\n[{name}] {len(wps)} waypoints -> sortie_1_{name}.yaml")
        for w in wps:
            print(f"    x={w['x']:+.2f} y={w['y']:+.2f} yaw={w['yaw_deg']:+.0f}")


if __name__ == "__main__":
    main()
