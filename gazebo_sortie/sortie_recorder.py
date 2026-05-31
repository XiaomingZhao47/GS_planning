#!/usr/bin/env python3
"""Continuous RGB-D + pose recorder for controller-driven sorties in Gazebo.

Runs as a passive node ALONGSIDE a controller (e.g. wall_follower_tb3.py).
The controller drives /cmd_vel; this node only listens and dumps frames.

Capture is *movement-gated*: a frame is saved whenever the robot has moved
>= --trans-gate metres OR rotated >= --rot-gate degrees since the last save.
This avoids piling up near-duplicate frames while stationary and guarantees a
translational baseline between views (which 3DGS needs to triangulate geometry).

Output layout matches what src/sim/sortie_to_nerf_synthetic.py consumes:

    <out>/sortie_<k>/
        images/rgb_<i>.png        (uint8 RGB)
        depth/depth_<i>.npy        (float32, metres, OpenCV camera frame)
        poses.npy                  ((N,4,4) world-from-cam, same y-up convention
                                    as docker .../sortie_executor.py)
        intrinsics.json

NOTE: like sortie_executor.py, this first cut treats base_link as the camera
frame (forward-facing) and uses Gazebo GT /odom for pose. Good enough to feed
the existing NeRF-synthetic converter; swapping in the TF base_link->camera
optical transform or RTAB-Map poses is a localized change.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import List

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge


def _quat_to_yaw(qx, qy, qz, qw):
    return math.atan2(2 * (qw * qz + qx * qy),
                      1 - 2 * (qy * qy + qz * qz))


class SortieRecorder(Node):
    def __init__(self, out_dir: Path, sortie_idx: int,
                 trans_gate: float, rot_gate_deg: float,
                 max_frames: int, duration: float):
        super().__init__("sge_sortie_recorder")
        self.bridge = CvBridge()
        self.sortie_dir = Path(out_dir) / f"sortie_{sortie_idx:02d}"
        (self.sortie_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.sortie_dir / "depth").mkdir(parents=True, exist_ok=True)

        qos = QoSProfile(depth=10,
                         reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST)
        self.rgb_msg = None
        self.depth_msg = None
        self.intrinsics = None
        self.odom = None

        self.create_subscription(Image, "/camera/image_raw", self._on_rgb, qos)
        self.create_subscription(Image, "/camera/depth/image_raw", self._on_depth, qos)
        self.create_subscription(CameraInfo, "/camera/camera_info", self._on_ci, qos)
        self.create_subscription(Odometry, "/odom", self._on_odom, qos)

        self.trans_gate = trans_gate
        self.rot_gate = math.radians(rot_gate_deg)
        self.max_frames = max_frames
        self.duration = duration

        self.poses: List[np.ndarray] = []
        self.idx = 0
        self.last_xy = None
        self.last_yaw = None
        self.t0 = None
        self.done = False

        self.timer = self.create_timer(0.1, self._tick)  # 10 Hz gate check
        self.get_logger().info(
            f"recorder -> {self.sortie_dir} | gate: {trans_gate:.2f}m / "
            f"{rot_gate_deg:.0f}deg | cap {max_frames} frames / {duration:.0f}s")

    # --- subscriptions ---
    def _on_rgb(self, m): self.rgb_msg = m
    def _on_depth(self, m): self.depth_msg = m
    def _on_odom(self, m): self.odom = m

    def _on_ci(self, m):
        if self.intrinsics is None:
            self.intrinsics = {
                "width": int(m.width), "height": int(m.height),
                "fx": float(m.k[0]), "fy": float(m.k[4]),
                "cx": float(m.k[2]), "cy": float(m.k[5]),
                "hfov_deg": math.degrees(2 * math.atan(m.width / (2 * m.k[0]))),
            }

    def _ready(self):
        return (self.rgb_msg is not None and self.depth_msg is not None
                and self.intrinsics is not None and self.odom is not None)

    def _tick(self):
        if self.done or not self._ready():
            return
        now = time.time()
        if self.t0 is None:
            self.t0 = now
            self.get_logger().info("topics live; capturing first frame")

        p = self.odom.pose.pose
        x, y, z = p.position.x, p.position.y, p.position.z
        yaw = _quat_to_yaw(p.orientation.x, p.orientation.y,
                           p.orientation.z, p.orientation.w)

        gate = self.last_xy is None
        if not gate:
            d = math.hypot(x - self.last_xy[0], y - self.last_xy[1])
            dyaw = abs((yaw - self.last_yaw + math.pi) % (2 * math.pi) - math.pi)
            gate = d >= self.trans_gate or dyaw >= self.rot_gate
        if gate:
            self._save(x, y, z, yaw)
            self.last_xy = (x, y)
            self.last_yaw = yaw

        if (self.max_frames and self.idx >= self.max_frames) or \
           (self.duration and now - self.t0 >= self.duration):
            self.finish()

    def _save(self, x, y, z, yaw):
        import cv2
        rgb = self.bridge.imgmsg_to_cv2(self.rgb_msg, desired_encoding="rgb8")
        depth = self.bridge.imgmsg_to_cv2(self.depth_msg, desired_encoding="32FC1")
        cv2.imwrite(str(self.sortie_dir / "images" / f"rgb_{self.idx:06d}.png"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        # Gazebo depth uses inf/nan for no-return; zero them (downstream masks z>0.1).
        depth = np.nan_to_num(depth.astype(np.float32),
                              nan=0.0, posinf=0.0, neginf=0.0)
        np.save(self.sortie_dir / "depth" / f"depth_{self.idx:06d}.npy", depth)

        # world-from-cam in the same y-up convention as sortie_executor.py
        R = np.array([[math.cos(yaw), 0, math.sin(yaw)],
                      [0, 1, 0],
                      [-math.sin(yaw), 0, math.cos(yaw)]])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, z, y]
        self.poses.append(T)
        self.idx += 1
        if self.idx % 10 == 0:
            self.get_logger().info(f"captured {self.idx} frames")

    def finish(self):
        if self.done:
            return
        self.done = True
        if self.poses:
            np.save(self.sortie_dir / "poses.npy",
                    np.stack(self.poses, axis=0))
            (self.sortie_dir / "intrinsics.json").write_text(
                json.dumps(self.intrinsics, indent=2))
            self.get_logger().info(
                f"sortie done: {len(self.poses)} frames -> {self.sortie_dir}")
        else:
            self.get_logger().warn("no frames captured (topics never became ready?)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output root dir")
    ap.add_argument("--sortie-idx", type=int, default=0)
    ap.add_argument("--trans-gate", type=float, default=0.10, help="metres")
    ap.add_argument("--rot-gate", type=float, default=12.0, help="degrees")
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--duration", type=float, default=90.0, help="seconds")
    args, _ = ap.parse_known_args()

    rclpy.init()
    node = SortieRecorder(Path(args.out), args.sortie_idx,
                          args.trans_gate, args.rot_gate,
                          args.max_frames, args.duration)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info("interrupted; flushing captured frames")
    finally:
        node.finish()  # write partial sortie on early exit
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
