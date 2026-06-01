#!/usr/bin/env python3
"""Path A — Gazebo wreck capture: teleport the kinematic freefly_cam to a
multi-altitude orbit around the Herkules wreck centroid, capture RGB+depth+pose
at each stop, write the sortie layout consumed by sortie_to_nerf_synthetic.py.

Pose handling
-------------
/model_states (from libgazebo_ros_state.so) gives the freefly_cam link's true
world pose (position + quaternion). The Inria 3DGS pipeline expects
world-from-cam in OpenCV convention (x-right, y-down, z-forward). Gazebo links
use x-forward, y-left, z-up; the camera *optical* frame is the OpenCV one.
We compose: world_R_cam_opencv = world_R_link @ LINK_R_OPTICAL.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState
from cv_bridge import CvBridge

# rotation taking a vector expressed in the camera optical frame
# (x-right, y-down, z-forward) into the gazebo link frame (x-forward, y-left, z-up)
LINK_R_OPTICAL = np.array([[0.0, 0.0, 1.0],
                           [-1.0, 0.0, 0.0],
                           [0.0, -1.0, 0.0]], dtype=np.float64)

CAM_NAME = "freefly_cam"


def quat_to_R(qx, qy, qz, qw):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ])


def euler_to_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


def orbit_poses(center, radius, n_yaw, alt_offsets):
    """Look-at orbit. Returns list of (x,y,z, qx,qy,qz,qw) in world."""
    cx, cy, cz = center
    out = []
    for h in alt_offsets:
        for k in range(n_yaw):
            ang = 2.0 * math.pi * k / n_yaw
            ex = cx + radius * math.cos(ang)
            ey = cy + radius * math.sin(ang)
            ez = cz + h
            dx, dy, dz = cx - ex, cy - ey, cz - ez
            horiz = math.hypot(dx, dy)
            yaw = math.atan2(dy, dx)
            # link-x is forward; pitch>0 = nose down (right-hand rule about +y-left).
            pitch = -math.atan2(dz, horiz)
            qx, qy, qz, qw = euler_to_quat(0.0, pitch, yaw)
            out.append((ex, ey, ez, qx, qy, qz, qw))
    return out


class WreckCapture(Node):
    def __init__(self, out_dir: Path):
        super().__init__("wreck_orbit_capture")
        self.out_dir = out_dir
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / "depth").mkdir(parents=True, exist_ok=True)
        self.bridge = CvBridge()
        q = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST)
        self.rgb = self.depth = self.intr = self.states = None
        self.create_subscription(Image, "/camera/image_raw",
                                 lambda m: setattr(self, "rgb", m), q)
        self.create_subscription(Image, "/camera/depth/image_raw",
                                 lambda m: setattr(self, "depth", m), q)
        self.create_subscription(CameraInfo, "/camera/camera_info", self._on_ci, q)
        self.create_subscription(ModelStates, "/model_states",
                                 lambda m: setattr(self, "states", m), q)
        self.cli = self.create_client(SetEntityState, "/set_entity_state")
        self.cli.wait_for_service(timeout_sec=10)
        self.poses_out: list = []
        self.idx = 0

    def _on_ci(self, m):
        if self.intr is None:
            self.intr = {
                "width": int(m.width), "height": int(m.height),
                "fx": float(m.k[0]), "fy": float(m.k[4]),
                "cx": float(m.k[2]), "cy": float(m.k[5]),
                "hfov_deg": math.degrees(2 * math.atan(m.width / (2 * m.k[0]))),
            }

    def _spin(self, dt):
        t0 = time.time()
        while time.time() - t0 < dt:
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait_ready(self, timeout=15.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if all(v is not None for v in (self.rgb, self.depth, self.intr, self.states)):
                return True
        return False

    def teleport(self, x, y, z, qx, qy, qz, qw):
        req = SetEntityState.Request()
        s = req.state
        s.name = CAM_NAME
        s.pose.position.x, s.pose.position.y, s.pose.position.z = float(x), float(y), float(z)
        s.pose.orientation.x, s.pose.orientation.y = float(qx), float(qy)
        s.pose.orientation.z, s.pose.orientation.w = float(qz), float(qw)
        fut = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        return fut.result() and fut.result().success

    def _cam_pose(self):
        if self.states is None or CAM_NAME not in self.states.name:
            return None
        return self.states.pose[self.states.name.index(CAM_NAME)]

    def capture_one(self):
        import cv2
        rgb = self.bridge.imgmsg_to_cv2(self.rgb, desired_encoding="rgb8")
        depth = self.bridge.imgmsg_to_cv2(self.depth, desired_encoding="32FC1")
        cv2.imwrite(str(self.out_dir / "images" / f"rgb_{self.idx:06d}.png"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        np.save(self.out_dir / "depth" / f"depth_{self.idx:06d}.npy", depth)
        p = self._cam_pose()
        R_link = quat_to_R(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)
        R_cam = R_link @ LINK_R_OPTICAL
        T = np.eye(4)
        T[:3, :3] = R_cam
        T[:3, 3] = [p.position.x, p.position.y, p.position.z]
        self.poses_out.append(T)
        self.idx += 1

    def run(self, targets, settle_s=0.5):
        if not self.wait_ready():
            self.get_logger().error("topics not ready")
            return 1
        self.get_logger().info(f"streaming N={len(targets)} orbit poses")
        for i, t in enumerate(targets):
            ok = self.teleport(*t)
            if not ok:
                self.get_logger().warn(f"teleport {i} failed; skipping")
                continue
            self._spin(settle_s)            # let the depth camera re-render
            if self.rgb is None or self.depth is None:
                continue
            self.capture_one()
            if self.idx % 10 == 0:
                self.get_logger().info(f"captured {self.idx}/{len(targets)}")

        np.save(self.out_dir / "poses.npy", np.stack(self.poses_out, axis=0))
        (self.out_dir / "intrinsics.json").write_text(json.dumps(self.intr, indent=2))
        self.get_logger().info(f"done: {self.idx} frames -> {self.out_dir}")
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/xiaoming/GS_planning_handoff/data/wreck_gazebo/sortie_00")
    # Wreck centroid in *world* after the 90deg yaw at z=-60. Mesh bbox transforms to
    # world ~ x[-11,11] y[-7,6] z[-64,-50] -> centroid (0, -0.4, -57).
    ap.add_argument("--cx", type=float, default=0.0)
    ap.add_argument("--cy", type=float, default=-0.4)
    ap.add_argument("--cz", type=float, default=-57.0)
    ap.add_argument("--radius", type=float, default=22.0)
    ap.add_argument("--n-yaw", type=int, default=18)
    ap.add_argument("--alt-offsets", type=float, nargs="+", default=[6.0, 0.0, -3.0])
    ap.add_argument("--settle", type=float, default=0.5)
    args = ap.parse_args()

    targets = orbit_poses((args.cx, args.cy, args.cz), args.radius,
                          args.n_yaw, args.alt_offsets)
    rclpy.init()
    node = WreckCapture(Path(args.out))
    try:
        rc = node.run(targets, settle_s=args.settle)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
