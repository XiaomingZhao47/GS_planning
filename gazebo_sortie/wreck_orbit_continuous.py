#!/usr/bin/env python3
"""Smooth continuous-orbit driver for the freefly_cam around the wreck.

Unlike wreck_orbit_capture (which teleports + settles + captures), this
runs SetEntityState at high rate (~20 Hz) along a continuous orbit so the
camera topic stream looks like a smooth flythrough -- the right thing for
recording a video.

Two revolutions over `duration` seconds, with altitude oscillating
sinusoidally between alt_min and alt_max.
"""
from __future__ import annotations
import argparse
import math
import time

import rclpy
from gazebo_msgs.srv import SetEntityState
from rclpy.node import Node


CAM_NAME = "freefly_cam"
LOOP_HZ = 8.0  # Each teleport is awaited; >10 Hz overwhelms /set_entity_state.


def euler_to_quat(roll: float, pitch: float, yaw: float):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


class ContinuousOrbit(Node):
    def __init__(self, center, radius, alt_min, alt_max, duration, revolutions):
        super().__init__("wreck_orbit_continuous")
        self.center = center
        self.radius = radius
        self.alt_min = alt_min
        self.alt_max = alt_max
        self.duration = duration
        self.revolutions = revolutions
        self.cli = self.create_client(SetEntityState, "/set_entity_state")
        if not self.cli.wait_for_service(timeout_sec=15.0):
            self.get_logger().error("set_entity_state service did not appear")
            raise SystemExit(1)

    def _teleport(self, x, y, z, qx, qy, qz, qw) -> None:
        req = SetEntityState.Request()
        s = req.state
        s.name = CAM_NAME
        s.pose.position.x = float(x)
        s.pose.position.y = float(y)
        s.pose.position.z = float(z)
        s.pose.orientation.x = float(qx)
        s.pose.orientation.y = float(qy)
        s.pose.orientation.z = float(qz)
        s.pose.orientation.w = float(qw)
        # Await each call -- fire-and-forget at >5 Hz piles up futures in
        # /set_entity_state and only the first one lands, so the camera
        # appears stuck. Short 0.3 s timeout keeps the loop responsive.
        fut = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=0.3)

    def run(self) -> None:
        cx, cy, cz = self.center
        n_steps = int(self.duration * LOOP_HZ)
        period = 1.0 / LOOP_HZ
        self.get_logger().info(
            f"orbiting r={self.radius} alt [{self.alt_min}, {self.alt_max}] "
            f"for {self.duration:.1f}s x{self.revolutions} revolutions "
            f"-> {n_steps} steps at {LOOP_HZ:.0f} Hz"
        )
        t0 = time.time()
        for i in range(n_steps):
            t = i / float(n_steps)            # 0 .. 1
            ang = 2.0 * math.pi * self.revolutions * t
            # Altitude oscillates 1.5 cycles over the run so the camera
            # sweeps across the wreck silhouette.
            alt = self.alt_min + (self.alt_max - self.alt_min) * 0.5 * (
                1.0 - math.cos(1.5 * 2.0 * math.pi * t)
            )
            ex = cx + self.radius * math.cos(ang)
            ey = cy + self.radius * math.sin(ang)
            ez = cz + alt
            dx, dy, dz = cx - ex, cy - ey, cz - ez
            horiz = math.hypot(dx, dy)
            yaw = math.atan2(dy, dx)
            pitch = -math.atan2(dz, horiz)
            qx, qy, qz, qw = euler_to_quat(0.0, pitch, yaw)
            step_start = time.time()
            self._teleport(ex, ey, ez, qx, qy, qz, qw)
            # Pace with real sleep -- spin_once returns the instant the
            # service future resolves, which is microseconds, so the entire
            # orbit would otherwise complete in <1 s.
            remaining = period - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)
        self.get_logger().info(f"orbit done ({time.time() - t0:.1f}s wall)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cx", type=float, default=0.0)
    ap.add_argument("--cy", type=float, default=-0.4)
    ap.add_argument("--cz", type=float, default=-57.0)
    ap.add_argument("--radius", type=float, default=22.0)
    ap.add_argument("--alt-min", type=float, default=-3.0)
    ap.add_argument("--alt-max", type=float, default=6.0)
    ap.add_argument("--duration", type=float, default=90.0)
    ap.add_argument("--revolutions", type=float, default=2.0)
    args = ap.parse_args()

    rclpy.init()
    node = ContinuousOrbit(
        (args.cx, args.cy, args.cz),
        args.radius, args.alt_min, args.alt_max,
        args.duration, args.revolutions,
    )
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
