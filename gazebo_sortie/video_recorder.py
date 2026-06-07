#!/usr/bin/env python3
"""Record /camera/image_raw to an mp4 file for a fixed duration.

Subscribes to the standard Gazebo+TB3 / freefly_cam topic
(/camera/image_raw, sensor_msgs/Image) and writes each frame into an
mp4 via OpenCV. Exits after --duration seconds.

Usage:
    python3 video_recorder.py --out /tmp/tb3.mp4 --duration 90
    python3 video_recorder.py --out /tmp/uuv.mp4 --duration 90 --fps 15
"""
from __future__ import annotations
import argparse
import sys
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class Recorder(Node):
    def __init__(self, out_path: str, fps: int, duration: float):
        super().__init__("video_recorder")
        self.bridge = CvBridge()
        self.out_path = out_path
        self.fps = fps
        self.duration = duration
        self.writer = None  # opened lazily on the first frame
        self.t_start = None
        self.n_frames = 0
        self.create_subscription(
            Image, "/camera/image_raw", self._on_img, qos_profile_sensor_data
        )
        self.get_logger().info(
            f"recording /camera/image_raw -> {out_path} at {fps} fps for {duration} s"
        )

    def _on_img(self, msg: Image) -> None:
        if self.t_start is None:
            self.t_start = time.time()
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if self.writer is None:
            h, w = bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.out_path, fourcc, self.fps, (w, h))
            self.get_logger().info(f"opened writer ({w}x{h})")
        self.writer.write(bgr)
        self.n_frames += 1
        if (time.time() - self.t_start) >= self.duration:
            self.get_logger().info(
                f"reached {self.duration:.1f} s; wrote {self.n_frames} frames"
            )
            self.shutdown()

    def shutdown(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        rclpy.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--duration", type=float, default=90.0)
    args = ap.parse_args()

    rclpy.init()
    node = Recorder(args.out, args.fps, args.duration)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()


if __name__ == "__main__":
    main()
