#!/usr/bin/env python3
"""
PA3 map publisher 

read ROS map YAML and the referenced grayscale image, builds a nav_msgs/OccupancyGrid

publish it once on /map with TRANSIENT_LOCAL 


    python3 map_publisher.py --ros-args -p yaml_filename:=/abs/path/maze.yml
"""
import math
import os
import sys

import numpy as np
import yaml
from PIL import Image

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from nav_msgs.msg import OccupancyGrid


class MapPublisher(Node):
    def __init__(self):
        super().__init__("pa3_map_publisher")
        self.declare_parameter("yaml_filename", "")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("topic", "/map")

        yaml_path = self.get_parameter("yaml_filename").value
        if not yaml_path:
            self.get_logger().fatal("yaml_filename parameter is required")
            sys.exit(2)

        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        img_rel = cfg["image"]
        img_path = img_rel if os.path.isabs(img_rel) else \
                   os.path.join(os.path.dirname(yaml_path), img_rel)

        resolution = float(cfg["resolution"])
        origin = cfg["origin"]                 # [x, y, theta]
        occupied_thresh = float(cfg["occupied_thresh"])
        free_thresh = float(cfg["free_thresh"])
        negate = int(cfg.get("negate", 0))

        img = np.array(Image.open(img_path).convert("L"))
        # image row 0 is the top
        img = np.flipud(img)

        # occupancy probability per pixel
        if negate:
            p = img.astype(np.float32) / 255.0
        else:
            p = (255.0 - img.astype(np.float32)) / 255.0

        data = np.full(p.shape, -1, dtype=np.int8)   # unknown
        data[p > occupied_thresh] = 100              # wall
        data[p < free_thresh] = 0                    # free

        msg = OccupancyGrid()
        msg.header.frame_id = self.get_parameter("frame_id").value
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = resolution
        msg.info.height, msg.info.width = data.shape
        msg.info.origin.position.x = float(origin[0])
        msg.info.origin.position.y = float(origin[1])
        theta = float(origin[2])
        msg.info.origin.orientation.z = math.sin(theta * 0.5)
        msg.info.origin.orientation.w = math.cos(theta * 0.5)
        msg.data = data.flatten().tolist()

        latched = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        topic = self.get_parameter("topic").value
        self.pub = self.create_publisher(OccupancyGrid, topic, latched)
        self.pub.publish(msg)

        wall = int((data == 100).sum())
        free = int((data == 0).sum())
        unknown = int((data == -1).sum())
        self.get_logger().info(
            f"published {msg.info.width}×{msg.info.height} map @ {resolution} m/cell "
            f"on {topic}  (free={free}, wall={wall}, unknown={unknown})")


def main():
    rclpy.init()
    node = MapPublisher()
    try:
        rclpy.spin(node)        
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
