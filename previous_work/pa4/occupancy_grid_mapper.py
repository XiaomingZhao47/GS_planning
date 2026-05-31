#!/usr/bin/env python3
"""
PA4: Occupancy Grid Mapping Node

Subscribes:
  /base_scan   sensor_msgs/LaserScan
  /tf          (odom -> base_link / scan frame)

Publishes:
  /map         nav_msgs/OccupancyGrid (in the odom frame TRANSIENT_LOCAL)

Algorithm:
  Probabilistic log-odds update with Bresenham raycasting per beam
  The map grows when the laser endpoint leaves current grid bounds
"""

import math

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)

from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan

import tf2_ros
from tf2_ros import (ConnectivityException, ExtrapolationException,
                     LookupException)


def quat_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def bresenham_line(x0: int, y0: int, x1: int, y1: int):
    """Return (cols, rows) arrays of integer cells along a 4-conn-ish line"""
    dx = abs(x1 - x0); dy = abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy
    cols = [x0]; rows = [y0]
    x, y = x0, y0
    while x != x1 or y != y1:
        e2 = 2 * err
        if e2 > -dy:
            err -= dy; x += sx
        if e2 < dx:
            err += dx; y += sy
        cols.append(x); rows.append(y)
    return cols, rows


class OccupancyMapper(Node):
    def __init__(self):
        super().__init__("pa4_occupancy_mapper")

        # ── parameters
        self.declare_parameter("resolution", 0.05)          # m / cell
        self.declare_parameter("initial_size", 200)         # cells per side
        self.declare_parameter("grow_chunk", 100)           # cells added on growth
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("scan_topic", "/base_scan")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("publish_rate", 2.0)         # Hz
        self.declare_parameter("beam_skip", 4)              # use every Nth beam
        self.declare_parameter("max_range_factor", 0.95)    
        # log-odds sensor model
        self.declare_parameter("p_occ", 0.7)                # P(occ | hit)  > 0.5
        self.declare_parameter("p_free", 0.3)               # P(occ | free) < 0.5
        self.declare_parameter("l_clamp", 5.0)              # clamp on log-odds
        # thresholds for the published OccupancyGrid
        self.declare_parameter("occ_thresh", 0.65)
        self.declare_parameter("free_thresh", 0.35)
        # snapshot recording
        self.declare_parameter("record_dir", "/tmp/pa4_maps")
        self.declare_parameter("record_period", 0.0)        # 0 = disabled
        self.declare_parameter("record_on_exit", True)

        # cache static params
        self.res = float(self.get_parameter("resolution").value)
        self.odom_frame = self.get_parameter("odom_frame").value
        self.beam_skip = max(1, int(self.get_parameter("beam_skip").value))
        self.max_range_factor = float(self.get_parameter("max_range_factor").value)
        self.l_occ = math.log(self.get_parameter("p_occ").value
                              / (1.0 - self.get_parameter("p_occ").value))
        self.l_free = math.log(self.get_parameter("p_free").value
                               / (1.0 - self.get_parameter("p_free").value))
        self.l_clamp = float(self.get_parameter("l_clamp").value)
        self.occ_thresh = float(self.get_parameter("occ_thresh").value)
        self.free_thresh = float(self.get_parameter("free_thresh").value)
        self.grow_chunk = int(self.get_parameter("grow_chunk").value)

        # ── grid (centred on origin: world 0,0 sits in the middle)
        n = int(self.get_parameter("initial_size").value)
        self.log_odds = np.zeros((n, n), dtype=np.float32)
        self.touched = np.zeros((n, n), dtype=bool)
        self.height = n
        self.width = n
        self.origin_x = -0.5 * n * self.res
        self.origin_y = -0.5 * n * self.res

        # ── tf2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── pubs / subs
        latched = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.pub = self.create_publisher(
            OccupancyGrid, self.get_parameter("map_topic").value, latched)
        self.create_subscription(
            LaserScan, self.get_parameter("scan_topic").value, self.scan_cb, 10)

        rate = float(self.get_parameter("publish_rate").value)
        self.create_timer(1.0 / rate, self.publish_map)

        # snapshot recording
        self.record_dir = self.get_parameter("record_dir").value
        rp = float(self.get_parameter("record_period").value)
        if rp > 0.0:
            self.create_timer(rp, lambda: self.save_snapshot())
            self.get_logger().info(f"recording every {rp:.1f}s → {self.record_dir}")

        self.get_logger().info(
            f"pa4_occupancy_mapper ready  res={self.res} m/cell, "
            f"grid {n}×{n}, l_occ={self.l_occ:+.2f}, l_free={self.l_free:+.2f}")

    # ── coordinate helpers
    def world_to_grid(self, x: float, y: float):
        return (int((x - self.origin_x) / self.res),
                int((y - self.origin_y) / self.res))

    def _grow_to_include(self, x: float, y: float):
        col, row = self.world_to_grid(x, y)
        margin = 2
        add_left = max(0, -col + margin)
        add_right = max(0, col - (self.width - 1) + margin)
        add_bottom = max(0, -row + margin)
        add_top = max(0, row - (self.height - 1) + margin)
        if not (add_left or add_right or add_bottom or add_top):
            return
        chunk = self.grow_chunk
        rnd = lambda v: ((v + chunk - 1) // chunk) * chunk if v else 0
        add_left = rnd(add_left); add_right = rnd(add_right)
        add_bottom = rnd(add_bottom); add_top = rnd(add_top)

        new_h = self.height + add_bottom + add_top
        new_w = self.width + add_left + add_right
        new_lo = np.zeros((new_h, new_w), dtype=np.float32)
        new_tc = np.zeros((new_h, new_w), dtype=bool)
        new_lo[add_bottom:add_bottom + self.height,
               add_left:add_left + self.width] = self.log_odds
        new_tc[add_bottom:add_bottom + self.height,
               add_left:add_left + self.width] = self.touched
        self.log_odds = new_lo
        self.touched = new_tc
        self.height = new_h
        self.width = new_w
        self.origin_x -= add_left * self.res
        self.origin_y -= add_bottom * self.res

    # ── scan callback, log-odds update
    def scan_cb(self, scan: LaserScan):
        # sensor pose in odom at scan time 
        try:
            tf = self.tf_buffer.lookup_transform(
                self.odom_frame, scan.header.frame_id,
                scan.header.stamp,
                timeout=rclpy.duration.Duration(seconds=0.05))
        except (LookupException, ConnectivityException, ExtrapolationException):
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.odom_frame, scan.header.frame_id, rclpy.time.Time())
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                self.get_logger().warn(f"tf {self.odom_frame}<-{scan.header.frame_id}: {e}",
                                       throttle_duration_sec=2.0)
                return

        sx = tf.transform.translation.x
        sy = tf.transform.translation.y
        syaw = quat_to_yaw(tf.transform.rotation)

        self._grow_to_include(sx, sy)
        valid_max = scan.range_max * self.max_range_factor

        for i in range(0, len(scan.ranges), self.beam_skip):
            r = scan.ranges[i]
            if math.isnan(r) or math.isinf(r) or r < scan.range_min:
                continue
            hit = r < valid_max
            r_use = min(r, scan.range_max)
            ang = syaw + scan.angle_min + i * scan.angle_increment
            ex = sx + r_use * math.cos(ang)
            ey = sy + r_use * math.sin(ang)
            self._grow_to_include(ex, ey)

            rc, rr = self.world_to_grid(sx, sy)
            ec, er = self.world_to_grid(ex, ey)
            cols, rows = bresenham_line(rc, rr, ec, er)

            cols_a = np.asarray(cols, dtype=np.intp)
            rows_a = np.asarray(rows, dtype=np.intp)
            if len(cols_a) > 1:
                self.log_odds[rows_a[:-1], cols_a[:-1]] += self.l_free
                self.touched[rows_a[:-1], cols_a[:-1]] = True
            last_lo = self.l_occ if hit else self.l_free
            self.log_odds[rows_a[-1], cols_a[-1]] += last_lo
            self.touched[rows_a[-1], cols_a[-1]] = True

        np.clip(self.log_odds, -self.l_clamp, self.l_clamp, out=self.log_odds)

    # ── periodic publish
    def publish_map(self):
        p = 1.0 / (1.0 + np.exp(-self.log_odds))
        data = np.full(p.shape, -1, dtype=np.int8)
        data[self.touched & (p < self.free_thresh)] = 0
        data[self.touched & (p > self.occ_thresh)] = 100

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.odom_frame
        msg.info.resolution = self.res
        msg.info.height, msg.info.width = data.shape
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.orientation.w = 1.0
        msg.data = data.flatten().tolist()
        self.pub.publish(msg)

    # ── save snapshot (PGM + YAML + raw log-odds)
    def save_snapshot(self, label: str = ""):
        import os, time
        os.makedirs(self.record_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = os.path.join(self.record_dir, f"map_{ts}" + (f"_{label}" if label else ""))

        # raw log-odds + touched mask, full precision
        np.save(base + "_logodds.npy", self.log_odds)
        np.save(base + "_touched.npy", self.touched)

        # PGM
        p = 1.0 / (1.0 + np.exp(-self.log_odds))
        img = np.full(p.shape, 205, dtype=np.uint8)
        img[self.touched & (p < self.free_thresh)] = 254
        img[self.touched & (p > self.occ_thresh)] = 0
        img = np.flipud(img)                     # PGM row 0 is top; OccupancyGrid row 0 is bottom

        with open(base + ".pgm", "wb") as f:
            f.write(f"P5\n{img.shape[1]} {img.shape[0]}\n255\n".encode())
            f.write(img.tobytes())

        # YAML with nav2 map_server format
        with open(base + ".yaml", "w") as f:
            f.write(f"image: {os.path.basename(base)}.pgm\n")
            f.write(f"resolution: {self.res}\n")
            f.write(f"origin: [{self.origin_x:.4f}, {self.origin_y:.4f}, 0.0]\n")
            f.write("negate: 0\n")
            f.write(f"occupied_thresh: {self.occ_thresh}\n")
            f.write(f"free_thresh: {self.free_thresh}\n")

        n_touched = int(self.touched.sum())
        n_occ = int((self.touched & (p > self.occ_thresh)).sum())
        n_free = int((self.touched & (p < self.free_thresh)).sum())
        self.get_logger().info(
            f"snapshot: {base}.pgm  ({self.width}×{self.height}, "
            f"touched={n_touched}, occ={n_occ}, free={n_free})")


def main():
    rclpy.init()
    node = OccupancyMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.get_parameter("record_on_exit").value:
            try:
                node.save_snapshot(label="final")
            except Exception as e:
                node.get_logger().warn(f"final snapshot failed: {e}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
