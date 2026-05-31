#!/usr/bin/env python3
"""
Wall-Following ROS 2 Node 

  - PID controller for angular velocity
  - PID-integrated front-wall avoidance 
  - FSM: FIND_WALL -> ALIGN_LEFT -> FOLLOW_WALL <-> INTERNAL_CORNER / SEARCH_WALL
  - Adaptive velocity scaling

Subscribes to: /base_scan (sensor_msgs/LaserScan)
Publishes to:  /cmd_vel  (geometry_msgs/Twist)
"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


class State:
    FIND_WALL = "FIND_WALL"
    ALIGN_LEFT = "ALIGN_LEFT"
    FOLLOW_WALL = "FOLLOW_WALL"
    INTERNAL_CORNER = "INTERNAL_CORNER"
    SEARCH_WALL = "SEARCH_WALL"


class WallFollower(Node):
    def __init__(self):
        super().__init__("wall_follower")

        # ── Parameters ───────────────────────────────────────────────
        self.declare_parameter("d_target", 0.5)
        self.declare_parameter("v_max", 0.5)
        self.declare_parameter("v_min", 0.05)

        # PID gains
        self.declare_parameter("kp", 1.5)
        self.declare_parameter("ki", 0.05)
        self.declare_parameter("kd", 0.6)
        self.declare_parameter("i_clamp", 0.3)

        # FSM thresholds
        self.declare_parameter("front_emergency_dist", 0.35)
        self.declare_parameter("front_slow_dist", 0.8)
        self.declare_parameter("wall_follow_dist", 1.0)
        self.declare_parameter("wall_lost_dist", 2.0)

        # Turn rates
        self.declare_parameter("align_turn_rate", 0.5)
        self.declare_parameter("search_turn_rate", 0.3)
        self.declare_parameter("corner_turn_rate", 0.7)

        # Adaptive velocity
        self.declare_parameter("error_slow_threshold", 0.5)
        self.declare_parameter("omega_max", 1.5)
        self.declare_parameter("control_rate", 20.0)

        # ── Internal state ───────────────────────────────────────────
        self.state = State.FIND_WALL
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = self.get_clock().now()
        self.state_enter_time = self.get_clock().now()
        self.latest_scan = None
        self.pid_initialized = False
        self._log_counter = 0
        self._prev_d_right = 0.0
        self._stuck_counter = 0

        # ── Pub / Sub ────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.scan_sub = self.create_subscription(
            LaserScan, "/base_scan", self.scan_callback, 10
        )

        rate = self.get_parameter("control_rate").value
        self.timer = self.create_timer(1.0 / rate, self.control_loop)

        self.get_logger().info(
            f"Wall follower v6.1 | d_target={self._p('d_target'):.2f}m "
            f"Kp={self._p('kp')} Ki={self._p('ki')} Kd={self._p('kd')} "
            f"front_emergency={self._p('front_emergency_dist')}m "
            f"front_slow={self._p('front_slow_dist')}m"
        )

    # ── Helpers ──────────────────────────────────────────────────────

    def _p(self, name: str):
        return self.get_parameter(name).value

    def _time_in_state(self) -> float:
        return (self.get_clock().now() - self.state_enter_time).nanoseconds * 1e-9

    @staticmethod
    def _min_in_range(scan: LaserScan, angle_min_deg: float, angle_max_deg: float) -> float:
        angle_min = math.radians(angle_min_deg)
        angle_max = math.radians(angle_max_deg)
        readings = []
        for i in range(len(scan.ranges)):
            angle = scan.angle_min + i * scan.angle_increment
            if angle_min <= angle <= angle_max:
                r = scan.ranges[i]
                if scan.range_min < r < scan.range_max and math.isfinite(r):
                    readings.append(r)
        return min(readings) if readings else float("inf")

    def _right_distance(self, scan: LaserScan) -> float:
        return self._min_in_range(scan, -100.0, -80.0)

    def _front_distance(self, scan: LaserScan) -> float:
        return self._min_in_range(scan, -25.0, 25.0)

    def _front_right_distance(self, scan: LaserScan) -> float:
        return self._min_in_range(scan, -55.0, -35.0)

    def _angle_to_closest(self, scan: LaserScan) -> float:
        """Return the angle (rad) to the closest valid reading."""
        min_r = float("inf")
        min_angle = 0.0
        for i in range(len(scan.ranges)):
            r = scan.ranges[i]
            if scan.range_min < r < scan.range_max and math.isfinite(r):
                if r < min_r:
                    min_r = r
                    min_angle = scan.angle_min + i * scan.angle_increment
        return min_angle

    # ── Callbacks ────────────────────────────────────────────────────

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg

    # ── PID Controller ───────────────────────────────────────────────

    def compute_pid(self, error: float, dt: float) -> float:
        kp = self._p("kp")
        ki = self._p("ki")
        kd = self._p("kd")
        i_clamp = self._p("i_clamp")
        omega_max = self._p("omega_max")

        p_term = kp * error

        self.integral += error * dt
        self.integral = max(-i_clamp, min(i_clamp, self.integral))
        i_term = ki * self.integral

        d_term = 0.0
        if self.pid_initialized and dt > 0:
            d_term = kd * (error - self.prev_error) / dt
        self.prev_error = error
        self.pid_initialized = True

        omega = p_term + i_term + d_term
        return max(-omega_max, min(omega_max, omega))

    # ── Adaptive Velocity ────────────────────────────────────────────

    def adaptive_velocity(self, error: float, d_front: float, d_front_right: float) -> float:
        v_max = self._p("v_max")
        v_min = self._p("v_min")
        threshold = self._p("error_slow_threshold")
        front_slow = self._p("front_slow_dist")
        front_emg = self._p("front_emergency_dist")

        # error-based slowdown 
        error_ratio = min(abs(error) / threshold, 1.0)
        error_ratio = error_ratio ** 2

        # front-proximity slowdown
        d_threat = d_front
        if d_front_right < front_slow:
            d_threat = min(d_front, d_front_right)

        front_ratio = 0.0
        if d_threat < front_slow:
            front_ratio = 1.0 - (d_threat - front_emg) / (front_slow - front_emg)
            front_ratio = max(0.0, min(1.0, front_ratio))

        ratio = max(error_ratio, front_ratio)
        return v_max - ratio * (v_max - v_min)

    # ── Front avoidance bias  ─────

    def front_avoidance_bias(self, d_front: float, d_front_right: float) -> float:
        """
        When front wall is close, add a positive (left-turn) bias to omega.
        Only uses d_front_right when it's genuinely close (< front_slow),
        NOT when it reads far through an open junction corridor.
        """
        front_slow = self._p("front_slow_dist")
        front_emg = self._p("front_emergency_dist")

        d_threat = d_front
        if d_front_right < front_slow:
            d_threat = min(d_front, d_front_right)

        if d_threat >= front_slow:
            return 0.0

        max_bias = 0.8
        ratio = 1.0 - (d_threat - front_emg) / (front_slow - front_emg)
        ratio = max(0.0, min(1.0, ratio))
        return max_bias * ratio

    # ── State transition helper ──────────────────────────────────────

    def _transition(self, new_state: str):
        old = self.state
        self.state = new_state
        self.state_enter_time = self.get_clock().now()
        self.prev_error = 0.0
        self.integral = 0.0
        self.pid_initialized = False
        self.get_logger().info(f"FSM: {old} -> {new_state}")

    # ── FSM + Control Loop ───────────────────────────────────────────

    def control_loop(self):
        if self.latest_scan is None:
            return

        scan = self.latest_scan
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now
        if dt <= 0:
            dt = 0.05

        d_right = self._right_distance(scan)
        d_front = self._front_distance(scan)
        d_front_right = self._front_right_distance(scan)

        cmd = Twist()

        front_emg = self._p("front_emergency_dist")
        front_slow = self._p("front_slow_dist")
        wall_follow = self._p("wall_follow_dist")
        wall_lost = self._p("wall_lost_dist")
        d_target = self._p("d_target")
        t = self._time_in_state()

        # ── FSM transitions ──────────────────────────────────────

        if self.state == State.FIND_WALL:
            if d_front < front_emg:
                self._transition(State.ALIGN_LEFT)
            elif d_right < wall_follow:
                self._transition(State.FOLLOW_WALL)

        elif self.state == State.ALIGN_LEFT:
            if d_front > front_emg * 2.5 and d_right < wall_follow:
                self._transition(State.FOLLOW_WALL)
            # Timeout: stuck turning in place
            elif t > 5.0:
                self.get_logger().warn("ALIGN_LEFT timeout")
                self._transition(State.FIND_WALL)

        elif self.state == State.FOLLOW_WALL:
            # INTERNAL_CORNER
            if d_front < front_emg:
                self._transition(State.INTERNAL_CORNER)
            elif d_front_right < front_emg and d_front_right < front_slow:
                self._transition(State.INTERNAL_CORNER)
            # SEARCH_WALL
            elif d_right > wall_lost and t > 2.0:
                self._transition(State.SEARCH_WALL)

        elif self.state == State.INTERNAL_CORNER:
            if t > 0.5:
                if d_front > front_emg * 2.0 and d_front_right > front_emg * 1.5:
                    if d_right < wall_lost:
                        self._transition(State.FOLLOW_WALL)
                    else:
                        self._transition(State.SEARCH_WALL)
            if t > 5.0:
                self.get_logger().warn("INTERNAL_CORNER timeout — backing up")
                self._transition(State.FIND_WALL)

        elif self.state == State.SEARCH_WALL:
            if t > 1.0:
                if d_right < wall_follow:
                    self._transition(State.FOLLOW_WALL)
                elif d_front < front_emg:
                    # hit a wall while searching then continue searching right
                    self._transition(State.INTERNAL_CORNER)
            if t > 10.0:
                self.get_logger().warn("SEARCH_WALL timeout")
                self._transition(State.FIND_WALL)

        # ── Actions ──────────────────────────────────────────────

        if self.state == State.FIND_WALL:
            closest = min(d_right, d_front, d_front_right)
            if closest < 0.4 and self._time_in_state() < 2.0:
                # too close to walls
                cmd.linear.x = -0.2
                cmd.angular.z = 0.3
            else:
                # rotate to face the closest wall
                angle_to_wall = self._angle_to_closest(scan)
                if abs(angle_to_wall) > 0.15:
                    cmd.linear.x = 0.0
                    cmd.angular.z = 0.5 if angle_to_wall > 0 else -0.5
                else:
                    cmd.linear.x = 0.4
                    cmd.angular.z = 0.0

        elif self.state == State.ALIGN_LEFT:
            cmd.linear.x = 0.0
            cmd.angular.z = self._p("align_turn_rate")

        elif self.state == State.FOLLOW_WALL:
            error = d_target - d_right
            omega_pid = self.compute_pid(error, dt)
            omega_front = self.front_avoidance_bias(d_front, d_front_right)
            omega = omega_pid + omega_front
            omega = max(-self._p("omega_max"), min(self._p("omega_max"), omega))

            cmd.linear.x = self.adaptive_velocity(error, d_front, d_front_right)
            cmd.angular.z = omega

        elif self.state == State.INTERNAL_CORNER:
            # If walls close on front and right, back up while turning
            if d_front < front_emg * 2.0 and d_right < 0.5:
                cmd.linear.x = -0.15
                cmd.angular.z = self._p("corner_turn_rate")
            else:
                cmd.linear.x = 0.0
                cmd.angular.z = self._p("corner_turn_rate")

        elif self.state == State.SEARCH_WALL:
            # corner recovery, drive forward + turn right
            if self._time_in_state() < 4.0:
                cmd.linear.x = 0.3
                cmd.angular.z = -0.5
            else:
                # drive toward closest wall
                angle_to_wall = self._angle_to_closest(scan)
                if abs(angle_to_wall) > 0.2:
                    cmd.linear.x = 0.1
                    cmd.angular.z = 0.5 if angle_to_wall > 0 else -0.5
                else:
                    cmd.linear.x = 0.4
                    cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)

        # ── stuck detection ───────────────────
        if self.state == State.FOLLOW_WALL:
            if (abs(d_right - self._prev_d_right) < 0.02 and
                    abs(cmd.angular.z) > 0.05 and
                    t > 3.0):
                self._stuck_counter += 1
            else:
                self._stuck_counter = 0
            self._prev_d_right = d_right

            if self._stuck_counter > 80:
                self.get_logger().warn("Stuck detected! Recovering.")
                self._stuck_counter = 0
                self._transition(State.INTERNAL_CORNER)
        else:
            self._stuck_counter = 0
            self._prev_d_right = d_right

        self._log_counter += 1
        if self._log_counter % 40 == 0:
            self.get_logger().info(
                f"[{self.state}] d_r={d_right:.2f} d_f={d_front:.2f} "
                f"d_fr={d_front_right:.2f} v={cmd.linear.x:.2f} w={cmd.angular.z:.2f}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = WallFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            stop_cmd = Twist()
            node.cmd_pub.publish(stop_cmd)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()