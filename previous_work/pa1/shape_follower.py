#!/usr/bin/env python3

# Author: Xiaoming Zhao
# Description: OPEN-LOOP ROS2 shape follower with trajectory logging + RViz vis.
#
#   Shapes: 1. Trapezoid  2. D-Shape  3. General Polygon
#
# Usage:
#   ros2 launch turtlebot3_gazebo empty_world.launch.py
#   python3 shape_follower.py
#   rviz2  
#
# Logs written to ~/ros2_ws/logs/
#   trajectory_OL_<shape>_<timestamp>.csv
#   waypoints_OL_<shape>_<timestamp>.csv

import math
import csv
import os
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.duration import Duration
from geometry_msgs.msg import Twist, PointStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Header
import tf2_ros

# ── Constants ────────────────────────────────────────────────────────────────
FREQUENCY        = 10       # Hz
LINEAR_VEL       = 0.15     # m/s
ANGULAR_VEL      = 0.3      # rad/s
WHEEL_BASE       = 0.287    # m, TurtleBot3 waffle_pi
USE_SIM_TIME     = True
STARTUP_TIMEOUT  = 20.0     # s
POSITION_TOL     = 0.05     # m
ANGLE_TOL        = 0.02     # rad
LOG_DIR          = os.path.expanduser('~/ros2_ws/logs')


# ── Helpers ──────────────────────────────────────────────────────────────────

def normalize_angle(a):
    while a > math.pi:  a -= 2.0 * math.pi
    while a < -math.pi: a += 2.0 * math.pi
    return a

def euler_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def make_pose_stamped(stamp, frame, x, y, yaw=0.0):
    """Build a PoseStamped for a Path message."""
    ps = PoseStamped()
    ps.header.stamp = stamp
    ps.header.frame_id = frame
    ps.pose.position.x = x
    ps.pose.position.y = y
    ps.pose.position.z = 0.0
    ps.pose.orientation.z = math.sin(yaw / 2.0)
    ps.pose.orientation.w = math.cos(yaw / 2.0)
    return ps


# ── Node ─────────────────────────────────────────────────────────────────────

class ShapeFollower(Node):

    def __init__(self):
        super().__init__('shape_follower')

        self.set_parameters([rclpy.parameter.Parameter(
            'use_sim_time', rclpy.Parameter.Type.BOOL, USE_SIM_TIME)])

        # Publishers
        self._cmd_pub = self.create_publisher(Twist, 'cmd_vel', 1)
        self._actual_path_pub = self.create_publisher(Path, 'actual_path', 10)
        self._ideal_path_pub  = self.create_publisher(Path, 'ideal_path', 10)

        # Subscribers
        self._odom_sub = self.create_subscription(
            Odometry, 'odom', self._odom_callback, 10)

        # TF2
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Robot state
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._odom_ready = False

        # Start pose
        self._start_x = 0.0
        self._start_y = 0.0
        self._start_yaw = 0.0

        # RViz paths
        self._actual_path = Path()
        self._actual_path.header.frame_id = 'odom'
        self._ideal_path = Path()
        self._ideal_path.header.frame_id = 'odom'

        # CSV logs
        self._traj_log = []
        self._waypoint_log = []

        os.makedirs(LOG_DIR, exist_ok=True)

    # ── Callbacks ────────────────────────────────────────────────────────

    def _odom_callback(self, msg):
        self._x = msg.pose.pose.position.x
        self._y = msg.pose.pose.position.y
        self._yaw = euler_from_quaternion(msg.pose.pose.orientation)
        self._odom_ready = True

    # ── Utility ──────────────────────────────────────────────────────────

    def _spin_once(self):
        rclpy.spin_once(self, timeout_sec=0.05)

    def _publish_cmd(self, linear_x, angular_z):
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self._cmd_pub.publish(twist)

    def _stop(self):
        self._publish_cmd(0.0, 0.0)

    def _sleep_rate(self):
        time.sleep(1.0 / FREQUENCY)
        self._spin_once()
        self._log_and_publish_pose()
        # publish ideal path 
        if self._ideal_path.poses:
            self._ideal_path.header.stamp = self.get_clock().now().to_msg()
            self._ideal_path_pub.publish(self._ideal_path)

    def _wait_for_sim_ready(self):
        self.get_logger().info('Waiting for simulation to be ready...')
        start = time.monotonic()
        while rclpy.ok():
            self._spin_once()
            if time.monotonic() - start >= STARTUP_TIMEOUT:
                self.get_logger().warn('Startup timeout. Continuing anyway.')
                return
            clock_ready = (not USE_SIM_TIME) or (self.get_clock().now().nanoseconds > 0)
            if clock_ready and self._odom_ready:
                self.get_logger().info('Simulation ready.')
                return

    def _record_start_pose(self):
        self._start_x = self._x
        self._start_y = self._y
        self._start_yaw = self._yaw

    def _reset_logs(self):
        self._traj_log.clear()
        self._waypoint_log.clear()
        self._actual_path = Path()
        self._actual_path.header.frame_id = 'odom'
        self._ideal_path = Path()
        self._ideal_path.header.frame_id = 'odom'

    # ── Logging + RViz ───────────────────────────────────────────────────

    def _log_and_publish_pose(self):
        """Record current pose to CSV log and publish RViz actual path."""
        self._traj_log.append((time.monotonic(), self._x, self._y, self._yaw))

        stamp = self.get_clock().now().to_msg()
        ps = make_pose_stamped(stamp, 'odom', self._x, self._y, self._yaw)
        self._actual_path.poses.append(ps)
        self._actual_path.header.stamp = stamp
        self._actual_path_pub.publish(self._actual_path)

    def _publish_ideal_path(self, waypoints):
        """Publish ideal waypoints as an RViz Path.

        Args:
            waypoints: list of (x, y) or (x, y, label) tuples.
        """
        stamp = self.get_clock().now().to_msg()
        self._ideal_path = Path()
        self._ideal_path.header.frame_id = 'odom'
        self._ideal_path.header.stamp = stamp
        for wp in waypoints:
            x, y = wp[0], wp[1]
            ps = make_pose_stamped(stamp, 'odom', x, y)
            self._ideal_path.poses.append(ps)
        self._ideal_path_pub.publish(self._ideal_path)

    def _save_logs(self, shape_name):
        """Write CSV files for post-run analysis."""
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')

        traj_path = os.path.join(LOG_DIR, f'trajectory_OL_{shape_name}_{ts}.csv')
        with open(traj_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['time_s', 'x', 'y', 'yaw'])
            t0 = self._traj_log[0][0] if self._traj_log else 0.0
            for t, x, y, yaw in self._traj_log:
                w.writerow([f'{t - t0:.4f}', f'{x:.6f}', f'{y:.6f}', f'{yaw:.6f}'])
        self.get_logger().info(f'  Trajectory saved: {traj_path}')

        wp_path = os.path.join(LOG_DIR, f'waypoints_OL_{shape_name}_{ts}.csv')
        with open(wp_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['x', 'y', 'label'])
            for entry in self._waypoint_log:
                x, y, label = entry[0], entry[1], entry[2]
                w.writerow([f'{x:.6f}', f'{y:.6f}', label])
        self.get_logger().info(f'  Waypoints saved:  {wp_path}')

        return traj_path, wp_path

    # ── motion primitives ────────────────────────────────────────────────

    def _turn_to_heading(self, target_yaw):
        """P-controlled in-place rotation with timeout.  ICC: R=0"""
        self.get_logger().info(
            f'  Turning to heading {math.degrees(target_yaw):.1f} deg')
        t0 = time.monotonic()
        MAX_TURN_TIME = 15.0

        while rclpy.ok():
            self._spin_once()
            err = normalize_angle(target_yaw - self._yaw)
            if abs(err) < ANGLE_TOL:
                break
            if time.monotonic() - t0 > MAX_TURN_TIME:
                self.get_logger().warn('  Turn timeout — breaking.')
                break
            omega = max(-ANGULAR_VEL, min(ANGULAR_VEL, 1.5 * err))
            self._publish_cmd(0.0, omega)
            self._sleep_rate()
        self._stop()
        self._sleep_rate()

    def _drive_straight(self, distance):
        """Drive distance metres. ICC at infinity v_l = v_r"""
        self.get_logger().info(f'  Driving straight {distance:.3f} m')
        x0, y0 = self._x, self._y
        while rclpy.ok():
            self._spin_once()
            d = math.hypot(self._x - x0, self._y - y0)
            if d >= abs(distance) - POSITION_TOL:
                break
            vel = LINEAR_VEL if distance > 0 else -LINEAR_VEL
            self._publish_cmd(vel, 0.0)
            self._sleep_rate()
        self._stop()
        self._sleep_rate()

    def _drive_arc(self, radius, arc_angle):
        """Open-loop arc: track yaw swept ICC = (x - R sin θ, y + R cos θ)"""
        if abs(radius) < 1e-6:
            self.get_logger().warn('Arc radius too small, skipping.')
            return

        omega = LINEAR_VEL / abs(radius)
        if arc_angle < 0:
            omega = -omega
        linear_x = omega * radius

        v_l = omega * (radius - WHEEL_BASE / 2.0)
        v_r = omega * (radius + WHEEL_BASE / 2.0)
        self.get_logger().info(
            f'  Arc: R={radius:.3f} m, sweep={math.degrees(arc_angle):.1f} deg, '
            f'v_l={v_l:.3f}, v_r={v_r:.3f}, omega={omega:.3f}')

        target_sweep = abs(arc_angle)
        yaw_prev = self._yaw
        swept = 0.0

        while rclpy.ok():
            self._spin_once()
            d_yaw = normalize_angle(self._yaw - yaw_prev)
            swept += abs(d_yaw)
            yaw_prev = self._yaw
            if swept >= target_sweep - ANGLE_TOL:
                break
            self._publish_cmd(linear_x, omega)
            self._sleep_rate()

        self._stop()
        self._sleep_rate()

    def _navigate_to_point(self, goal_x, goal_y):
        """Navigate to odom-frame point with TF2 + fallback"""
        self.get_logger().info(
            f'  Navigating to ({goal_x:.3f}, {goal_y:.3f}) in odom frame')

        while rclpy.ok():
            self._spin_once()
            dx = goal_x - self._x
            dy = goal_y - self._y
            dist = math.hypot(dx, dy)
            if dist < POSITION_TOL:
                self.get_logger().info('  Reached goal point.')
                break

            local_x, local_y = self._transform_to_base_link(goal_x, goal_y)
            angle_to_goal = math.atan2(local_y, local_x)

            if abs(angle_to_goal) > 0.15:
                target_heading = math.atan2(dy, dx)
                self._turn_to_heading(target_heading)
            else:
                omega = 2.0 * angle_to_goal
                omega = max(-ANGULAR_VEL, min(ANGULAR_VEL, omega))
                v = min(LINEAR_VEL, dist)
                self._publish_cmd(v, omega)
                self._sleep_rate()

        self._stop()
        self._sleep_rate()

    def _transform_to_base_link(self, goal_x, goal_y):
        try:
            trans = self._tf_buffer.lookup_transform(
                'base_link', 'odom', rclpy.time.Time(),
                timeout=Duration(seconds=0.5))
            tx = trans.transform.translation.x
            ty = trans.transform.translation.y
            q = trans.transform.rotation
            tf_yaw = euler_from_quaternion(q)
            cos_y = math.cos(tf_yaw)
            sin_y = math.sin(tf_yaw)
            return cos_y * goal_x + sin_y * goal_y + tx, \
                   -sin_y * goal_x + cos_y * goal_y + ty
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            dx = goal_x - self._x
            dy = goal_y - self._y
            cos_y = math.cos(-self._yaw)
            sin_y = math.sin(-self._yaw)
            return dx * cos_y - dy * sin_y, dx * sin_y + dy * cos_y

    # ══════════════════════════════════════════════════════════════════════
    #   SHAPE 1 Trapezoid
    # ══════════════════════════════════════════════════════════════════════

    def run_trapezoid(self, r):
        self.get_logger().info(f'=== Trapezoid with r = {r:.3f} m ===')
        self._reset_logs()
        self._wait_for_sim_ready()
        self._record_start_pose()

        theta = self._start_yaw
        sx, sy = self._start_x, self._start_y

        tr_x = sx + r * math.cos(theta - math.pi / 4.0)  # top-right
        tr_y = sy + r * math.sin(theta - math.pi / 4.0)
        tl_x = sx + r * math.cos(theta + math.pi / 4.0)  # top-left
        tl_y = sy + r * math.sin(theta + math.pi / 4.0)

        # base vertices along perpendicular to forward
        # trapezoid shape
        br_x = sx + r * math.cos(theta - math.pi / 2.0)  # bottom-right
        br_y = sy + r * math.sin(theta - math.pi / 2.0)
        bl_x = sx + r * math.cos(theta + math.pi / 2.0)  # bottom-left
        bl_y = sy + r * math.sin(theta + math.pi / 2.0)

        # Traverse through start to bottom-right to top-right 
        #                        to top-left to bottom-left to start
        vertices = [
            (br_x, br_y, 'Bottom-Right'),
            (tr_x, tr_y, 'Top-Right'),
            (tl_x, tl_y, 'Top-Left'),
            (bl_x, bl_y, 'Bottom-Left'),
            (sx,   sy,   'Start (close)'),
        ]

        # Log + publish ideal path
        self._waypoint_log.append((sx, sy, 'Start'))
        ideal_pts = [(sx, sy)]
        for vx, vy, name in vertices:
            self._waypoint_log.append((vx, vy, name))
            ideal_pts.append((vx, vy))
        self._publish_ideal_path(ideal_pts)

        self.get_logger().info(
            f'  Start: ({sx:.3f}, {sy:.3f}), heading={math.degrees(theta):.1f} deg')
        for vx, vy, name in vertices:
            self.get_logger().info(f'  {name}: ({vx:.3f}, {vy:.3f})')

        for vx, vy, name in vertices:
            self.get_logger().info(f'--- Heading to {name} ---')
            dx = vx - self._x
            dy = vy - self._y
            target_heading = math.atan2(dy, dx)
            dist = math.hypot(dx, dy)

            turn_angle = normalize_angle(target_heading - self._yaw)
            self.get_logger().info(
                f'  ICC in-place turn: {math.degrees(turn_angle):.1f} deg, '
                f'R_ICC=0, v_l={-ANGULAR_VEL * WHEEL_BASE / 2:.3f}, '
                f'v_r={ANGULAR_VEL * WHEEL_BASE / 2:.3f}')

            self._turn_to_heading(target_heading)
            self._drive_straight(dist)

        self._stop()
        ce = math.hypot(self._x - sx, self._y - sy)
        self.get_logger().info(f'=== Trapezoid complete  |  Closure error: {ce:.4f} m ===')
        self._save_logs('trapezoid')

    # ══════════════════════════════════════════════════════════════════════
    #   SHAPE 2 D-Shape
    # ══════════════════════════════════════════════════════════════════════

    def run_d_shape(self, r):
        """D-shape: straight stroke (2r) + semicircle (radius r).

        Semicircle center = midpoint of stroke 
        """
        self.get_logger().info(f'=== D-Shape with r = {r:.3f} m ===')
        self._reset_logs()
        self._wait_for_sim_ready()
        self._record_start_pose()

        sx, sy, theta = self._start_x, self._start_y, self._start_yaw

        # Stroke end: 2r forward
        ex = sx + 2.0 * r * math.cos(theta)
        ey = sy + 2.0 * r * math.sin(theta)

        # Semicircle center: midpoint 
        cx = sx + r * math.cos(theta)
        cy = sy + r * math.sin(theta)

        # Arc start angle: from center toward stroke-end
        arc_start = math.atan2(ey - cy, ex - cx)  # = theta

        # Ideal path: stroke + semicircle samples
        self._waypoint_log.append((sx, sy, 'Start'))
        self._waypoint_log.append((ex, ey, 'Stroke-End'))
        ideal_pts = [(sx, sy), (ex, ey)]
        n_samples = 30
        for i in range(1, n_samples + 1):
            a = arc_start - i * math.pi / n_samples
            wx = cx + r * math.cos(a)
            wy = cy + r * math.sin(a)
            self._waypoint_log.append((wx, wy, f'Arc-{i}'))
            ideal_pts.append((wx, wy))
        ideal_pts.append((sx, sy))
        self._waypoint_log.append((sx, sy, 'Close'))
        self._publish_ideal_path(ideal_pts)

        # straight stroke
        self.get_logger().info('Step 1: Straight stroke forward 2r')
        self.get_logger().info(
            f'  Linear motion: v_l = v_r = {LINEAR_VEL:.3f}, omega = 0')
        self._drive_straight(2.0 * r)

        # semicircle via waypoints 
        self.get_logger().info('Step 2: Semicircle (transition linear -> circular)')
        R_turn = r
        omega_arc = LINEAR_VEL / R_turn
        v_l = omega_arc * (R_turn + WHEEL_BASE / 2.0)
        v_r = omega_arc * (R_turn - WHEEL_BASE / 2.0)
        self.get_logger().info(
            f'  Circular: center=({cx:.2f},{cy:.2f}), R_ICC={R_turn:.3f}, omega={omega_arc:.3f}')
        self.get_logger().info(
            f'  v_l = {v_l:.3f} (outer), v_r = {v_r:.3f} (inner)')

        # Break semicircle into ~15 deg 
        n_steps = 12   # 180 / 12 = 15 per step
        for i in range(1, n_steps + 1):
            angle = arc_start - i * math.pi / n_steps  # CW sweep
            wx = cx + r * math.cos(angle)
            wy = cy + r * math.sin(angle)
            dx = wx - self._x
            dy = wy - self._y
            dist = math.hypot(dx, dy)
            if dist > POSITION_TOL:
                target_heading = math.atan2(dy, dx)
                self._turn_to_heading(target_heading)
                self._drive_straight(dist)

        self._stop()
        ce = math.hypot(self._x - sx, self._y - sy)
        self.get_logger().info(f'=== D-Shape complete  |  Closure error: {ce:.4f} m ===')
        self._save_logs('d_shape')

    # ══════════════════════════════════════════════════════════════════════
    #   SHAPE 3 General Polygon
    # ══════════════════════════════════════════════════════════════════════

    def run_polygon(self, vertices):
        self.get_logger().info(f'=== Polygon with {len(vertices)} vertices ===')
        self._reset_logs()
        self._wait_for_sim_ready()
        self._record_start_pose()

        ideal_pts = []
        for i, (vx, vy) in enumerate(vertices):
            self._waypoint_log.append((vx, vy, f'V{i}'))
            ideal_pts.append((vx, vy))
            self.get_logger().info(f'  V{i}: ({vx:.3f}, {vy:.3f})')
        ideal_pts.append(vertices[0])
        self._waypoint_log.append((vertices[0][0], vertices[0][1], 'V0-close'))
        self._publish_ideal_path(ideal_pts)

        # _navigate_to_point oscillates on small heading errors
        closed = list(vertices) + [vertices[0]]
        for i, (gx, gy) in enumerate(closed):
            label = f'V{i}' if i < len(vertices) else 'V0 (close)'
            self.get_logger().info(
                f'--- Navigating to {label}: ({gx:.3f}, {gy:.3f}) ---')
            dx = gx - self._x
            dy = gy - self._y
            dist = math.hypot(dx, dy)
            if dist > POSITION_TOL:
                target_heading = math.atan2(dy, dx)
                self._turn_to_heading(target_heading)
                self._drive_straight(dist)

        self._stop()
        ce = math.hypot(self._x - vertices[0][0], self._y - vertices[0][1])
        self.get_logger().info(f'=== Polygon complete  |  Closure error: {ce:.4f} m ===')
        self._save_logs('polygon')


# ══════════════════════════════════════════════════════════════════════════════
#  selection menu
# ══════════════════════════════════════════════════════════════════════════════

def get_user_choice(node):
    print('\n' + '=' * 50)
    print('   TurtleBot3 Shape Follower  [OPEN-LOOP]')
    print('=' * 50)
    print('  1. Trapezoid       (radius r)')
    print('  2. D-Shape         (radius r)')
    print('  3. General Polygon (coordinate list)')
    print('  q. Quit')
    print('=' * 50)

    choice = input('\nChoice (1/2/3/q): ').strip()
    rclpy.spin_once(node, timeout_sec=0.05)

    if choice.lower() == 'q':
        return None, None

    if choice in ('1', '2'):
        r_str = input('Radius r (metres): ').strip()
        rclpy.spin_once(node, timeout_sec=0.05)
        try:
            r = float(r_str)
            if r <= 0:
                print('Must be positive.'); return 'bad', None
            return int(choice), r
        except ValueError:
            print('Invalid number.'); return 'bad', None

    elif choice == '3':
        print('Enter vertices as x,y  (one per line, "done" to finish).')
        verts = []
        while True:
            line = input(f'  V{len(verts)+1} (x,y): ').strip()
            rclpy.spin_once(node, timeout_sec=0.05)
            if line.lower() == 'done':
                break
            try:
                parts = line.replace('(', '').replace(')', '').split(',')
                x, y = float(parts[0]), float(parts[1])
                verts.append((x, y))
                print(f'    Added ({x:.3f},{y:.3f})')
            except (ValueError, IndexError):
                print('    Format: x,y')
        if len(verts) < 3:
            print('Need >= 3 vertices.'); return 'bad', None
        return 3, verts
    else:
        print('Unknown.'); return 'bad', None


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = ShapeFollower()
    try:
        while rclpy.ok():
            choice, param = get_user_choice(node)
            if choice is None:
                print('Goodbye!'); break
            elif choice == 'bad':
                continue
            elif choice == 1:  node.run_trapezoid(param)
            elif choice == 2:  node.run_d_shape(param)
            elif choice == 3:  node.run_polygon(param)
            print('\nDone! Select another shape or quit.\n')
    except KeyboardInterrupt:
        node.get_logger().info('Ctrl+C — stopping.')
        node._stop()
    finally:
        node._stop()
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()