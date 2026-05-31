#!/usr/bin/env python3

# Author: Xiaoming Zhao
# Description: CLOSED-LOOP ROS2 shape follower with /odom feedback
#
#   vs open-loop
#     - PID heading control 
#     - error correction on straight segments
#     - arc following with radial correction
#     - 20 Hz control rate 
#     - tolerances (3 cm / 0.015 rad vs 5 cm / 0.02 rad)
#
# Usage:
#   ros2 launch turtlebot3_gazebo empty_world.launch.py
#   python3 shape_follower_closed_loop.py
#   rviz2  
#
# Logs written to ~/ros2_ws/logs/
#   trajectory_CL_<shape>_<timestamp>.csv
#   waypoints_CL_<shape>_<timestamp>.csv

import math
import csv
import os
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.duration import Duration
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
import tf2_ros

# ── Constants ────────────────────────────────────────────────────────────────
FREQUENCY        = 20       # Hz 
LINEAR_VEL       = 0.15     # m/s
ANGULAR_VEL      = 0.4      # rad/s max
WHEEL_BASE       = 0.287    # m
USE_SIM_TIME     = True
STARTUP_TIMEOUT  = 20.0     # s
POSITION_TOL     = 0.03     # m  
ANGLE_TOL        = 0.015    # rad 

# PID gains for heading controller
KP_HEADING       = 2.0
KI_HEADING       = 0.02
KD_HEADING       = 0.2

# correction gain
KP_CROSS_TRACK   = 0.8       

# arc look-ahead
LOOK_AHEAD_DIST  = 0.10     # m

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

def cross_track_error(px, py, ax, ay, bx, by):
    """Signed perpendicular distance from (px,py) to line A->B.
    Positive = left of the line."""
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab_len = math.hypot(abx, aby)
    if ab_len < 1e-9:
        return 0.0
    return (abx * apy - aby * apx) / ab_len

def point_on_circle(cx, cy, r, angle):
    return cx + r * math.cos(angle), cy + r * math.sin(angle)

def make_pose_stamped(stamp, frame, x, y, yaw=0.0):
    ps = PoseStamped()
    ps.header.stamp = stamp
    ps.header.frame_id = frame
    ps.pose.position.x = x
    ps.pose.position.y = y
    ps.pose.orientation.z = math.sin(yaw / 2.0)
    ps.pose.orientation.w = math.cos(yaw / 2.0)
    return ps


# ── Node ─────────────────────────────────────────────────────────────────────

class ShapeFollowerClosedLoop(Node):

    def __init__(self):
        super().__init__('shape_follower_closed_loop')

        self.set_parameters([rclpy.parameter.Parameter(
            'use_sim_time', rclpy.Parameter.Type.BOOL, USE_SIM_TIME)])

        # Publishers
        self._cmd_pub = self.create_publisher(Twist, 'cmd_vel', 1)
        self._actual_path_pub = self.create_publisher(Path, 'actual_path_cl', 10)
        self._ideal_path_pub  = self.create_publisher(Path, 'ideal_path_cl', 10)

        # Subscribers
        self._odom_sub = self.create_subscription(
            Odometry, 'odom', self._odom_cb, 10)

        # TF2
        self._tf_buf = tf2_ros.Buffer()
        self._tf_lis = tf2_ros.TransformListener(self._tf_buf, self)

        # State
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._odom_ready = False
        self._sx = self._sy = self._syaw = 0.0

        # PID state
        self._heading_integral = 0.0
        self._heading_prev_err = 0.0

        # RViz paths
        self._actual_path = Path()
        self._actual_path.header.frame_id = 'odom'
        self._ideal_path = Path()
        self._ideal_path.header.frame_id = 'odom'

        # CSV
        self._traj_log = []
        self._waypoint_log = []

        os.makedirs(LOG_DIR, exist_ok=True)

    # ── Callbacks ────────────────────────────────────────────────────────

    def _odom_cb(self, msg):
        self._x = msg.pose.pose.position.x
        self._y = msg.pose.pose.position.y
        self._yaw = euler_from_quaternion(msg.pose.pose.orientation)
        self._odom_ready = True

    # ── Utility ──────────────────────────────────────────────────────────

    def _spin(self):
        rclpy.spin_once(self, timeout_sec=0.02)

    def _cmd(self, v, w):
        t = Twist(); t.linear.x = v; t.angular.z = w
        self._cmd_pub.publish(t)

    def _stop(self):
        self._cmd(0.0, 0.0)

    def _tick(self):
        time.sleep(1.0 / FREQUENCY)
        self._spin()
        self._log_and_publish_pose()
        # publish ideal path 
        if self._ideal_path.poses:
            self._ideal_path.header.stamp = self.get_clock().now().to_msg()
            self._ideal_path_pub.publish(self._ideal_path)

    def _wait_sim(self):
        self.get_logger().info('Waiting for simulation ...')
        t0 = time.monotonic()
        while rclpy.ok():
            self._spin()
            if time.monotonic() - t0 >= STARTUP_TIMEOUT:
                self.get_logger().warn('Startup timeout.'); return
            ok_clk = (not USE_SIM_TIME) or self.get_clock().now().nanoseconds > 0
            if ok_clk and self._odom_ready:
                self.get_logger().info('Simulation ready.'); return

    def _record_start(self):
        self._sx, self._sy, self._syaw = self._x, self._y, self._yaw

    def _reset_pid(self):
        self._heading_integral = 0.0
        self._heading_prev_err = 0.0

    def _reset_logs(self):
        self._traj_log.clear()
        self._waypoint_log.clear()
        self._actual_path = Path()
        self._actual_path.header.frame_id = 'odom'
        self._ideal_path = Path()
        self._ideal_path.header.frame_id = 'odom'

    # ── Logging + RViz ───────────────────────────────────────────────────

    def _log_and_publish_pose(self):
        self._traj_log.append((time.monotonic(), self._x, self._y, self._yaw))
        stamp = self.get_clock().now().to_msg()
        ps = make_pose_stamped(stamp, 'odom', self._x, self._y, self._yaw)
        self._actual_path.poses.append(ps)
        self._actual_path.header.stamp = stamp
        self._actual_path_pub.publish(self._actual_path)

    def _publish_ideal_path(self, waypoints):
        stamp = self.get_clock().now().to_msg()
        self._ideal_path = Path()
        self._ideal_path.header.frame_id = 'odom'
        self._ideal_path.header.stamp = stamp
        for wp in waypoints:
            ps = make_pose_stamped(stamp, 'odom', wp[0], wp[1])
            self._ideal_path.poses.append(ps)
        self._ideal_path_pub.publish(self._ideal_path)

    def _save_logs(self, shape_name):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')

        traj_path = os.path.join(LOG_DIR, f'trajectory_CL_{shape_name}_{ts}.csv')
        with open(traj_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['time_s', 'x', 'y', 'yaw'])
            t0 = self._traj_log[0][0] if self._traj_log else 0.0
            for t, x, y, yaw in self._traj_log:
                w.writerow([f'{t - t0:.4f}', f'{x:.6f}', f'{y:.6f}', f'{yaw:.6f}'])
        self.get_logger().info(f'  Trajectory saved: {traj_path}')

        wp_path = os.path.join(LOG_DIR, f'waypoints_CL_{shape_name}_{ts}.csv')
        with open(wp_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['x', 'y', 'label'])
            for entry in self._waypoint_log:
                w.writerow([f'{entry[0]:.6f}', f'{entry[1]:.6f}', entry[2]])
        self.get_logger().info(f'  Waypoints saved:  {wp_path}')

    # ── PID heading ──────────────────────────────────────────────────────

    def _pid_heading(self, target_yaw, dt=None):
        if dt is None: dt = 1.0 / FREQUENCY
        err = normalize_angle(target_yaw - self._yaw)
        self._heading_integral += err * dt
        self._heading_integral = max(-1.0, min(1.0, self._heading_integral))
        d_err = (err - self._heading_prev_err) / dt if dt > 0 else 0.0
        self._heading_prev_err = err
        omega = KP_HEADING * err + KI_HEADING * self._heading_integral + KD_HEADING * d_err
        return max(-ANGULAR_VEL, min(ANGULAR_VEL, omega))

    # ── closed loop motion primitives ────────────────────────────────────

    def _turn_to_heading_cl(self, target_yaw):
        """PID in-place rotation.  ICC: R=0."""
        self.get_logger().info(
            f'  [CL] Turn to {math.degrees(target_yaw):.1f} deg')
        self._reset_pid()
        while rclpy.ok():
            self._spin()
            err = normalize_angle(target_yaw - self._yaw)
            if abs(err) < ANGLE_TOL: break
            self._cmd(0.0, self._pid_heading(target_yaw))
            self._tick()
        self._stop(); self._tick()

    def _drive_straight_cl(self, ax, ay, bx, by):
        """Drive A->B with cross-track error correction every cycle.

        Control law:
            cte = signed perpendicular distance to line A->B
            correction = atan(K_ct * cte)   (bounded to ±pi/2)
            heading_desired = bearing_to_B + correction
        """
        dist_total = math.hypot(bx - ax, by - ay)
        self.get_logger().info(
            f'  [CL] Straight {dist_total:.3f} m  '
            f'({ax:.2f},{ay:.2f})->({bx:.2f},{by:.2f})')
        self._reset_pid()

        while rclpy.ok():
            self._spin()
            dx, dy = bx - self._x, by - self._y
            dist_remain = math.hypot(dx, dy)
            if dist_remain < POSITION_TOL: break

            cte = cross_track_error(self._x, self._y, ax, ay, bx, by)
            bearing = math.atan2(dy, dx)
            # Use atan to bound the correction angle to ±~38° max
            correction = math.atan(KP_CROSS_TRACK * cte)
            heading_desired = bearing + correction

            omega = self._pid_heading(heading_desired)
            v = min(LINEAR_VEL, dist_remain * 2.0)
            v = max(0.02, v)

            self._cmd(v, omega)
            self._tick()

        self._stop(); self._tick()

    def _drive_arc_cl(self, cx, cy, radius, start_angle, sweep):
        """Follow a circular arc by tracking sequential waypoints on the circle.

        breaks the arc into small angular steps and navigates to each
        waypoint in sequence using _drive_straight_cl

        Args:
            cx, cy:       Center of the arc in odom frame
            radius:       Radius 
            start_angle:  Starting angle on the circle 
            sweep:        Signed sweep angle (positive=CCW, negative=CW)
        """
        self.get_logger().info(
            f'  [CL] Arc: center=({cx:.2f},{cy:.2f}), R={radius:.3f}, '
            f'sweep={math.degrees(sweep):.1f} deg')

        # break arc into small segments 
        n_steps = max(int(abs(sweep) / (math.pi / 12.0)), 6)
        d_angle = sweep / n_steps

        for i in range(1, n_steps + 1):
            angle = start_angle + i * d_angle
            wx, wy = point_on_circle(cx, cy, radius, angle)

            dx, dy = wx - self._x, wy - self._y
            dist = math.hypot(dx, dy)
            if dist > POSITION_TOL:
                target_h = math.atan2(dy, dx)
                self._turn_to_heading_cl(target_h)
                self._drive_straight_cl(self._x, self._y, wx, wy)

        self._stop(); self._tick()

    def _to_base_link(self, gx, gy):
        try:
            tr = self._tf_buf.lookup_transform(
                'base_link', 'odom', rclpy.time.Time(),
                timeout=Duration(seconds=0.3))
            tx, ty = tr.transform.translation.x, tr.transform.translation.y
            yaw = euler_from_quaternion(tr.transform.rotation)
            c, s = math.cos(yaw), math.sin(yaw)
            return c * gx + s * gy + tx, -s * gx + c * gy + ty
        except Exception:
            dx, dy = gx - self._x, gy - self._y
            c, s = math.cos(-self._yaw), math.sin(-self._yaw)
            return dx * c - dy * s, dx * s + dy * c

    # ══════════════════════════════════════════════════════════════════════
    #   SHAPE 1 Trapezoid 
    # ══════════════════════════════════════════════════════════════════════

    def run_trapezoid(self, r):
        self.get_logger().info(f'=== [CL] Trapezoid r={r:.3f} ===')
        self._reset_logs()
        self._wait_sim()
        self._record_start()

        theta = self._syaw
        sx, sy = self._sx, self._sy

        tr_x = sx + r * math.cos(theta - math.pi / 4)  # top-right
        tr_y = sy + r * math.sin(theta - math.pi / 4)
        tl_x = sx + r * math.cos(theta + math.pi / 4)  # top-left
        tl_y = sy + r * math.sin(theta + math.pi / 4)

        # half-width = r 
        br_x = sx + r * math.cos(theta - math.pi / 2)  # bottom-right
        br_y = sy + r * math.sin(theta - math.pi / 2)
        bl_x = sx + r * math.cos(theta + math.pi / 2)  # bottom-left
        bl_y = sy + r * math.sin(theta + math.pi / 2)

        # Perimeter order: no crossings
        verts = [
            (br_x, br_y, 'Bot-R'),
            (tr_x, tr_y, 'Top-R'),
            (tl_x, tl_y, 'Top-L'),
            (bl_x, bl_y, 'Bot-L'),
            (sx,   sy,   'Start'),
        ]

        self._waypoint_log.append((sx, sy, 'Start'))
        ideal_pts = [(sx, sy)]
        for vx, vy, name in verts:
            self._waypoint_log.append((vx, vy, name))
            ideal_pts.append((vx, vy))
        self._publish_ideal_path(ideal_pts)

        self.get_logger().info(
            f'  Start: ({sx:.3f},{sy:.3f}) hdg={math.degrees(theta):.1f}')
        for vx, vy, n in verts:
            self.get_logger().info(f'  {n}: ({vx:.3f},{vy:.3f})')

        for vx, vy, name in verts:
            self.get_logger().info(f'--- To {name} ---')
            dx, dy = vx - self._x, vy - self._y
            target_h = math.atan2(dy, dx)
            turn = normalize_angle(target_h - self._yaw)
            self.get_logger().info(
                f'  ICC turn: {math.degrees(turn):.1f} deg, R=0, '
                f'v_l={-ANGULAR_VEL*WHEEL_BASE/2:.3f}, '
                f'v_r={ANGULAR_VEL*WHEEL_BASE/2:.3f}')
            self._turn_to_heading_cl(target_h)
            self._drive_straight_cl(self._x, self._y, vx, vy)

        self._stop()
        ce = math.hypot(self._x - sx, self._y - sy)
        self.get_logger().info(f'=== [CL] Trapezoid complete  |  Closure error: {ce:.4f} m ===')
        self._save_logs('trapezoid')

    # ══════════════════════════════════════════════════════════════════════
    #   SHAPE 2 D-Shape 
    # ══════════════════════════════════════════════════════════════════════

    def run_d_shape(self, r):
        """Closed-loop D-shape: straight stroke (2r) + semicircle (radius r).

        Geometry:
            - Stroke: from start S to end E, length 2r along heading.
            - Semicircle center: midpoint M = S + r * forward
            - Semicircle radius: r
            - Stroke end E is on the circle at angle = heading direction
            - Start S is on the circle at angle = heading + 180°
            - Semicircle sweeps from E back to S (clockwise = -π)

        This forms a proper "D" shape: | + )
        """
        self.get_logger().info(f'=== [CL] D-Shape r={r:.3f} ===')
        self._reset_logs()
        self._wait_sim()
        self._record_start()

        sx, sy, theta = self._sx, self._sy, self._syaw

        # 2r forward from start
        ex = sx + 2.0 * r * math.cos(theta)
        ey = sy + 2.0 * r * math.sin(theta)

        # midpoint of stroke 
        cx = sx + r * math.cos(theta)
        cy = sy + r * math.sin(theta)

        # direction from center to stroke-end
        arc_start = math.atan2(ey - cy, ex - cx)  # = theta

        # stroke + semicircle samples + close
        self._waypoint_log.append((sx, sy, 'Start'))
        self._waypoint_log.append((ex, ey, 'Stroke-End'))
        ideal_pts = [(sx, sy), (ex, ey)]
        n_samples = 30
        for i in range(1, n_samples + 1):
            a = arc_start - i * math.pi / n_samples  # CW sweep
            wx, wy = point_on_circle(cx, cy, r, a)
            self._waypoint_log.append((wx, wy, f'Arc-{i}'))
            ideal_pts.append((wx, wy))
        ideal_pts.append((sx, sy))
        self._waypoint_log.append((sx, sy, 'Close'))
        self._publish_ideal_path(ideal_pts)

        # straight stroke with cross correction
        self.get_logger().info('Step 1: Straight stroke (2r)')
        self.get_logger().info(f'  Linear: v_l=v_r={LINEAR_VEL:.3f}, omega=0')
        self._drive_straight_cl(sx, sy, ex, ey)

        # Ssemicircle center=midpoint, radius=r, CW
        self.get_logger().info('Step 2: Semicircle (transition linear -> circular)')
        omega_arc = LINEAR_VEL / r
        v_l = omega_arc * (r + WHEEL_BASE / 2.0)
        v_r = omega_arc * (r - WHEEL_BASE / 2.0)
        self.get_logger().info(
            f'  Circular: center=({cx:.2f},{cy:.2f}), R_ICC={r:.3f}, '
            f'omega={omega_arc:.3f}, v_l={v_l:.3f}, v_r={v_r:.3f}')
        self._drive_arc_cl(cx=cx, cy=cy, radius=r,
                           start_angle=arc_start, sweep=-math.pi)

        self._stop()
        ce = math.hypot(self._x - sx, self._y - sy)
        self.get_logger().info(f'=== [CL] D-Shape complete  |  Closure error: {ce:.4f} m ===')
        self._save_logs('d_shape')

    # ══════════════════════════════════════════════════════════════════════
    #   SHAPE 3 General Polygon 
    # ══════════════════════════════════════════════════════════════════════

    def run_polygon(self, vertices):
        self.get_logger().info(f'=== [CL] Polygon, {len(vertices)} vertices ===')
        self._reset_logs()
        self._wait_sim()
        self._record_start()

        ideal_pts = []
        for i, (vx, vy) in enumerate(vertices):
            self._waypoint_log.append((vx, vy, f'V{i}'))
            ideal_pts.append((vx, vy))
            self.get_logger().info(f'  V{i}: ({vx:.3f},{vy:.3f})')
        ideal_pts.append(vertices[0])
        self._waypoint_log.append((vertices[0][0], vertices[0][1], 'V0-close'))
        self._publish_ideal_path(ideal_pts)

        closed = list(vertices) + [vertices[0]]
        for i, (gx, gy) in enumerate(closed):
            label = f'V{i}' if i < len(vertices) else 'V0 (close)'
            self.get_logger().info(f'--- To {label}: ({gx:.3f},{gy:.3f}) ---')
            dx, dy = gx - self._x, gy - self._y
            dist = math.hypot(dx, dy)
            if dist > POSITION_TOL:
                target_h = math.atan2(dy, dx)
                self._turn_to_heading_cl(target_h)
                self._drive_straight_cl(self._x, self._y, gx, gy)

        self._stop()
        ce = math.hypot(self._x - vertices[0][0], self._y - vertices[0][1])
        self.get_logger().info(f'=== [CL] Polygon complete  |  Closure error: {ce:.4f} m ===')
        self._save_logs('polygon')


# ══════════════════════════════════════════════════════════════════════════════
#  selection menu
# ══════════════════════════════════════════════════════════════════════════════

def get_user_choice(node):
    print('\n' + '=' * 50)
    print('  TurtleBot3 Shape Follower  [CLOSED-LOOP]')
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
    node = ShapeFollowerClosedLoop()
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