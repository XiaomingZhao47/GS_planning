#!/usr/bin/env python3
"""
PA3: Occupancy-Grid Path Planner for ROS 2

Coordinate frames are resolved with tf2

  ros2 run nav2_map_server map_server --ros-args -p yaml_filename:=src/pa3/maze.yml
  ros2 run nav2_lifecycle_manager lifecycle_manager \
       --ros-args -p autostart:=true -p node_names:=[map_server]
  python3 src/pa3/path_planner.py            
"""

import heapq
import math
import threading
from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from geometry_msgs.msg import PoseArray, Pose, Twist, Quaternion
from nav_msgs.msg import OccupancyGrid

import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


# ── Helpers ─────────────────────────────────────────────────────────────────

def yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


def quat_to_yaw(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def gaussian_kernel(size: int) -> np.ndarray:
    """2D Gaussian kernel with sigma chosen from kernel size"""
    if size not in (3, 5):
        raise ValueError("blur_kernel must be 3 or 5")
    sigma = (size - 1) / 4.0 + 0.3      # OpenCV convention
    ax = np.arange(size) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma * sigma))
    return k / k.sum()


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square dilation of a boolean mask by radius cells

    Used to inflate obstacles by the robot's radius before planning so the
    path stays clear of walls regardless of which algorithm 
    """
    if radius <= 0:
        return mask.copy()
    h, w = mask.shape
    pad = radius
    padded = np.pad(mask.astype(np.uint8), pad, mode="constant", constant_values=0)
    out = np.zeros_like(mask, dtype=bool)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            out |= padded[pad + dr:pad + dr + h, pad + dc:pad + dc + w].astype(bool)
    return out


def gaussian_blur(occ: np.ndarray, kernel: np.ndarray, unknown_mask: np.ndarray) -> np.ndarray:
    """convolve occ with kernel"""
    h, w = occ.shape
    pad = kernel.shape[0] // 2

    weights = (~unknown_mask).astype(np.float32)
    occ_f = occ.astype(np.float32) * weights

    occ_pad = np.pad(occ_f, pad, mode="edge")
    w_pad = np.pad(weights, pad, mode="edge")

    out = np.zeros_like(occ_f)
    norm = np.zeros_like(occ_f)
    ks = kernel.shape[0]
    for i in range(ks):
        for j in range(ks):
            k = kernel[i, j]
            out += k * occ_pad[i:i + h, j:j + w]
            norm += k * w_pad[i:i + h, j:j + w]

    safe = norm > 1e-6
    blurred = np.where(safe, out / np.where(safe, norm, 1.0), occ_f)
    blurred[unknown_mask] = -1.0
    return blurred


# ── Planner ─────────────────────────────────────────────────────────────────

class GridPlanner:
    """grid planner"""

    NEIGH_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    NEIGH_8 = NEIGH_4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    def __init__(self, cost_grid: np.ndarray, connectivity: int = 8):
        if connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")
        self.cost = cost_grid
        self.h, self.w = cost_grid.shape
        self.neigh = self.NEIGH_8 if connectivity == 8 else self.NEIGH_4

    def _passable(self, r: int, c: int) -> bool:
        return 0 <= r < self.h and 0 <= c < self.w and math.isfinite(self.cost[r, c])

    def _passable_step(self, r: int, c: int, dr: int, dc: int) -> bool:
        """reject diagonals that cut a corner"""
        if not self._passable(r + dr, c + dc):
            return False
        if dr != 0 and dc != 0:
            if not (self._passable(r + dr, c) and self._passable(r, c + dc)):
                return False
        return True

    def _step_cost(self, r: int, c: int, dr: int, dc: int, weighted: bool) -> float:
        base = math.hypot(dr, dc)                    # 1.0 or sqrt of 2
        if weighted:
            base *= 1.0 + self.cost[r + dr, c + dc]  # prefer corridor center
        return base

    @staticmethod
    def _reconstruct(came_from, end):
        path = [end]
        while path[-1] in came_from:
            path.append(came_from[path[-1]])
        path.reverse()
        return path

    # ── BFS 
    def bfs(self, start, goal):
        if not (self._passable(*start) and self._passable(*goal)):
            return None
        visited = {start}
        came_from = {}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur == goal:
                return self._reconstruct(came_from, goal)
            for dr, dc in self.neigh:
                if not self._passable_step(cur[0], cur[1], dr, dc):
                    continue
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt in visited:
                    continue
                visited.add(nxt)
                came_from[nxt] = cur
                q.append(nxt)
        return None

    # ── DFS 
    def dfs(self, start, goal):
        if not (self._passable(*start) and self._passable(*goal)):
            return None
        visited = {start}
        came_from = {}
        stack = [start]
        while stack:
            cur = stack.pop()
            if cur == goal:
                return self._reconstruct(came_from, goal)
            for dr, dc in self.neigh:
                if not self._passable_step(cur[0], cur[1], dr, dc):
                    continue
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt in visited:
                    continue
                visited.add(nxt)
                came_from[nxt] = cur
                stack.append(nxt)
        return None

    # ── A* 
    def astar(self, start, goal, weighted: bool = True):
        if not (self._passable(*start) and self._passable(*goal)):
            return None

        if len(self.neigh) == 4:
            def h(n):                                    # manhattan
                return abs(n[0] - goal[0]) + abs(n[1] - goal[1])
        else:
            def h(n):                                    # octile, admissible for 8-conn
                dr = abs(n[0] - goal[0]); dc = abs(n[1] - goal[1])
                return (dr + dc) + (math.sqrt(2) - 2) * min(dr, dc)

        g_score = {start: 0.0}
        came_from = {}
        counter = 0
        pq = [(h(start), counter, start)]
        closed = set()

        while pq:
            _, _, cur = heapq.heappop(pq)
            if cur in closed:
                continue
            if cur == goal:
                return self._reconstruct(came_from, goal)
            closed.add(cur)
            for dr, dc in self.neigh:
                if not self._passable_step(cur[0], cur[1], dr, dc):
                    continue
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt in closed:
                    continue
                tentative = g_score[cur] + self._step_cost(cur[0], cur[1], dr, dc, weighted)
                if tentative < g_score.get(nxt, math.inf):
                    g_score[nxt] = tentative
                    came_from[nxt] = cur
                    counter += 1
                    heapq.heappush(pq, (tentative + h(nxt), counter, nxt))
        return None


# ── Node ────────────────────────────────────────────────────────────────────

class PathPlannerNode(Node):
    def __init__(self):
        super().__init__("pa3_path_planner")

        # ── parameters
        self.declare_parameter("algorithm", "astar")           # bfs | dfs | astar
        self.declare_parameter("connectivity", 8)              # 4 | 8
        self.declare_parameter("weighted", True)               # weighted A*
        self.declare_parameter("blur_kernel", 5)               # 3 | 5
        self.declare_parameter("obstacle_thresh", 50)          
        self.declare_parameter("inflate_thresh", 30)           
        self.declare_parameter("inflate_radius", 4)            
                                                                
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("waypoint_tol", 0.15)           # m
        self.declare_parameter("goal_tol", 0.10)               # m
        self.declare_parameter("k_lin", 0.5)
        self.declare_parameter("k_ang", 1.5)
        self.declare_parameter("v_max", 0.25)
        self.declare_parameter("w_max", 1.2)
        self.declare_parameter("control_rate", 20.0)
        self.declare_parameter("path_stride", 1)               # downsample published path

        # ── state
        self.raw_map: OccupancyGrid | None = None
        self.smoothed_map: OccupancyGrid | None = None
        self.cost_grid: np.ndarray | None = None
        self.path_world: list[tuple[float, float]] = []
        self.current_wp_idx = 0
        
        self.goal_active = False
        self.goal_xy: tuple[float, float] | None = None
        self._tick_count = 0
        self._lock = threading.Lock()

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
        self.create_subscription(OccupancyGrid, "/map", self.map_cb, latched)
        self.smoothed_pub = self.create_publisher(OccupancyGrid, "/map_smoothed", latched)
        self.path_pub = self.create_publisher(PoseArray, "/pose_sequence", 10)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # ── control loop
        rate = self.get_parameter("control_rate").value
        self.create_timer(1.0 / rate, self.control_loop)

        # ── goal-prompt thread
        threading.Thread(target=self._goal_prompt_loop, daemon=True).start()

        self.get_logger().info("pa3_path_planner ready — waiting for /map")

    # ── map callback; blur, publish, build cost grid
    def map_cb(self, msg: OccupancyGrid):
        h, w = msg.info.height, msg.info.width
        data = np.array(msg.data, dtype=np.int16).reshape(h, w)
        unknown = data < 0
        occ = np.where(unknown, 0, data).astype(np.float32)

        ksz = int(self.get_parameter("blur_kernel").value)
        kernel = gaussian_kernel(ksz)
        blurred = gaussian_blur(occ, kernel, unknown)

        smoothed = OccupancyGrid()
        smoothed.header = msg.header
        smoothed.info = msg.info
        out = np.where(unknown, -1, np.clip(np.round(blurred), 0, 100)).astype(np.int8)
        smoothed.data = out.flatten().tolist()

        # cost grid for the planner
        obs = int(self.get_parameter("obstacle_thresh").value)
        infl = int(self.get_parameter("inflate_thresh").value)
        radius = int(self.get_parameter("inflate_radius").value)

        obstacle_mask = unknown | (data >= obs) | (blurred >= infl)
        obstacle_mask = dilate_mask(obstacle_mask, radius)
        cost = np.where(obstacle_mask, np.inf, blurred / 100.0).astype(np.float32)

        with self._lock:
            self.raw_map = msg
            self.smoothed_map = smoothed
            self.cost_grid = cost

        self.smoothed_pub.publish(smoothed)
        self.get_logger().info(
            f"map received {w}×{h}, res={msg.info.resolution:.3f} m — /map_smoothed published")

    # ── coordinate transforms
    def world_to_grid(self, x: float, y: float):
        info = self.raw_map.info
        col = int((x - info.origin.position.x) / info.resolution)
        row = int((y - info.origin.position.y) / info.resolution)
        return row, col

    def grid_to_world(self, row: int, col: int):
        info = self.raw_map.info
        x = info.origin.position.x + (col + 0.5) * info.resolution
        y = info.origin.position.y + (row + 0.5) * info.resolution
        return x, y

    def lookup_robot_xy_yaw(self):
        map_frame = self.get_parameter("map_frame").value
        base_frame = self.get_parameter("base_frame").value
        try:
            tf = self.tf_buffer.lookup_transform(
                map_frame, base_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5))
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"tf lookup {map_frame}->{base_frame} failed: {e}")
            return None
        t = tf.transform.translation
        return t.x, t.y, quat_to_yaw(tf.transform.rotation)

    # ── plan
    def plan_to(self, gx: float, gy: float):
        with self._lock:
            grid = self.cost_grid
            raw = self.raw_map
        if grid is None or raw is None:
            self.get_logger().warn("no map yet — cannot plan")
            return

        rxyy = self.lookup_robot_xy_yaw()
        if rxyy is None:
            return
        rx, ry, _ = rxyy

        start = self.world_to_grid(rx, ry)
        goal = self.world_to_grid(gx, gy)

        algo = self.get_parameter("algorithm").value.lower()
        conn = int(self.get_parameter("connectivity").value)
        weighted = bool(self.get_parameter("weighted").value)

        planner = GridPlanner(grid, connectivity=conn)
        self.get_logger().info(
            f"planning {algo}/{conn}-conn weighted={weighted}  "
            f"start={start} goal={goal}")

        if algo == "bfs":
            cells = planner.bfs(start, goal)
        elif algo == "dfs":
            cells = planner.dfs(start, goal)
        elif algo == "astar":
            cells = planner.astar(start, goal, weighted=weighted)
        else:
            self.get_logger().error(f"unknown algorithm '{algo}'")
            return

        if cells is None:
            self.get_logger().warn("no path found")
            return

        stride = max(1, int(self.get_parameter("path_stride").value))
        sampled = cells[::stride]
        if sampled[-1] != cells[-1]:
            sampled.append(cells[-1])
        world = [self.grid_to_world(r, c) for (r, c) in sampled]

        # publish PoseArray with yaw to next waypoint
        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = self.get_parameter("map_frame").value
        for i, (x, y) in enumerate(world):
            nx, ny = world[i + 1] if i + 1 < len(world) else world[i]
            yaw = math.atan2(ny - y, nx - x) if i + 1 < len(world) else \
                  math.atan2(world[-1][1] - world[-2][1], world[-1][0] - world[-2][0]) \
                  if len(world) >= 2 else 0.0
            p = Pose()
            p.position.x = x
            p.position.y = y
            p.orientation = yaw_to_quat(yaw)
            pa.poses.append(p)
        self.path_pub.publish(pa)

        with self._lock:
            self.path_world = world
            self.current_wp_idx = 0
            self.goal_active = True
            self.goal_xy = (float(gx), float(gy))
        self.get_logger().info(f"path with {len(world)} waypoints published on /pose_sequence")

    # ── controller
    def control_loop(self):
        with self._lock:
            active = self.goal_active
            path = list(self.path_world)
            idx = self.current_wp_idx
            goal_xy = self.goal_xy
        if not active or not path:
            return

        rxyy = self.lookup_robot_xy_yaw()
        if rxyy is None:
            return
        rx, ry, ryaw = rxyy

        # advance to next waypoint when close
        wp_tol = self.get_parameter("waypoint_tol").value
        goal_tol = self.get_parameter("goal_tol").value
        while idx < len(path) - 1 and math.hypot(path[idx][0] - rx, path[idx][1] - ry) < wp_tol:
            idx += 1

        tx, ty = path[idx]
        dx, dy = tx - rx, ty - ry
        dist = math.hypot(dx, dy)

        # ── pose-vs-goal log (throttled to once per ~1 s)
        rate = float(self.get_parameter("control_rate").value)
        log_every = max(1, int(rate))                 # ≈ 1 Hz
        self._tick_count += 1
        if goal_xy is not None and self._tick_count % log_every == 0:
            gx, gy = goal_xy
            goal_dist = math.hypot(gx - rx, gy - ry)
            self.get_logger().info(
                f"pose=({rx:+.2f}, {ry:+.2f}, {math.degrees(ryaw):+6.1f}°)  "
                f"goal=({gx:+.2f}, {gy:+.2f})  "
                f"dist_to_goal={goal_dist:.2f} m  "
                f"wp {idx + 1}/{len(path)} (dist={dist:.2f} m)")

        if idx == len(path) - 1 and dist < goal_tol:
            self.cmd_pub.publish(Twist())
            with self._lock:
                self.goal_active = False
                self.current_wp_idx = 0
            self.get_logger().info(
                f"goal reached  pose=({rx:+.2f}, {ry:+.2f})  "
                f"goal=({goal_xy[0]:+.2f}, {goal_xy[1]:+.2f})" if goal_xy else "goal reached")
            return

        with self._lock:
            self.current_wp_idx = idx

        desired_yaw = math.atan2(dy, dx)
        yaw_err = math.atan2(math.sin(desired_yaw - ryaw), math.cos(desired_yaw - ryaw))

        k_lin = self.get_parameter("k_lin").value
        k_ang = self.get_parameter("k_ang").value
        v_max = self.get_parameter("v_max").value
        w_max = self.get_parameter("w_max").value

        cmd = Twist()
        if abs(yaw_err) > 0.4:
            cmd.linear.x = 0.0
        else:
            cmd.linear.x = max(0.0, min(v_max, k_lin * dist * math.cos(yaw_err)))
        cmd.angular.z = max(-w_max, min(w_max, k_ang * yaw_err))
        self.cmd_pub.publish(cmd)

    def _goal_prompt_loop(self):
        while rclpy.ok():
            try:
                line = input("\nEnter goal as 'x y' in map frame (or 'q' to quit): ").strip()
            except EOFError:
                return
            if not line:
                continue
            if line.lower() in ("q", "quit", "exit"):
                rclpy.shutdown()
                return
            try:
                gx, gy = (float(t) for t in line.split()[:2])
            except ValueError:
                print("  ✗ expected two numbers, e.g.  7.5 3.2")
                continue
            self.plan_to(gx, gy)


def main():
    rclpy.init()
    node = PathPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
