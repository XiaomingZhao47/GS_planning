# PA3

## Files

- **`bringup.launch.py`** — starts Stage with `maze.world`, the local map publisher, the static `map to odom` TF, and RViz
- **`map_publisher.py`** — reads `maze.yml` + `maze.png` and publishes `nav_msgs/OccupancyGrid` once on `/map` with `TRANSIENT_LOCAL`
- **`path_planner.py`** — the planner node. Subscribes to `/map`, blurs and inflates it, runs BFS / DFS / A*, publishes `/map_smoothed` and `/pose_sequence`, and drives the robot via `/cmd_vel`. Goal is read from stdin; algorithm is selected via `--ros-args -p algorithm:=...`.

## Usage

Source the workspace, then in two terminals

```bash
# Terminal 1 — sim + map + TF + RViz
ros2 launch /home/xiaoming/ros2_ws/src/pa3/bringup.launch.py

# Terminal 2 — planner
python3 /home/xiaoming/ros2_ws/src/pa3/path_planner.py --ros-args \
  -p algorithm:=astar -p connectivity:=8 -p weighted:=true \
  -p blur_kernel:=5 -p obstacle_thresh:=65 \
  -p inflate_thresh:=10 -p inflate_radius:=4 -p path_stride:=3
```


## `path_planner.py` parameters

| Param | Default |  |
|---|---|---|
| `algorithm` | `astar` | Search algorithm: `bfs` \| `dfs` \| `astar`. |
| `connectivity` | `8` | Grid connectivity: `4` cardinals only or `8` cardinals + diagonals |
| `weighted` | `true` | A\* only, multiply step cost by `(1 + blurred/100)` to prefer corridor centres |
| `blur_kernel` | `5` | Gaussian kernel size for `/map_smoothed`: `3` or `5`. |
| `obstacle_thresh` | `50` | Raw cell value `≥` this → hard obstacle. |
| `inflate_thresh` | `30` | Blurred value `≥` this to hard obstacle  |
| `inflate_radius` | `4` | Cells of hard dilation around every obstacle  |
| `map_frame` | `map` | Target frame for `lookup_transform` |
| `base_frame` | `base_link` | Source frame for `lookup_transform` |
| `waypoint_tol` | `0.15` m | Advance to next waypoint when within this distance |
| `goal_tol` | `0.10` m | Goal considered reached when within this distance |
| `k_lin` | `0.5` | Linear velocity proportional gain |
| `k_ang` | `1.5` | Angular velocity proportional gain |
| `v_max` | `0.25` m/s | Linear velocity saturation. |
| `w_max` | `1.2` rad/s | Angular velocity saturation |
| `control_rate` | `20.0` Hz | Controller / pose-log tick rate |
| `path_stride` | `1` | Downsample factor for the published `PoseArray` |

## `map_publisher.py` parameters

| Param | Default |  |
|---|---|---|
| `yaml_filename` | *(required)* | Absolute path to a ROS map YAML (`image`, `resolution`, `origin`, thresholds) |
| `frame_id` | `map` | Frame stamped on the published `OccupancyGrid` |
| `topic` | `/map` | Topic to publish on (`TRANSIENT_LOCAL`) |

## `bringup.launch.py` arguments

| Arg | Default |  |
|---|---|---|
| `world` | `<this_dir>/maze` | Stage world path **without** `.world` extension. |
| `map_yaml` | `<this_dir>/maze.yml` | YAML passed to `map_publisher.py`. |
| `spawn_xy` | `2.0,2.0` | Robot spawn pose in `maze.world`  |
| `rviz` | `true` | Launch RViz |
