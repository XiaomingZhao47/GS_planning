# PA1: Shape-Following Robot

TurtleBot3 shape follower for ROS 2 Humble with open-loop and closed-loop control

## Shapes

1. **Trapezoid** isosceles trapezoid parameterized by radius `r`
2. **D-Shape** straight stroke `2r` + semicircle of radius `r`
3. **General Polygon** arbitrary `(x,y)` vertices in the `odom` frame

## Usage

```bash
# Terminal 1: Launch Gazebo
export TURTLEBOT3_MODEL=waffle_pi
ros2 launch turtlebot3_gazebo empty_world.launch.py

# Terminal 2: Run open-loop
python3 src/shape_follower.py

# closed-loop
python3 src/shape_follower_closed_loop.py

# Terminal 3 RViz 
rviz2
# /actual_path, /ideal_path, /actual_path_cl, /ideal_path_cl
# Fixed Frame to "odom"
```

## Reset between runs

```bash
ros2 service call /reset_simulation std_srvs/srv/Empty
```

## Results

| Shape | Open-Loop | Closed-Loop 
|-------|-----------|-------------
| Trapezoid (r=5) | 0.181 m | 0.026 m 
| D-Shape (r=5) | 0.039 m | 0.001 m 
| Polygon (diamond ±5) | 0.203 m | 0.028 m |

## Logs

CSV trajectory and waypoint logs are saved to `~/ros2_ws/logs/` after each run
