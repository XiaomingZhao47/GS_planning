# PA2: Wall-Following Robot

Right-hand wall follower using PID control and a 5-state FSM in ROS 2 

## Run

```bash
# Terminal 1: launch Stage
ros2 launch stage_ros2 stage.launch.py \
    world:=$HOME/ros2_ws/src/pa2/2017-02-11-00-31-57 \
    enforce_prefixes:=false one_tf_tree:=true

# Terminal 2: run wall follower
python3 wall_follower.py
```

## Design

- **PID** (`Kp=1.5, Ki=0.05, Kd=0.6`): controls ω from error `d_target − d_right`
- **Front avoidance bias**: steers left as front wall approaches, handles corners without FSM transitions
- **Adaptive velocity**: quadratic slowdown from 0.5 m/s (straight) to 0.05 m/s (corners)
- **FSM**: `FIND_WALL → ALIGN_LEFT → FOLLOW_WALL ↔ INTERNAL_CORNER / SEARCH_WALL`
- **Recovery**: timeouts (5–10s) and stuck detection on all states; backup when wedged

## Tuning

All parameters adjustable at runtime

```bash
ros2 param set /wall_follower kp 2.0
ros2 param set /wall_follower d_target 0.4
```
