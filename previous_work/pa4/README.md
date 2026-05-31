# PA4

Occupancy-grid mapping in ROS 2 with log odds Bayesian update + Bresenham raycasting

## Files

- **`occupancy_grid_mapper.py`** — the mapper node. Subscribes `/base_scan`, publishes `/map`.
- **`bringup.launch.py`** — bringup contains wall_follwer, map, and rviz

## Run

```bash
source install/setup.bash
ros2 launch /ros2_ws/src/pa4/pa4_bringup/bringup.launch.py
```

RViz: fixed frame `odom`, add Map on `/map` with Durability = Transient Local


