# SGE: Sortie-based Gaussian Exploration

A sortie-based explore--build--deliberate--execute framework that plans next-best-view (NBV) sorties over a 3D Gaussian Splatting (3DGS) world model plus a depth-fused occupancy grid.

**$Q_3$-refined**: a GS-aware NBV gain function that scores candidate poses by `unknown + 0.5 × under-reconstructed` cells along forward rays, where "under-reconstructed" is restricted to a *near-surface* mask, and selection uses NMS with a per-altitude cap. 

On a 5-seed pre-registered wreck sweep, $Q_3$-refined wins volumetric IG on every metric (PSNR/SSIM/LPIPS) while training to fewer Gaussians than every IG baseline.

---

## Demo videos

The framework is evaluated across two robot cases. Each video is the live `/camera/image_raw` stream captured during a Gazebo simulation of the corresponding case.

### 2D — TurtleBot3 ground-robot wall-follower

[videos/2D_tb3_house.mp4](videos/2D_tb3_house.mp4) — 1 min 47 s

<video src="videos/2D_tb3_house.mp4" controls width="640"></video>

A waffle-pi TB3 with an RGB-D camera explores the `turtlebot3_house` world under a PID controller.

### 3D — Underwater free-flying UUV camera around the Herkules wreck

[videos/3D_wreck_orbit.mp4](videos/3D_wreck_orbit.mp4) — 1 min 37 s

<video src="videos/3D_wreck_orbit.mp4" controls width="640"></video>

A 6-DoF flying camera orbits the submerged wreck through two full revolutions. 

---


| Folder | Contents |
|---|---|
| [`gazebo_sortie/`](gazebo_sortie) | Gazebo-side code: TB3 wall-follower, UUV orbit driver, the planner-ablation driver (`exp1_run.py`), the seed wrapper, the multi-seed aggregator, and the video recorder |
| [`src/`](src) | Habitat-Sim runner, 3DGS reconstruction wrapper, 3-class log-odds occupancy grid, the five planners benchmarked on `apartment_1` (frontier, volumetric IG, reactive-GS, deliberative, SGE), and evaluation |
| [`videos/`](videos) | The two demo videos |
| [`previous_work/`](previous programming assignment) | Includes previous assignment that used as implementation components |
---

## Notes

- External components are used as libraries: the Inria 3D Gaussian Splatting optimiser is invoked as a subprocess; Habitat-Sim and Gazebo are the simulators; OctoMap was re-implemented in NumPy.

