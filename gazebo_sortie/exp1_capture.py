#!/usr/bin/env python3
"""Exp 1 capture: 72 candidate viewpoints (planner pool) + 18 held-out test
poses, both around the Herkules wreck. Same intrinsics / lighting / camera —
only the pose sets differ. The held-out yaws are offset midway between
candidate yaws and use altitudes interleaved with the candidates, so no
held-out view duplicates a candidate.
"""
from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import rclpy
from wreck_orbit_capture import WreckCapture, euler_to_quat

CENTER       = (0.0, -0.4, -57.0)
R_CAND       = 22.0
R_HELD       = 20.0                                    # slightly inside the candidate shell
CAND_YAWS    = [i * (360.0 / 18) for i in range(18)]   # 0, 20, 40, ..., 340  (18 yaws)
CAND_ALTS    = [-3.0, 0.0, 3.0, 6.0]                   # 4 altitudes => 72 candidates
HELD_YAWS    = [10.0 + i * 40.0 for i in range(9)]     # midway-offset yaws
HELD_ALTS    = [1.5, 4.5]                              # interleaved altitudes => 18 held-out


def yaw_alt_poses(center, radius, yaws_deg, alts):
    cx, cy, cz = center
    out = []
    for h in alts:
        for yd in yaws_deg:
            ang = math.radians(yd)
            ex, ey, ez = cx + radius * math.cos(ang), cy + radius * math.sin(ang), cz + h
            dx, dy, dz = cx - ex, cy - ey, cz - ez
            yaw = math.atan2(dy, dx)
            pitch = -math.atan2(dz, math.hypot(dx, dy))
            qx, qy, qz, qw = euler_to_quat(0.0, pitch, yaw)
            out.append((ex, ey, ez, qx, qy, qz, qw))
    return out


def main():
    cand = yaw_alt_poses(CENTER, R_CAND, CAND_YAWS, CAND_ALTS)
    held = yaw_alt_poses(CENTER, R_HELD, HELD_YAWS, HELD_ALTS)
    print(f"will capture {len(cand)} candidates + {len(held)} held-out")
    rclpy.init()
    OUT = Path("/home/xiaoming/GS_planning_handoff/data/wreck_exp1")
    n1 = WreckCapture(OUT / "candidates"); n1.run(cand, settle_s=0.4); n1.destroy_node()
    n2 = WreckCapture(OUT / "heldout");    n2.run(held, settle_s=0.4); n2.destroy_node()
    rclpy.shutdown()
    print("DONE")


if __name__ == "__main__":
    main()
