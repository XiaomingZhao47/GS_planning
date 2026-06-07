#!/bin/bash
# Launch TB3 house Gazebo + wall_follower + record /camera/image_raw for 90 s.
set -o pipefail

ROOT=/home/xiaoming/GS_planning_handoff
OUT=$ROOT/data/videos/tb3_house.mp4
mkdir -p "$(dirname $OUT)"

source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL=waffle_pi
export DISPLAY=:1

echo "[tb3] launching gazebo + tb3 house world..."
ros2 launch turtlebot3_gazebo turtlebot3_house.launch.py \
    > /tmp/tb3_gazebo.log 2>&1 &
GAZEBO_PID=$!
echo "[tb3] gazebo PID=$GAZEBO_PID"

# Wait for the camera topic to be live. Some Gazebo runs need ~25 s with the house world.
echo "[tb3] waiting for /camera/image_raw..."
DEADLINE=$(($(date +%s) + 60))
until ros2 topic list 2>/dev/null | grep -q "^/camera/image_raw$"; do
    if [ $(date +%s) -gt $DEADLINE ]; then
        echo "[tb3] FAIL: /camera/image_raw never appeared"
        kill -9 $GAZEBO_PID 2>/dev/null
        pkill -9 -f "gzserver|gzclient" 2>/dev/null
        exit 1
    fi
    sleep 2
done
# Give it a couple more seconds so the publisher is actually pushing frames.
sleep 5
echo "[tb3] camera topic live"

echo "[tb3] starting wall_follower in background..."
cd "$ROOT"
python3 gazebo_sortie/wall_follower_tb3.py > /tmp/tb3_wallfollower.log 2>&1 &
WF_PID=$!
echo "[tb3] wall_follower PID=$WF_PID"

# Let it start moving so the recording captures actual motion, not the initial idle frame.
sleep 3

echo "[tb3] recording $OUT for 90 s..."
python3 "$ROOT/gazebo_sortie/video_recorder.py" --out "$OUT" --duration 90 --fps 15
RC=$?
echo "[tb3] recorder finished rc=$RC"

echo "[tb3] tearing down..."
kill -INT $WF_PID 2>/dev/null; sleep 1; kill -9 $WF_PID 2>/dev/null
kill -INT $GAZEBO_PID 2>/dev/null; sleep 2
pkill -9 -f "gzserver|gzclient|spawn_entity" 2>/dev/null
pkill -9 -f "ros2 launch turtlebot3_gazebo" 2>/dev/null
sleep 2

ls -lh "$OUT" 2>&1
echo "[tb3] done"
exit $RC
