#!/bin/bash
# Launch wreck Gazebo + continuous freefly_cam orbit + record /camera/image_raw for 90 s.
set -o pipefail

ROOT=/home/xiaoming/GS_planning_handoff
OUT=$ROOT/data/videos/wreck_orbit.mp4
mkdir -p "$(dirname $OUT)"

source /opt/ros/humble/setup.bash
export DISPLAY=:1

# The wreck world references model:// URIs; the models live in gazebo_sortie/wreck/models.
export GAZEBO_MODEL_PATH=$ROOT/gazebo_sortie/wreck/models:${GAZEBO_MODEL_PATH:-}

echo "[wreck] launching gzserver + gzclient with wreck.world..."
# gazebo_ros wrappers don't auto-find the world; invoke gzserver directly with
# the ROS plugin libs so /set_entity_state and the camera topic come up.
gzserver "$ROOT/gazebo_sortie/wreck/wreck.world" \
    --verbose \
    -s libgazebo_ros_init.so \
    -s libgazebo_ros_factory.so \
    -s libgazebo_ros_state.so \
    > /tmp/wreck_gzserver.log 2>&1 &
GZ_PID=$!
gzclient > /tmp/wreck_gzclient.log 2>&1 &
CLIENT_PID=$!
echo "[wreck] gzserver PID=$GZ_PID, gzclient PID=$CLIENT_PID"

echo "[wreck] waiting for /camera/image_raw + /set_entity_state..."
DEADLINE=$(($(date +%s) + 60))
until ros2 topic list 2>/dev/null | grep -q "^/camera/image_raw$" \
   && ros2 service list 2>/dev/null | grep -q "^/set_entity_state$"; do
    if [ $(date +%s) -gt $DEADLINE ]; then
        echo "[wreck] FAIL: camera topic or set_entity_state never appeared"
        kill -9 $GZ_PID $CLIENT_PID 2>/dev/null
        pkill -9 -f "gzserver|gzclient" 2>/dev/null
        exit 1
    fi
    sleep 2
done
sleep 5
echo "[wreck] gazebo ready"

echo "[wreck] starting continuous orbit driver in background..."
cd "$ROOT"
python3 gazebo_sortie/wreck_orbit_continuous.py --duration 90 \
    > /tmp/wreck_orbit.log 2>&1 &
ORBIT_PID=$!
echo "[wreck] orbit PID=$ORBIT_PID"

# Let it actually start moving for a few seconds before we begin the video.
sleep 3

echo "[wreck] recording $OUT for 90 s..."
python3 "$ROOT/gazebo_sortie/video_recorder.py" --out "$OUT" --duration 90 --fps 15
RC=$?
echo "[wreck] recorder finished rc=$RC"

echo "[wreck] tearing down..."
kill -INT $ORBIT_PID 2>/dev/null; sleep 1; kill -9 $ORBIT_PID 2>/dev/null
kill -INT $GZ_PID $CLIENT_PID 2>/dev/null; sleep 2
pkill -9 -f "gzserver|gzclient|wreck_orbit_continuous" 2>/dev/null
sleep 2

ls -lh "$OUT" 2>&1
echo "[wreck] done"
exit $RC
