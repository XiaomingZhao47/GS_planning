#!/bin/bash
# Concatenate the TB3 house clip and the wreck orbit clip into a single
# gazebo_exploration.mp4 with brief title overlays on each segment.
set -euo pipefail

ROOT=/home/xiaoming/GS_planning_handoff
VID=$ROOT/data/videos
TB3=$VID/tb3_house.mp4
UUV=$VID/wreck_orbit.mp4
OUT=$VID/gazebo_exploration.mp4

if [ ! -f "$TB3" ]; then echo "missing $TB3"; exit 1; fi
if [ ! -f "$UUV" ]; then echo "missing $UUV"; exit 1; fi

TB3_DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$TB3")
UUV_DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$UUV")
echo "tb3:   $TB3_DUR s"
echo "wreck: $UUV_DUR s"

WORK=$(mktemp -d)
trap "rm -rf $WORK" EXIT

# Re-encode each clip with a sticky title overlay in the top-left for the first 5 s.
ffmpeg -y -loglevel error -i "$TB3" \
    -vf "drawtext=text='TB3 house (2D ground robot, wall-follower)':fontcolor=white:fontsize=22:box=1:boxcolor=black@0.55:boxborderw=8:x=20:y=20:enable='lt(t,5)'" \
    -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -an \
    "$WORK/tb3_titled.mp4"

ffmpeg -y -loglevel error -i "$UUV" \
    -vf "drawtext=text='Herkules wreck (6-DoF free-flying UUV camera)':fontcolor=white:fontsize=22:box=1:boxcolor=black@0.55:boxborderw=8:x=20:y=20:enable='lt(t,5)'" \
    -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -an \
    "$WORK/uuv_titled.mp4"

# Concat (segments are now identically encoded, so the demuxer concat is safe)
cat > "$WORK/list.txt" <<EOF
file '$WORK/tb3_titled.mp4'
file '$WORK/uuv_titled.mp4'
EOF
ffmpeg -y -loglevel error -f concat -safe 0 -i "$WORK/list.txt" -c copy "$OUT"

echo ""
echo "wrote $OUT"
ffprobe -v error -show_entries format=duration,size:stream=width,height,r_frame_rate -of compact=p=0 "$OUT"
