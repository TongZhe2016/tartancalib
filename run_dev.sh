#!/bin/bash
# Development runner for tartancalib (ROS Noetic / Ubuntu 20.04)
# Mounts source code instead of baking it into the image.
#
# First-time setup:
#   1. Build the dev image:    bash run_dev.sh --build
#   2. Start the container:    bash run_dev.sh
#   3. Inside container:       catkin build -j$(nproc) && source devel/setup.bash
#
# After modifying source, just re-run inside the container:
#   catkin build -j$(nproc)

set -e

IMAGE_NAME="tartancalib-dev"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$1" = "--build" ]; then
    echo "[run_dev.sh] Building image: $IMAGE_NAME"
    docker build -t "$IMAGE_NAME" -f "$SCRIPT_DIR/Dockerfile_ros1_20_04_dev" "$SCRIPT_DIR"
    echo "[run_dev.sh] Build done."
    exit 0
fi

# Allow X11 forwarding for GUI tools
xhost +local:root > /dev/null 2>&1 || true

DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/data"

echo "[run_dev.sh] Source: $SCRIPT_DIR -> /catkin_ws/src/kalibr"
echo "[run_dev.sh] Data:   $DATA_DIR -> /data"
echo "[run_dev.sh] Starting container. Run 'catkin build -j\$(nproc)' if this is your first time."

docker run -it --rm \
    -e DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v "$SCRIPT_DIR:/catkin_ws/src/kalibr" \
    -v "$DATA_DIR:/data" \
    "$IMAGE_NAME"
