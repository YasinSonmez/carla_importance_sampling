#!/usr/bin/env bash
# Pull and run the published CARLA+SHARC image for standalone experiments.

set -euo pipefail

IMAGE="ausar/carla-sharc:latest"
CONTAINER_NAME="carla-sharc-standalone"
USE_GPU=1
RECREATE=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOUNT_DST="/home/workspace/my_files/standalone_carla_acc"

print_usage() {
  cat <<'EOF'
Usage: ./run_container.sh [options]

Options:
  --image NAME       Docker image (default: ausar/carla-sharc:latest)
  --container NAME   Container name (default: carla-sharc-standalone)
  --no-gpu           Do not request --gpus all
  --recreate         Remove existing container before starting
  -h, --help         Show help

Environment overrides:
  IMAGE, CONTAINER_NAME
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE="$2"
      shift 2
      ;;
    --container)
      CONTAINER_NAME="$2"
      shift 2
      ;;
    --no-gpu)
      USE_GPU=0
      shift
      ;;
    --recreate)
      RECREATE=1
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      print_usage
      exit 1
      ;;
  esac
done

echo "==> Pulling image: ${IMAGE}"
docker pull "${IMAGE}"

if docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  if [[ ${RECREATE} -eq 1 ]]; then
    echo "==> Recreating existing container: ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}" >/dev/null
  else
    if docker inspect "${CONTAINER_NAME}" --format '{{.State.Running}}' | grep -q true; then
      echo "==> Container already running: ${CONTAINER_NAME}"
      echo "==> Mounted project path in container: ${MOUNT_DST}"
      exit 0
    fi
    echo "==> Starting existing container: ${CONTAINER_NAME}"
    docker start "${CONTAINER_NAME}" >/dev/null
    echo "==> Mounted project path in container: ${MOUNT_DST}"
    exit 0
  fi
fi

COMMON_ARGS=(
  --detach
  --name "${CONTAINER_NAME}"
  --privileged
  --network host
  --shm-size=8gb
  --ulimit memlock=-1
  --ulimit stack=67108864
  --volume "${SCRIPT_DIR}:${MOUNT_DST}"
  --workdir "${MOUNT_DST}"
)

START_CMD=(bash -lc "tail -f /dev/null")
X11_ARGS=()

if [[ -n "${DISPLAY:-}" && -d /tmp/.X11-unix ]]; then
  xhost +local:docker >/dev/null 2>&1 || true
  X11_ARGS+=(
    -e "DISPLAY=${DISPLAY}"
    -e "QT_X11_NO_MITSHM=1"
    -e "SDL_VIDEODRIVER=x11"
    -v "/tmp/.X11-unix:/tmp/.X11-unix"
  )
fi

if [[ ${USE_GPU} -eq 1 ]]; then
  echo "==> Starting container with GPU support"
  if ! docker run "${COMMON_ARGS[@]}" "${X11_ARGS[@]}" --gpus all --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e __NV_PRIME_RENDER_OFFLOAD=1 -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
    "${IMAGE}" "${START_CMD[@]}"; then
    echo "WARNING: Failed with --runtime=nvidia. Retrying with --gpus all only." >&2
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    if ! docker run "${COMMON_ARGS[@]}" "${X11_ARGS[@]}" --gpus all \
      -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
      "${IMAGE}" "${START_CMD[@]}"; then
      echo "WARNING: GPU launch failed. Retrying without GPU flag." >&2
      docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
      docker run "${COMMON_ARGS[@]}" "${X11_ARGS[@]}" "${IMAGE}" "${START_CMD[@]}"
    fi
  fi
else
  echo "==> Starting container without GPU flag"
  docker run "${COMMON_ARGS[@]}" "${X11_ARGS[@]}" "${IMAGE}" "${START_CMD[@]}"
fi

echo "==> Container is ready: ${CONTAINER_NAME}"
echo "==> Mounted project path in container: ${MOUNT_DST}"
echo "==> Next: run one of the experiment scripts from this folder."
