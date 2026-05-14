#!/usr/bin/env bash
# One-command runner: execute dense PID gap calibration map inside a running docker container.

set -euo pipefail

CONTAINER_NAME="carla-sharc-standalone"
CONTAINER_USER="admin"
HOST_OUTPUT_DIR=""
RUN_ID=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAP_SCRIPT_HOST_PATH="${SCRIPT_DIR}/is_pac/pid_gap_map_calibration.py"
BASE_SCRIPT_HOST_PATH="${SCRIPT_DIR}/is_pac/carla_acc_warmstart.py"

print_usage() {
  cat <<'EOF'
Usage: ./run_pid_gap_map_docker.sh [wrapper-options] [python-options]

Wrapper options:
  --container NAME       Docker container name (default: carla-sharc-standalone)
  --user NAME            Container user (default: admin)
  --host-output-dir DIR  Host output directory (default: ./runs/pid_gap_maps)
  --run-id ID            Custom run identifier
  -h, --help             Show help

Python options:
  Any other option is forwarded to is_pac/pid_gap_map_calibration.py.

Example:
  ./run_pid_gap_map_docker.sh --repeats-per-point 2 --initial-gap-step 0.5 --lead-speed-step 0.25
EOF
}

if [[ -z "${HOST_OUTPUT_DIR}" ]]; then
  HOST_OUTPUT_DIR="${SCRIPT_DIR}/runs/pid_gap_maps"
fi

if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="pid_gap_map_$(date +%Y%m%d_%H%M%S)_$$"
fi

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --container)
      CONTAINER_NAME="$2"
      shift 2
      ;;
    --user)
      CONTAINER_USER="$2"
      shift 2
      ;;
    --host-output-dir)
      HOST_OUTPUT_DIR="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -f "${MAP_SCRIPT_HOST_PATH}" ]]; then
  echo "ERROR: script not found at ${MAP_SCRIPT_HOST_PATH}" >&2
  exit 1
fi

if [[ ! -f "${BASE_SCRIPT_HOST_PATH}" ]]; then
  echo "ERROR: dependency script not found at ${BASE_SCRIPT_HOST_PATH}" >&2
  exit 1
fi

if ! docker inspect "${CONTAINER_NAME}" --format '{{.State.Running}}' 2>/dev/null | grep -q true; then
  echo "ERROR: container '${CONTAINER_NAME}' is not running." >&2
  echo "Start it first: ./run_container.sh --container ${CONTAINER_NAME}" >&2
  exit 1
fi

mkdir -p "${HOST_OUTPUT_DIR}"

CONTAINER_WORK_DIR="/tmp/${RUN_ID}"
CONTAINER_MAP_SCRIPT_PATH="${CONTAINER_WORK_DIR}/pid_gap_map_calibration.py"
CONTAINER_BASE_SCRIPT_PATH="${CONTAINER_WORK_DIR}/carla_acc_warmstart.py"
CONTAINER_OUTPUT_DIR="${CONTAINER_WORK_DIR}/outputs"

DEFAULT_ARGS=(
  --start-carla-server
  --start-with-xvfb
  --video
  --town Town10HD_Opt
  --no-rendering
  --weather-profile custom
  --gpu-warmup
  --ego-target-speed 5.0
  --lead-speed-min 2.0
  --lead-speed-max 10.0
  --lead-speed-step 0.5
  --initial-gap-min 8.0
  --initial-gap-max 32.0
  --initial-gap-step 1.0
  --target-gap-min 8.0
  --target-gap-max 32.0
  --target-gap-step 1.0
  --speed-settle-tol 0.2
  --speed-settle-hysteresis 0.08
  --settle-ticks 8
  --max-settle-time 8.0
  --repeats-per-point 1
  --spawn-z-offset 0.02
  --straight-lookahead-m 90
  --straight-min-length-m 90
  --straight-scan-max-m 220
  --straight-max-yaw-delta-deg 6
  --straight-require-no-junction
  --straight-sort-by-length
  --output-dir "${CONTAINER_OUTPUT_DIR}"
)

echo "==> Container: ${CONTAINER_NAME} (user: ${CONTAINER_USER})"
echo "==> Run ID:    ${RUN_ID}"
echo "==> Script:    ${MAP_SCRIPT_HOST_PATH}"
echo "==> Outputs:   ${HOST_OUTPUT_DIR}/${RUN_ID}"

docker exec -u "${CONTAINER_USER}" "${CONTAINER_NAME}" bash -lc "mkdir -p '${CONTAINER_WORK_DIR}'"
docker cp "${MAP_SCRIPT_HOST_PATH}" "${CONTAINER_NAME}:${CONTAINER_MAP_SCRIPT_PATH}"
docker cp "${BASE_SCRIPT_HOST_PATH}" "${CONTAINER_NAME}:${CONTAINER_BASE_SCRIPT_PATH}"

PY_CMD=(python3 "${CONTAINER_MAP_SCRIPT_PATH}" "${DEFAULT_ARGS[@]}" "${EXTRA_ARGS[@]}")
printf -v PY_CMD_STR '%q ' "${PY_CMD[@]}"

set +e
docker exec -u "${CONTAINER_USER}" "${CONTAINER_NAME}" bash -lc "
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate carla 2>/dev/null || true
export PYTHONDONTWRITEBYTECODE=1
${PY_CMD_STR}
"
RUN_STATUS=$?
set -e

HOST_RUN_DIR="${HOST_OUTPUT_DIR}/${RUN_ID}"
mkdir -p "${HOST_RUN_DIR}"

if docker exec -u "${CONTAINER_USER}" "${CONTAINER_NAME}" bash -lc "[ -d '${CONTAINER_OUTPUT_DIR}' ]"; then
  docker cp "${CONTAINER_NAME}:${CONTAINER_OUTPUT_DIR}/." "${HOST_RUN_DIR}"
  echo "==> Copied outputs to: ${HOST_RUN_DIR}"
else
  echo "WARNING: container output directory not found: ${CONTAINER_OUTPUT_DIR}" >&2
fi

docker exec -u "${CONTAINER_USER}" "${CONTAINER_NAME}" bash -lc "rm -rf '${CONTAINER_WORK_DIR}'" >/dev/null 2>&1 || true

if [[ ${RUN_STATUS} -ne 0 ]]; then
  echo "ERROR: run failed with status ${RUN_STATUS}" >&2
  exit ${RUN_STATUS}
fi

echo "==> Run completed successfully."
