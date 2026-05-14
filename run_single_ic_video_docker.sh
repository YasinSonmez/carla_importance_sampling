#!/usr/bin/env bash
# One-command runner: run one fixed initial condition and produce trajectory + summary + video.

set -euo pipefail

CONTAINER_NAME="carla-sharc-standalone"
CONTAINER_USER="admin"
HOST_OUTPUT_DIR=""
RUN_ID=""
H0="20.0"
V_LEAD0="8.4"
V_EGO0="7.0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_HOST_PATH="${SCRIPT_DIR}/is_pac/carla_acc_warmstart.py"

print_usage() {
  cat <<'EOF'
Usage: ./run_single_ic_video_docker.sh [wrapper-options] [python-options]

Wrapper options:
  --container NAME       Docker container name (default: carla-sharc-standalone)
  --user NAME            Container user (default: admin)
  --host-output-dir DIR  Host output directory (default: ./runs/single_ic)
  --run-id ID            Custom run identifier
  --h0 VALUE             Initial gap in meters (default: 20.0)
  --v-lead0 VALUE        Initial lead speed m/s (default: 8.4)
  --v-ego0 VALUE         Initial ego speed m/s (default: 7.0)
  -h, --help             Show help

Python options:
  Any additional args are forwarded to is_pac/carla_acc_warmstart.py.

Example:
  ./run_single_ic_video_docker.sh --run-id demo_ic --port 2230 --h0 20 --v-lead0 8.4 --v-ego0 7.0 --t-max 8
EOF
}

if [[ -z "${HOST_OUTPUT_DIR}" ]]; then
  HOST_OUTPUT_DIR="${SCRIPT_DIR}/runs/single_ic"
fi

if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="single_ic_$(date +%Y%m%d_%H%M%S)_$$"
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
    --h0)
      H0="$2"
      shift 2
      ;;
    --v-lead0)
      V_LEAD0="$2"
      shift 2
      ;;
    --v-ego0)
      V_EGO0="$2"
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

if [[ ! -f "${SCRIPT_HOST_PATH}" ]]; then
  echo "ERROR: script not found at ${SCRIPT_HOST_PATH}" >&2
  exit 1
fi

if ! docker inspect "${CONTAINER_NAME}" --format '{{.State.Running}}' 2>/dev/null | grep -q true; then
  echo "ERROR: container '${CONTAINER_NAME}' is not running." >&2
  echo "Start it first: ./run_container.sh --container ${CONTAINER_NAME}" >&2
  exit 1
fi

mkdir -p "${HOST_OUTPUT_DIR}"

CONTAINER_WORK_DIR="/tmp/${RUN_ID}"
CONTAINER_SCRIPT_PATH="${CONTAINER_WORK_DIR}/carla_acc_warmstart.py"
CONTAINER_OUTPUT_DIR="${CONTAINER_WORK_DIR}/outputs"

DEFAULT_ARGS=(
  --start-carla-server
  --start-with-xvfb
  --town Town10HD_Opt
  --weather-profile custom
  --scenario-profile toy_drag_brake
  --sample-mode fixed
  --num-samples 1
  --h0 "${H0}"
  --v-lead0 "${V_LEAD0}"
  --v-ego0 "${V_EGO0}"
  --warmstart-mode strict_only
  --spawn-z-offset 0.02
  --strict-ground-z-offset 0.02
  --max-warmup-ticks 0
  --settle-ticks 1
  --gap-tolerance 0.05
  --speed-tolerance 0.40
  --strict-snap-settle-ticks 3
  --strict-max-iterations 40
  --strict-retry-attempts 3
  --fail-on-init-mismatch
  --straight-lookahead-m 90
  --straight-min-length-m 90
  --straight-scan-max-m 220
  --straight-max-yaw-delta-deg 6
  --straight-require-no-junction
  --straight-sort-by-length
  --video
  --video-hud
  --exclude-warmstart-in-video
  --video-width 1920
  --video-height 1080
  --video-crf 18
  --video-preset slow
  --accel 0.25
  --drag-b 0.01
  --disturbance-uniform 0.03
  --speed-noise-uniform 0.03
  --gpu-warmup
  --native-recorder
  --output-dir "${CONTAINER_OUTPUT_DIR}"
)

echo "==> Container: ${CONTAINER_NAME} (user: ${CONTAINER_USER})"
echo "==> Run ID:    ${RUN_ID}"
echo "==> Request:   h0=${H0}, v_lead0=${V_LEAD0}, v_ego0=${V_EGO0}"
echo "==> Outputs:   ${HOST_OUTPUT_DIR}/${RUN_ID}"

docker exec -u "${CONTAINER_USER}" "${CONTAINER_NAME}" bash -lc "mkdir -p '${CONTAINER_WORK_DIR}'"
docker cp "${SCRIPT_HOST_PATH}" "${CONTAINER_NAME}:${CONTAINER_SCRIPT_PATH}"

PY_CMD=(python3 "${CONTAINER_SCRIPT_PATH}" "${DEFAULT_ARGS[@]}" "${EXTRA_ARGS[@]}")
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
