# Standalone CARLA ACC PID-Map Package

This folder is a minimal standalone package for the CARLA ACC workflows we built:

- Dense PID map generation
- Inverse-map error evaluation
- Single desired initial-condition run with video

It is self-contained and does not import code from the parent SHARC repository.

## What is included

- `is_pac/carla_acc_warmstart.py`
- `is_pac/pid_gap_map_calibration.py`
- `is_pac/pid_gap_map_error_eval.py`
- `run_container.sh`
- `run_pid_gap_map_docker.sh`
- `run_pid_gap_map_eval_docker.sh`
- `run_single_ic_video_docker.sh`

## Container image

Published image used for this standalone workflow:

```bash
docker pull ausar/carla-sharc:latest
```

`run_container.sh` will pull this image automatically.

## Prerequisites

- Linux with Docker installed
- NVIDIA GPU + NVIDIA Container Toolkit for normal CARLA speed
- User permission to run docker (`docker ps` works without sudo)

## 1) Start the experiment container

From this folder:

```bash
chmod +x run_container.sh run_pid_gap_map_docker.sh run_pid_gap_map_eval_docker.sh run_single_ic_video_docker.sh
./run_container.sh
```

Default container name is `carla-sharc-standalone`.

## 2) Quick verification (fast smoke run)

These commands are lightweight checks that the package works end-to-end.

If you see `Port <N> is already in use`, switch to a different free `--port` value.
You can also clear stale CARLA server processes inside the container with:

```bash
docker exec carla-sharc-standalone bash -lc "pkill -f CarlaUE4-Linux-Shipping || true"
```

### 2.1 Smoke map generation

```bash
./run_pid_gap_map_docker.sh \
  --run-id verify_map_smoke \
  --port 2231 \
  --lead-speed-min 4 --lead-speed-max 6 --lead-speed-step 1 \
  --initial-gap-min 14 --initial-gap-max 18 --initial-gap-step 2 \
  --target-gap-min 14 --target-gap-max 18 --target-gap-step 2 \
  --repeats-per-point 1 --max-settle-time 6
```

Expected key artifact:

```bash
ls runs/pid_gap_maps/verify_map_smoke/inverse_map.csv
```

### 2.2 Smoke inverse-map evaluation

```bash
./run_pid_gap_map_eval_docker.sh \
  --run-id verify_eval_smoke \
  --port 2242 \
  --map-csv ./runs/pid_gap_maps/verify_map_smoke/inverse_map.csv \
  --eval-cases 20 \
  --seed 123
```

Expected key artifact:

```bash
ls runs/pid_gap_map_eval/verify_eval_smoke/eval_summary.json
```

### 2.3 Single initial condition with video

```bash
./run_single_ic_video_docker.sh \
  --run-id verify_single_ic \
  --port 2301 \
  --h0 20 --v-lead0 8.4 --v-ego0 7.0 \
  --t-max 6
```

Expected key artifacts:

```bash
ls runs/single_ic/verify_single_ic/episodes_summary.csv
find runs/single_ic/verify_single_ic -name '*.mp4' -print
```

## 3) Full run: extra-dense map + rigorous eval

The commands below are the full run configuration used for the larger experiment.

### 3.1 Extra-dense map generation

```bash
./run_pid_gap_map_docker.sh \
  --run-id pid_map_extra_dense_full_20260513 \
  --port 2401 \
  --lead-speed-min 2 --lead-speed-max 10 --lead-speed-step 0.25 \
  --initial-gap-min 8 --initial-gap-max 32 --initial-gap-step 0.5 \
  --target-gap-min 8 --target-gap-max 32 --target-gap-step 0.5 \
  --repeats-per-point 1 \
  --seed 456
```

Key output:

```bash
ls runs/pid_gap_maps/pid_map_extra_dense_full_20260513/inverse_map.csv
```

### 3.2 Rigorous eval (500 cases x 3 seeds)

```bash
./run_pid_gap_map_eval_docker.sh \
  --run-id pid_eval_rigorous_500_seed123_20260513 \
  --port 31123 \
  --map-csv ./runs/pid_gap_maps/pid_map_extra_dense_full_20260513/inverse_map.csv \
  --eval-cases 500 \
  --seed 123

./run_pid_gap_map_eval_docker.sh \
  --run-id pid_eval_rigorous_500_seed456_20260513 \
  --port 31156 \
  --map-csv ./runs/pid_gap_maps/pid_map_extra_dense_full_20260513/inverse_map.csv \
  --eval-cases 500 \
  --seed 456

./run_pid_gap_map_eval_docker.sh \
  --run-id pid_eval_rigorous_500_seed789_20260513 \
  --port 2404 \
  --map-csv ./runs/pid_gap_maps/pid_map_extra_dense_full_20260513/inverse_map.csv \
  --eval-cases 500 \
  --seed 789
```

