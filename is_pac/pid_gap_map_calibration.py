#!/usr/bin/env python3
"""
Dense PID-based gap calibration map for CARLA car-following.

Purpose:
- Fix ego target speed (default 5 m/s).
- Sweep initial gap d_init and lead target speed v_lead_target.
- Run both vehicles under PID speed control only (no gap teleport/snap).
- Measure settled gap d_final when both vehicles reach target speeds.
- Build forward map: (d_init, v_lead_target) -> (d_final, settle_time).
- Build inverse map: (d_target, v_lead_target) -> required d_init.

Outputs:
- raw_trials.csv
- forward_map.csv
- inverse_map.csv
- monotonicity_report.csv
- forward_gap_heatmap.png
- delta_gap_heatmap.png
- settle_time_heatmap.png
- settle_success_rate_heatmap.png
- inverse_required_dinit_heatmap.png
- monotonic_slices.png
- run_manifest.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from carla_acc_warmstart import (  # pylint: disable=wrong-import-position
    CollisionMonitor,
    LongitudinalPID,
    bumper_gap_m,
    build_weather,
    clear_stale_actors,
    connect_world,
    destroy_actor_safe,
    find_straight_spawn_indices,
    forward_speed_mps,
    is_rpc_open,
    lane_steer,
    load_world_with_town_fallback,
    make_vehicle_pair,
    run_gpu_warmup,
    set_async_mode,
    set_sync_mode,
    start_carla_server,
    stop_carla_server,
    wait_for_rpc,
)

try:
    import carla  # type: ignore
except Exception as exc:
    raise SystemExit(
        "Could not import carla Python API. Activate CARLA environment first."
    ) from exc


@dataclass
class TrialResult:
    repeat_idx: int
    case_idx: int
    initial_gap_m: float
    lead_target_speed_mps: float
    ego_target_speed_mps: float
    final_gap_m: float
    delta_gap_m: float
    final_lead_speed_mps: float
    final_ego_speed_mps: float
    settled: bool
    collision: bool
    settle_time_s: float
    ticks_executed: int
    spawn_ok: bool
    error: str


@dataclass
class AggregateResult:
    initial_gap_m: float
    lead_target_speed_mps: float
    ego_target_speed_mps: float
    repeats: int
    success_rate: float
    settled_count: int
    final_gap_mean_m: float
    final_gap_std_m: float
    delta_gap_mean_m: float
    delta_gap_std_m: float
    settle_time_mean_s: float
    settle_time_std_s: float
    final_lead_speed_mean_mps: float
    final_ego_speed_mean_mps: float


def grid_values(v_min: float, v_max: float, step: float) -> np.ndarray:
    if step <= 0.0:
        raise ValueError("step must be > 0")
    if v_max < v_min:
        raise ValueError("max must be >= min")
    n = int(math.floor((v_max - v_min) / step + 1e-9)) + 1
    vals = v_min + np.arange(n, dtype=float) * step
    if vals[-1] < v_max - 1e-9:
        vals = np.append(vals, v_max)
    vals = np.round(vals, 6)
    return vals


def safe_nanmean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def safe_nanstd(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanstd(arr))


def run_single_trial(
    world: "carla.World",
    carla_map: "carla.Map",
    traffic_manager: "carla.TrafficManager",
    candidate_spawn_indices: List[int],
    spawn_index: int,
    spawn_z_offset: float,
    initial_gap_m: float,
    lead_target_speed_mps: float,
    ego_target_speed_mps: float,
    dt: float,
    speed_settle_tol_mps: float,
    speed_settle_hysteresis_mps: float,
    settle_ticks_required: int,
    max_settle_time_s: float,
    ego_model: str,
    lead_model: str,
    repeat_idx: int,
    case_idx: int,
) -> TrialResult:
    ego: Optional["carla.Vehicle"] = None
    lead: Optional["carla.Vehicle"] = None
    collision_sensor: Optional["carla.Sensor"] = None

    monitor = CollisionMonitor()

    final_gap_m = float("nan")
    final_lead_speed_mps = float("nan")
    final_ego_speed_mps = float("nan")
    settle_time_s = float("nan")
    ticks_executed = 0
    settled = False
    spawn_ok = False
    error = ""

    try:
        ego, lead, _, _ = make_vehicle_pair(
            world=world,
            carla_map=carla_map,
            spawn_index=spawn_index,
            desired_gap_m=initial_gap_m,
            ego_model=ego_model,
            lead_model=lead_model,
            spawn_z_offset=spawn_z_offset,
            candidate_spawn_indices=candidate_spawn_indices,
        )
        spawn_ok = True

        tm_port = traffic_manager.get_port()
        ego.set_autopilot(False, tm_port)
        lead.set_autopilot(False, tm_port)

        collision_bp = world.get_blueprint_library().find("sensor.other.collision")
        collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=ego)
        collision_sensor.listen(monitor.callback)

        lead_pid = LongitudinalPID()
        ego_pid = LongitudinalPID()

        max_ticks = max(1, int(math.ceil(max_settle_time_s / dt)))
        speed_ok_count = 0

        world.tick()

        for k in range(1, max_ticks + 1):
            ticks_executed = k

            final_gap_m = bumper_gap_m(lead, ego)
            final_lead_speed_mps = max(0.0, forward_speed_mps(lead))
            final_ego_speed_mps = max(0.0, forward_speed_mps(ego))

            lead_throttle, lead_brake = lead_pid.step(lead_target_speed_mps, final_lead_speed_mps, dt)
            ego_throttle, ego_brake = ego_pid.step(ego_target_speed_mps, final_ego_speed_mps, dt)

            lead.apply_control(
                carla.VehicleControl(
                    throttle=lead_throttle,
                    brake=lead_brake,
                    steer=lane_steer(lead, carla_map),
                )
            )
            ego.apply_control(
                carla.VehicleControl(
                    throttle=ego_throttle,
                    brake=ego_brake,
                    steer=lane_steer(ego, carla_map),
                )
            )

            world.tick()

            final_gap_m = bumper_gap_m(lead, ego)
            final_lead_speed_mps = max(0.0, forward_speed_mps(lead))
            final_ego_speed_mps = max(0.0, forward_speed_mps(ego))

            if monitor.collided:
                break

            lead_err = abs(final_lead_speed_mps - lead_target_speed_mps)
            ego_err = abs(final_ego_speed_mps - ego_target_speed_mps)

            # Hysteresis avoids one-tick tolerance boundary chatter that can
            # prevent convergence declaration in otherwise settled trajectories.
            if speed_ok_count <= 0:
                lead_ok = lead_err <= speed_settle_tol_mps
                ego_ok = ego_err <= speed_settle_tol_mps
            else:
                lead_ok = lead_err <= (speed_settle_tol_mps + speed_settle_hysteresis_mps)
                ego_ok = ego_err <= (speed_settle_tol_mps + speed_settle_hysteresis_mps)

            if lead_ok and ego_ok:
                speed_ok_count += 1
                if speed_ok_count >= settle_ticks_required:
                    settled = True
                    settle_time_s = k * dt
                    break
            else:
                speed_ok_count = 0

    except Exception as exc:  # pragma: no cover
        error = str(exc)

    finally:
        destroy_actor_safe(collision_sensor)
        destroy_actor_safe(lead)
        destroy_actor_safe(ego)
        try:
            world.tick()
        except Exception:
            pass

    if not spawn_ok and not error:
        error = "spawn_failed"

    delta_gap_m = float("nan")
    if not math.isnan(final_gap_m):
        delta_gap_m = final_gap_m - initial_gap_m

    return TrialResult(
        repeat_idx=repeat_idx,
        case_idx=case_idx,
        initial_gap_m=initial_gap_m,
        lead_target_speed_mps=lead_target_speed_mps,
        ego_target_speed_mps=ego_target_speed_mps,
        final_gap_m=final_gap_m,
        delta_gap_m=delta_gap_m,
        final_lead_speed_mps=final_lead_speed_mps,
        final_ego_speed_mps=final_ego_speed_mps,
        settled=settled,
        collision=monitor.collided,
        settle_time_s=settle_time_s,
        ticks_executed=ticks_executed,
        spawn_ok=spawn_ok,
        error=error,
    )


def aggregate_trials(trials: List[TrialResult], repeats_per_point: int) -> List[AggregateResult]:
    grouped: Dict[Tuple[float, float, float], List[TrialResult]] = {}
    for tr in trials:
        key = (
            round(tr.initial_gap_m, 6),
            round(tr.lead_target_speed_mps, 6),
            round(tr.ego_target_speed_mps, 6),
        )
        grouped.setdefault(key, []).append(tr)

    out: List[AggregateResult] = []
    for (d0, v_lead, v_ego), group in sorted(grouped.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        settled_ok = [g for g in group if g.settled and (not g.collision) and g.spawn_ok and (g.error == "")]
        success_rate = len(settled_ok) / float(max(1, repeats_per_point))

        out.append(
            AggregateResult(
                initial_gap_m=d0,
                lead_target_speed_mps=v_lead,
                ego_target_speed_mps=v_ego,
                repeats=len(group),
                success_rate=success_rate,
                settled_count=len(settled_ok),
                final_gap_mean_m=safe_nanmean([g.final_gap_m for g in settled_ok]),
                final_gap_std_m=safe_nanstd([g.final_gap_m for g in settled_ok]),
                delta_gap_mean_m=safe_nanmean([g.delta_gap_m for g in settled_ok]),
                delta_gap_std_m=safe_nanstd([g.delta_gap_m for g in settled_ok]),
                settle_time_mean_s=safe_nanmean([g.settle_time_s for g in settled_ok]),
                settle_time_std_s=safe_nanstd([g.settle_time_s for g in settled_ok]),
                final_lead_speed_mean_mps=safe_nanmean([g.final_lead_speed_mps for g in settled_ok]),
                final_ego_speed_mean_mps=safe_nanmean([g.final_ego_speed_mps for g in settled_ok]),
            )
        )

    return out


def save_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_grid(
    lead_values: np.ndarray,
    init_gap_values: np.ndarray,
    aggregates: List[AggregateResult],
) -> Dict[str, np.ndarray]:
    lead_idx = {round(v, 6): i for i, v in enumerate(lead_values.tolist())}
    gap_idx = {round(v, 6): j for j, v in enumerate(init_gap_values.tolist())}

    shape = (len(lead_values), len(init_gap_values))
    final_gap = np.full(shape, np.nan, dtype=float)
    delta_gap = np.full(shape, np.nan, dtype=float)
    settle_time = np.full(shape, np.nan, dtype=float)
    success_rate = np.full(shape, np.nan, dtype=float)

    for ag in aggregates:
        i = lead_idx.get(round(ag.lead_target_speed_mps, 6))
        j = gap_idx.get(round(ag.initial_gap_m, 6))
        if i is None or j is None:
            continue
        final_gap[i, j] = ag.final_gap_mean_m
        delta_gap[i, j] = ag.delta_gap_mean_m
        settle_time[i, j] = ag.settle_time_mean_s
        success_rate[i, j] = ag.success_rate

    return {
        "final_gap": final_gap,
        "delta_gap": delta_gap,
        "settle_time": settle_time,
        "success_rate": success_rate,
    }


def build_inverse_grid(
    lead_values: np.ndarray,
    init_gap_values: np.ndarray,
    final_gap_grid: np.ndarray,
    target_gap_values: np.ndarray,
) -> np.ndarray:
    inv = np.full((len(lead_values), len(target_gap_values)), np.nan, dtype=float)

    for i in range(len(lead_values)):
        y_init = init_gap_values.copy()
        x_final = final_gap_grid[i, :].copy()

        mask = np.isfinite(x_final)
        if int(np.sum(mask)) < 2:
            continue

        x = x_final[mask]
        y = y_init[mask]

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        x_round = np.round(x, 4)
        unique_x, inv_idx = np.unique(x_round, return_inverse=True)
        unique_y = np.array([np.mean(y[inv_idx == k]) for k in range(len(unique_x))], dtype=float)

        if unique_x.size < 2:
            continue

        for j, d_target in enumerate(target_gap_values):
            if d_target < unique_x[0] or d_target > unique_x[-1]:
                continue
            inv[i, j] = float(np.interp(d_target, unique_x, unique_y))

    return inv


def monotonicity_report(
    lead_values: np.ndarray,
    init_gap_values: np.ndarray,
    final_gap_grid: np.ndarray,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, v_lead in enumerate(lead_values):
        y = final_gap_grid[i, :]
        mask = np.isfinite(y)
        x = init_gap_values[mask]
        y = y[mask]

        if y.size < 2:
            rows.append(
                {
                    "lead_target_speed_mps": float(v_lead),
                    "n_points": int(y.size),
                    "nondecreasing_ratio": float("nan"),
                    "min_local_slope": float("nan"),
                    "max_local_slope": float("nan"),
                }
            )
            continue

        dx = np.diff(x)
        dy = np.diff(y)
        slopes = dy / dx
        nondec_ratio = float(np.mean(slopes >= -1e-3))

        rows.append(
            {
                "lead_target_speed_mps": float(v_lead),
                "n_points": int(y.size),
                "nondecreasing_ratio": nondec_ratio,
                "min_local_slope": float(np.min(slopes)),
                "max_local_slope": float(np.max(slopes)),
            }
        )
    return rows


def _imshow_heatmap(
    out_path: Path,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    z_grid: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    colorbar_label: str,
    cmap_name: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    if plt is None:
        return

    z_masked = np.ma.masked_invalid(z_grid)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("lightgray")

    dx = (x_vals[1] - x_vals[0]) if len(x_vals) > 1 else 1.0
    dy = (y_vals[1] - y_vals[0]) if len(y_vals) > 1 else 1.0
    extent = [
        float(x_vals[0] - dx / 2.0),
        float(x_vals[-1] + dx / 2.0),
        float(y_vals[0] - dy / 2.0),
        float(y_vals[-1] + dy / 2.0),
    ]

    fig, ax = plt.subplots(figsize=(10.5, 6.5), dpi=160)
    im = ax.imshow(
        z_masked,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(colorbar_label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(color="white", alpha=0.08, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_monotonic_slices(
    out_path: Path,
    init_gap_values: np.ndarray,
    lead_values: np.ndarray,
    final_gap_grid: np.ndarray,
) -> None:
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(10.5, 6.5), dpi=160)

    n_lines = min(6, len(lead_values))
    selected = np.linspace(0, len(lead_values) - 1, n_lines, dtype=int)
    selected = np.unique(selected)

    for idx in selected:
        y = final_gap_grid[idx, :]
        ax.plot(
            init_gap_values,
            y,
            marker="o",
            markersize=2.5,
            linewidth=1.6,
            label=f"v_lead*={lead_values[idx]:.1f} m/s",
        )

    ax.plot(init_gap_values, init_gap_values, "k--", linewidth=1.4, label="identity: d_final=d_init")
    ax.set_xlabel("Initial gap d_init [m]")
    ax.set_ylabel("Settled gap d_final [m]")
    ax.set_title("Forward Map Slices: d_init -> d_final")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dense PID-based initial-gap calibration map in CARLA")

    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2010)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--town", default="Town10HD_Opt")

    parser.add_argument("--start-carla-server", action="store_true")
    parser.add_argument("--carla-root", default=os.environ.get("CARLA_ROOT", "/home/workspace/carla_0.9.16"))
    parser.add_argument("--carla-start-timeout", type=float, default=180.0)
    parser.add_argument("--start-with-xvfb", dest="start_with_xvfb", action="store_true")
    parser.add_argument("--no-xvfb", dest="start_with_xvfb", action="store_false")
    parser.add_argument("--xvfb-screen-depth", type=int, default=24)
    parser.add_argument("--carla-quality-level", default="Low", choices=["Low", "Epic"])
    parser.add_argument("--carla-res-x", type=int, default=1280)
    parser.add_argument("--carla-res-y", type=int, default=720)
    parser.add_argument("--carla-prefernvidia", dest="carla_prefernvidia", action="store_true")
    parser.add_argument("--no-carla-prefernvidia", dest="carla_prefernvidia", action="store_false")
    parser.add_argument("--video", dest="video", action="store_true", help="Start CARLA without -nullrhi (recommended for this container)")
    parser.add_argument("--no-video", dest="video", action="store_false", help="Allow server launcher to use -nullrhi")

    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=456)
    parser.add_argument("--no-rendering", action="store_true")
    parser.add_argument("--gpu-warmup", action="store_true")

    parser.add_argument(
        "--weather-profile",
        default="custom",
        choices=[
            "custom",
            "professional_wet_golden_hour",
            "wet_sunset",
            "wet_cloudy_sunset",
            "mid_rain_sunset",
            "clear_sunset",
            "clear_noon",
        ],
    )
    parser.add_argument("--sun-altitude-angle", type=float, default=None)
    parser.add_argument("--sun-azimuth-angle", type=float, default=None)
    parser.add_argument("--wetness", type=float, default=None)

    parser.add_argument("--ego-target-speed", type=float, default=5.0)
    parser.add_argument("--lead-speed-min", type=float, default=2.0)
    parser.add_argument("--lead-speed-max", type=float, default=10.0)
    parser.add_argument("--lead-speed-step", type=float, default=0.5)

    parser.add_argument("--initial-gap-min", type=float, default=8.0)
    parser.add_argument("--initial-gap-max", type=float, default=32.0)
    parser.add_argument("--initial-gap-step", type=float, default=1.0)

    parser.add_argument("--target-gap-min", type=float, default=8.0)
    parser.add_argument("--target-gap-max", type=float, default=32.0)
    parser.add_argument("--target-gap-step", type=float, default=1.0)

    parser.add_argument("--speed-settle-tol", type=float, default=0.2)
    parser.add_argument("--speed-settle-hysteresis", type=float, default=0.08)
    parser.add_argument("--settle-ticks", type=int, default=8)
    parser.add_argument("--max-settle-time", type=float, default=8.0)
    parser.add_argument("--repeats-per-point", type=int, default=1)

    parser.add_argument("--spawn-index", type=int, default=0)
    parser.add_argument("--spawn-z-offset", type=float, default=0.02)
    parser.add_argument("--ego-model", default="vehicle.lincoln.mkz_2020")
    parser.add_argument("--lead-model", default="vehicle.lincoln.mkz_2020")

    parser.add_argument("--straight-road-only", dest="straight_road_only", action="store_true")
    parser.add_argument("--allow-curved-spawn", dest="straight_road_only", action="store_false")
    parser.add_argument("--straight-lookahead-m", type=float, default=90.0)
    parser.add_argument("--straight-sample-step-m", type=float, default=2.0)
    parser.add_argument("--straight-max-yaw-delta-deg", type=float, default=6.0)
    parser.add_argument("--straight-min-length-m", type=float, default=90.0)
    parser.add_argument("--straight-scan-max-m", type=float, default=220.0)
    parser.add_argument("--straight-require-no-junction", dest="straight_require_no_junction", action="store_true")
    parser.add_argument("--straight-allow-junction", dest="straight_require_no_junction", action="store_false")
    parser.add_argument("--straight-sort-by-length", dest="straight_sort_by_length", action="store_true")
    parser.add_argument("--straight-keep-map-order", dest="straight_sort_by_length", action="store_false")

    parser.add_argument("--output-dir", default="")

    parser.set_defaults(
        start_with_xvfb=True,
        carla_prefernvidia=True,
        no_rendering=True,
        video=True,
        straight_road_only=True,
        straight_require_no_junction=True,
        straight_sort_by_length=True,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.output_dir:
        out_root = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = Path("runs") / f"pid_gap_map_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    lead_values = grid_values(args.lead_speed_min, args.lead_speed_max, args.lead_speed_step)
    init_gap_values = grid_values(args.initial_gap_min, args.initial_gap_max, args.initial_gap_step)
    target_gap_values = grid_values(args.target_gap_min, args.target_gap_max, args.target_gap_step)

    n_points = len(lead_values) * len(init_gap_values)
    total_trials = n_points * max(1, args.repeats_per_point)

    print("=" * 72)
    print("CARLA PID Gap Calibration Map")
    print("=" * 72)
    print(f"Output dir: {out_root.resolve()}")
    print(f"Town: {args.town}")
    print(f"Ego target speed fixed: {args.ego_target_speed:.3f} m/s")
    print(
        "Grid: "
        f"v_lead in [{args.lead_speed_min}, {args.lead_speed_max}] step {args.lead_speed_step} "
        f"({len(lead_values)} values), "
        f"d_init in [{args.initial_gap_min}, {args.initial_gap_max}] step {args.initial_gap_step} "
        f"({len(init_gap_values)} values)"
    )
    print(f"Total trials: {total_trials} (repeats={args.repeats_per_point})")

    carla_proc: Optional[Any] = None
    carla_log_path: Optional[Path] = None

    if args.start_carla_server:
        if is_rpc_open(args.host, args.port, timeout_s=1.0):
            raise SystemExit(
                f"Port {args.port} is already in use. Choose another --port or run without --start-carla-server."
            )
        print("Starting CARLA server from script...")
        carla_proc, carla_log_path = start_carla_server(args, out_root)
        if not wait_for_rpc(args.host, args.port, args.carla_start_timeout):
            stop_carla_server(carla_proc)
            log_msg = f" See log: {carla_log_path}" if carla_log_path else ""
            raise SystemExit(
                f"CARLA did not open RPC on {args.host}:{args.port} within {args.carla_start_timeout}s.{log_msg}"
            )
        print(f"CARLA ready. Log: {carla_log_path}")

    client, world = connect_world(args.host, args.port, args.timeout)

    if args.town:
        print(f"Loading town: {args.town}")
        world, loaded_town = load_world_with_town_fallback(client, args.town)
        print(f"Loaded town: {loaded_town}")

    set_sync_mode(world, args.dt, no_rendering=args.no_rendering)

    tm_port = max(8000, args.port + 6000)
    traffic_manager = client.get_trafficmanager(tm_port)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_random_device_seed(args.seed)

    weather = build_weather(args)
    world.set_weather(weather)
    for tl in world.get_actors().filter("traffic.traffic_light"):
        tl.set_state(carla.TrafficLightState.Green)
        tl.freeze(True)

    world.tick()

    stale = clear_stale_actors(client, world)
    if stale:
        print(f"Cleared stale actors: {stale}")

    if args.gpu_warmup:
        print("Running GPU warmup...")
        run_gpu_warmup(world)
        world.tick()

    carla_map = world.get_map()
    all_spawn_points = carla_map.get_spawn_points()
    candidate_spawn_indices: List[int] = list(range(len(all_spawn_points)))

    if args.straight_road_only:
        straight, scored = find_straight_spawn_indices(
            carla_map=carla_map,
            lookahead_m=args.straight_lookahead_m,
            step_m=args.straight_sample_step_m,
            max_yaw_delta_deg=args.straight_max_yaw_delta_deg,
            min_length_m=args.straight_min_length_m,
            scan_max_m=args.straight_scan_max_m,
            require_no_junction=args.straight_require_no_junction,
            sort_by_length_desc=args.straight_sort_by_length,
        )
        if straight:
            candidate_spawn_indices = straight
            top_preview = ", ".join([f"{idx}:{dist:.0f}m" for idx, dist in scored[:5]])
            print(
                f"Using straight-lane spawn subset: {len(straight)} / {len(all_spawn_points)} "
                f"(min={args.straight_min_length_m:.0f}m, no_junction={int(args.straight_require_no_junction)})"
            )
            if top_preview:
                print(f"Top straight candidates (spawn_idx:length): {top_preview}")
        else:
            print("WARNING: no straight-lane candidates found with requested thresholds; using all spawns.")

    trials: List[TrialResult] = []

    started = time.time()
    done = 0
    progress_every = max(10, total_trials // 40)

    try:
        case_counter = 0
        for repeat_idx in range(args.repeats_per_point):
            for v_lead in lead_values:
                for d_init in init_gap_values:
                    case_counter += 1
                    tr = run_single_trial(
                        world=world,
                        carla_map=carla_map,
                        traffic_manager=traffic_manager,
                        candidate_spawn_indices=candidate_spawn_indices,
                        spawn_index=args.spawn_index + case_counter,
                        spawn_z_offset=args.spawn_z_offset,
                        initial_gap_m=float(d_init),
                        lead_target_speed_mps=float(v_lead),
                        ego_target_speed_mps=float(args.ego_target_speed),
                        dt=args.dt,
                        speed_settle_tol_mps=args.speed_settle_tol,
                        speed_settle_hysteresis_mps=args.speed_settle_hysteresis,
                        settle_ticks_required=args.settle_ticks,
                        max_settle_time_s=args.max_settle_time,
                        ego_model=args.ego_model,
                        lead_model=args.lead_model,
                        repeat_idx=repeat_idx,
                        case_idx=case_counter,
                    )
                    trials.append(tr)

                    done += 1
                    if done % progress_every == 0 or done == total_trials:
                        elapsed = max(1e-9, time.time() - started)
                        rate = done / elapsed
                        rem = (total_trials - done) / max(1e-9, rate)
                        print(
                            f"Progress {done}/{total_trials} ({100.0 * done / total_trials:.1f}%) "
                            f"| elapsed {elapsed/60.0:.1f} min | ETA {rem/60.0:.1f} min"
                        )

        aggregates = aggregate_trials(trials, repeats_per_point=max(1, args.repeats_per_point))
        grids = to_grid(lead_values, init_gap_values, aggregates)
        inverse_grid = build_inverse_grid(
            lead_values=lead_values,
            init_gap_values=init_gap_values,
            final_gap_grid=grids["final_gap"],
            target_gap_values=target_gap_values,
        )
        mono_rows = monotonicity_report(lead_values, init_gap_values, grids["final_gap"])

        raw_csv = out_root / "raw_trials.csv"
        forward_csv = out_root / "forward_map.csv"
        inverse_csv = out_root / "inverse_map.csv"
        mono_csv = out_root / "monotonicity_report.csv"

        save_csv(
            raw_csv,
            fieldnames=[
                "repeat_idx",
                "case_idx",
                "initial_gap_m",
                "lead_target_speed_mps",
                "ego_target_speed_mps",
                "final_gap_m",
                "delta_gap_m",
                "final_lead_speed_mps",
                "final_ego_speed_mps",
                "settled",
                "collision",
                "settle_time_s",
                "ticks_executed",
                "spawn_ok",
                "error",
            ],
            rows=[
                {
                    "repeat_idx": r.repeat_idx,
                    "case_idx": r.case_idx,
                    "initial_gap_m": r.initial_gap_m,
                    "lead_target_speed_mps": r.lead_target_speed_mps,
                    "ego_target_speed_mps": r.ego_target_speed_mps,
                    "final_gap_m": r.final_gap_m,
                    "delta_gap_m": r.delta_gap_m,
                    "final_lead_speed_mps": r.final_lead_speed_mps,
                    "final_ego_speed_mps": r.final_ego_speed_mps,
                    "settled": int(r.settled),
                    "collision": int(r.collision),
                    "settle_time_s": r.settle_time_s,
                    "ticks_executed": r.ticks_executed,
                    "spawn_ok": int(r.spawn_ok),
                    "error": r.error,
                }
                for r in trials
            ],
        )

        save_csv(
            forward_csv,
            fieldnames=[
                "initial_gap_m",
                "lead_target_speed_mps",
                "ego_target_speed_mps",
                "repeats",
                "success_rate",
                "settled_count",
                "final_gap_mean_m",
                "final_gap_std_m",
                "delta_gap_mean_m",
                "delta_gap_std_m",
                "settle_time_mean_s",
                "settle_time_std_s",
                "final_lead_speed_mean_mps",
                "final_ego_speed_mean_mps",
            ],
            rows=[
                {
                    "initial_gap_m": a.initial_gap_m,
                    "lead_target_speed_mps": a.lead_target_speed_mps,
                    "ego_target_speed_mps": a.ego_target_speed_mps,
                    "repeats": a.repeats,
                    "success_rate": a.success_rate,
                    "settled_count": a.settled_count,
                    "final_gap_mean_m": a.final_gap_mean_m,
                    "final_gap_std_m": a.final_gap_std_m,
                    "delta_gap_mean_m": a.delta_gap_mean_m,
                    "delta_gap_std_m": a.delta_gap_std_m,
                    "settle_time_mean_s": a.settle_time_mean_s,
                    "settle_time_std_s": a.settle_time_std_s,
                    "final_lead_speed_mean_mps": a.final_lead_speed_mean_mps,
                    "final_ego_speed_mean_mps": a.final_ego_speed_mean_mps,
                }
                for a in aggregates
            ],
        )

        inv_rows: List[Dict[str, Any]] = []
        for i, v_lead in enumerate(lead_values):
            for j, d_target in enumerate(target_gap_values):
                inv_rows.append(
                    {
                        "lead_target_speed_mps": float(v_lead),
                        "target_final_gap_m": float(d_target),
                        "required_initial_gap_m": float(inverse_grid[i, j]),
                    }
                )

        save_csv(
            inverse_csv,
            fieldnames=["lead_target_speed_mps", "target_final_gap_m", "required_initial_gap_m"],
            rows=inv_rows,
        )

        save_csv(
            mono_csv,
            fieldnames=[
                "lead_target_speed_mps",
                "n_points",
                "nondecreasing_ratio",
                "min_local_slope",
                "max_local_slope",
            ],
            rows=mono_rows,
        )

        if plt is not None:
            _imshow_heatmap(
                out_path=out_root / "forward_gap_heatmap.png",
                x_vals=init_gap_values,
                y_vals=lead_values,
                z_grid=grids["final_gap"],
                title="Forward Map: Settled Gap d_final [m]",
                xlabel="Initial gap d_init [m]",
                ylabel="Lead target speed v_lead* [m/s]",
                colorbar_label="d_final [m]",
                cmap_name="viridis",
            )
            _imshow_heatmap(
                out_path=out_root / "delta_gap_heatmap.png",
                x_vals=init_gap_values,
                y_vals=lead_values,
                z_grid=grids["delta_gap"],
                title="Gap Shift Map: d_final - d_init [m]",
                xlabel="Initial gap d_init [m]",
                ylabel="Lead target speed v_lead* [m/s]",
                colorbar_label="delta gap [m]",
                cmap_name="coolwarm",
            )
            _imshow_heatmap(
                out_path=out_root / "settle_time_heatmap.png",
                x_vals=init_gap_values,
                y_vals=lead_values,
                z_grid=grids["settle_time"],
                title="Settling Time Map [s]",
                xlabel="Initial gap d_init [m]",
                ylabel="Lead target speed v_lead* [m/s]",
                colorbar_label="settle time [s]",
                cmap_name="magma",
            )
            _imshow_heatmap(
                out_path=out_root / "settle_success_rate_heatmap.png",
                x_vals=init_gap_values,
                y_vals=lead_values,
                z_grid=grids["success_rate"],
                title="Success Rate Map (settled, no collision)",
                xlabel="Initial gap d_init [m]",
                ylabel="Lead target speed v_lead* [m/s]",
                colorbar_label="success rate",
                cmap_name="cividis",
                vmin=0.0,
                vmax=1.0,
            )
            _imshow_heatmap(
                out_path=out_root / "inverse_required_dinit_heatmap.png",
                x_vals=target_gap_values,
                y_vals=lead_values,
                z_grid=inverse_grid,
                title="Inverse Map: required d_init for target d_final",
                xlabel="Target final gap d_target [m]",
                ylabel="Lead target speed v_lead* [m/s]",
                colorbar_label="required d_init [m]",
                cmap_name="plasma",
            )
            plot_monotonic_slices(
                out_path=out_root / "monotonic_slices.png",
                init_gap_values=init_gap_values,
                lead_values=lead_values,
                final_gap_grid=grids["final_gap"],
            )

        settle_ok = int(np.sum(np.isfinite(grids["final_gap"])))
        total_cells = int(grids["final_gap"].size)

        manifest = {
            "args": vars(args),
            "output_dir": str(out_root.resolve()),
            "timestamp": datetime.now().isoformat(),
            "n_lead_values": int(len(lead_values)),
            "n_init_gap_values": int(len(init_gap_values)),
            "repeats_per_point": int(args.repeats_per_point),
            "total_trials": int(total_trials),
            "forward_cells_with_data": settle_ok,
            "forward_cells_total": total_cells,
            "raw_trials_csv": str(raw_csv),
            "forward_map_csv": str(forward_csv),
            "inverse_map_csv": str(inverse_csv),
            "monotonicity_csv": str(mono_csv),
            "carla_server_log": str(carla_log_path) if carla_log_path else None,
            "plots_created": plt is not None,
        }
        with (out_root / "run_manifest.json").open("w") as fh:
            json.dump(manifest, fh, indent=2)

        print("=" * 72)
        print("Calibration completed")
        print(f"Forward map cells with data: {settle_ok}/{total_cells}")
        print(f"Raw trials: {raw_csv}")
        print(f"Forward map: {forward_csv}")
        print(f"Inverse map: {inverse_csv}")
        print(f"Monotonic report: {mono_csv}")
        if plt is None:
            print("WARNING: matplotlib unavailable, plots were skipped.")
        else:
            print("Plots saved in output directory.")

    finally:
        try:
            traffic_manager.set_synchronous_mode(False)
        except Exception:
            pass
        try:
            set_async_mode(world)
        except Exception:
            pass
        stop_carla_server(carla_proc)


if __name__ == "__main__":
    main()
