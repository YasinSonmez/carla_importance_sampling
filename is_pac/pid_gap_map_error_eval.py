#!/usr/bin/env python3
"""
Evaluate inverse PID gap map accuracy in CARLA.

Workflow:
- Load inverse map CSV: (v_lead_target, d_target_final) -> required d_init.
- Sample off-grid test conditions (continuous v_lead_target, d_target_final).
- Interpolate required d_init from the inverse map.
- Run CARLA trial with PID speed control and measured settle condition.
- Report distance error e_d = d_final_measured - d_target_final.
- Report relative-speed error e_v = (v_rel_final - v_rel_target), where
    v_rel = v_lead - v_ego.

Outputs:
- eval_trials.csv
- eval_summary.json
- eval_error_hist.png (2 subplots: distance, velocity)
- eval_error_vs_target.png (2 subplots: distance, velocity)
- eval_error_vs_speed.png (2 subplots: distance, velocity)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
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
import sys

if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from pid_gap_map_calibration import run_single_trial  # pylint: disable=wrong-import-position
from carla_acc_warmstart import (  # pylint: disable=wrong-import-position
    build_weather,
    clear_stale_actors,
    connect_world,
    find_straight_spawn_indices,
    is_rpc_open,
    load_world_with_town_fallback,
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
    raise SystemExit("Could not import carla Python API.") from exc


@dataclass
class EvalCase:
    case_idx: int
    lead_target_speed_mps: float
    target_final_gap_m: float
    required_initial_gap_m: float
    final_gap_m: float
    gap_error_m: float
    abs_gap_error_m: float
    final_lead_speed_mps: float
    final_ego_speed_mps: float
    target_rel_speed_mps: float
    final_rel_speed_mps: float
    rel_speed_error_mps: float
    abs_rel_speed_error_mps: float
    settled: bool
    collision: bool
    spawn_ok: bool
    ticks_executed: int
    error: str


class InverseMapInterpolator:
    def __init__(self, inverse_csv: Path) -> None:
        rows = list(csv.DictReader(inverse_csv.open()))
        if not rows:
            raise ValueError(f"inverse map is empty: {inverse_csv}")

        self.lead_values = sorted({float(r["lead_target_speed_mps"]) for r in rows})
        self.target_gap_values = sorted({float(r["target_final_gap_m"]) for r in rows})

        lead_idx = {v: i for i, v in enumerate(self.lead_values)}
        gap_idx = {v: j for j, v in enumerate(self.target_gap_values)}

        self.grid = np.full((len(self.lead_values), len(self.target_gap_values)), np.nan, dtype=float)
        for r in rows:
            v = float(r["lead_target_speed_mps"])
            g = float(r["target_final_gap_m"])
            d = r["required_initial_gap_m"]
            try:
                d_val = float(d)
            except Exception:
                d_val = float("nan")
            self.grid[lead_idx[v], gap_idx[g]] = d_val

        self.v_min = float(min(self.lead_values))
        self.v_max = float(max(self.lead_values))
        self.g_min = float(min(self.target_gap_values))
        self.g_max = float(max(self.target_gap_values))

    def _interp_on_row(self, row_idx: int, target_gap: float) -> Optional[float]:
        x = np.asarray(self.target_gap_values, dtype=float)
        y = self.grid[row_idx, :]
        mask = np.isfinite(y)
        if int(np.sum(mask)) < 2:
            return None
        x_m = x[mask]
        y_m = y[mask]
        if target_gap < x_m[0] or target_gap > x_m[-1]:
            return None
        return float(np.interp(target_gap, x_m, y_m))

    def query(self, lead_target_speed: float, target_gap: float) -> Optional[float]:
        if lead_target_speed < self.v_min or lead_target_speed > self.v_max:
            return None
        if target_gap < self.g_min or target_gap > self.g_max:
            return None

        lead = np.asarray(self.lead_values, dtype=float)
        i_hi = int(np.searchsorted(lead, lead_target_speed, side="left"))
        if i_hi <= 0:
            i0 = i1 = 0
        elif i_hi >= len(lead):
            i0 = i1 = len(lead) - 1
        else:
            i0 = i_hi - 1
            i1 = i_hi

        d0 = self._interp_on_row(i0, target_gap)
        d1 = self._interp_on_row(i1, target_gap)

        if d0 is None and d1 is None:
            return None
        if i0 == i1:
            return d0 if d0 is not None else d1
        if d0 is None:
            return d1
        if d1 is None:
            return d0

        v0 = lead[i0]
        v1 = lead[i1]
        if abs(v1 - v0) < 1e-12:
            return d0
        alpha = (lead_target_speed - v0) / (v1 - v0)
        return float((1.0 - alpha) * d0 + alpha * d1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate inverse PID gap map prediction error in CARLA")

    parser.add_argument("--inverse-map-csv", required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2220)
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
    parser.add_argument("--video", dest="video", action="store_true")
    parser.add_argument("--no-video", dest="video", action="store_false")

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
    parser.add_argument("--eval-cases", type=int, default=120)
    parser.add_argument("--max-sample-attempts", type=int, default=6000)

    parser.add_argument("--speed-settle-tol", type=float, default=0.2)
    parser.add_argument("--speed-settle-hysteresis", type=float, default=0.08)
    parser.add_argument("--settle-ticks", type=int, default=8)
    parser.add_argument("--max-settle-time", type=float, default=10.0)

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


def _plot_hist(path: Path, gap_errors: np.ndarray, rel_speed_errors: np.ndarray) -> None:
    if plt is None:
        return
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8.8), dpi=160)

    axes[0].hist(gap_errors, bins=30, color="#1f77b4", alpha=0.85)
    axes[0].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_title("Distance Error Histogram (d_final - d_target)")
    axes[0].set_xlabel("Distance error [m]")
    axes[0].set_ylabel("Count")
    axes[0].grid(alpha=0.25)

    axes[1].hist(rel_speed_errors, bins=30, color="#ff7f0e", alpha=0.85)
    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_title("Relative-Speed Error Histogram (v_rel_final - v_rel_target)")
    axes[1].set_xlabel("Velocity error [m/s]")
    axes[1].set_ylabel("Count")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_scatter(
    path: Path,
    x: np.ndarray,
    gap_err: np.ndarray,
    rel_speed_err: np.ndarray,
    title: str,
    xlabel: str,
) -> None:
    if plt is None:
        return
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8.8), dpi=160, sharex=True)

    axes[0].scatter(x, gap_err, s=20, alpha=0.75)
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Distance error [m]")
    axes[0].grid(alpha=0.25)

    axes[1].scatter(x, rel_speed_err, s=20, alpha=0.75, color="#ff7f0e")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Velocity error [m/s]")
    axes[1].grid(alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(path)
    plt.close(fig)


def _nan_to_none(v: float) -> Optional[float]:
    if v is None:
        return None
    if math.isnan(v):
        return None
    return float(v)


def main() -> None:
    args = parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    inv_path = Path(args.inverse_map_csv)
    if not inv_path.exists():
        raise SystemExit(f"Inverse map csv not found: {inv_path}")

    interp = InverseMapInterpolator(inv_path)

    if args.output_dir:
        out_root = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = Path("runs") / f"pid_gap_eval_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("CARLA PID Inverse Map Error Evaluation")
    print("=" * 72)
    print(f"Output dir: {out_root.resolve()}")
    print(f"Inverse map: {inv_path}")
    print(f"Requested eval cases: {args.eval_cases}")

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

    world.set_weather(build_weather(args))
    for tl in world.get_actors().filter("traffic.traffic_light"):
        tl.set_state(carla.TrafficLightState.Green)
        tl.freeze(True)

    world.tick()
    clear_stale_actors(client, world)

    if args.gpu_warmup:
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

    cases: List[Tuple[float, float, float]] = []
    attempts = 0
    while len(cases) < args.eval_cases and attempts < args.max_sample_attempts:
        attempts += 1

        v = rng.uniform(interp.v_min, interp.v_max)
        g = rng.uniform(interp.g_min, interp.g_max)

        # Nudge away from exact grid lines for off-grid validation.
        nearest_v = min(interp.lead_values, key=lambda x: abs(x - v))
        nearest_g = min(interp.target_gap_values, key=lambda x: abs(x - g))
        if abs(nearest_v - v) < 1e-4:
            v = min(interp.v_max - 1e-4, v + 0.11)
        if abs(nearest_g - g) < 1e-4:
            g = min(interp.g_max - 1e-4, g + 0.13)

        d_init = interp.query(v, g)
        if d_init is None or not np.isfinite(d_init):
            continue

        cases.append((float(v), float(g), float(d_init)))

    if len(cases) < args.eval_cases:
        print(
            f"WARNING: only generated {len(cases)} valid eval cases out of requested {args.eval_cases}."
        )

    started = datetime.now().isoformat()
    eval_rows: List[EvalCase] = []

    try:
        for i, (v_lead, g_target, d_init) in enumerate(cases, start=1):
            tr = run_single_trial(
                world=world,
                carla_map=carla_map,
                traffic_manager=traffic_manager,
                candidate_spawn_indices=candidate_spawn_indices,
                spawn_index=args.spawn_index + i,
                spawn_z_offset=args.spawn_z_offset,
                initial_gap_m=d_init,
                lead_target_speed_mps=v_lead,
                ego_target_speed_mps=args.ego_target_speed,
                dt=args.dt,
                speed_settle_tol_mps=args.speed_settle_tol,
                speed_settle_hysteresis_mps=args.speed_settle_hysteresis,
                settle_ticks_required=args.settle_ticks,
                max_settle_time_s=args.max_settle_time,
                ego_model=args.ego_model,
                lead_model=args.lead_model,
                repeat_idx=0,
                case_idx=i,
            )

            if np.isfinite(tr.final_gap_m):
                e = float(tr.final_gap_m - g_target)
                ae = float(abs(e))
            else:
                e = float("nan")
                ae = float("nan")

            if np.isfinite(tr.final_lead_speed_mps) and np.isfinite(tr.final_ego_speed_mps):
                target_rel_speed = float(v_lead - args.ego_target_speed)
                final_rel_speed = float(tr.final_lead_speed_mps - tr.final_ego_speed_mps)
                v_err = float(final_rel_speed - target_rel_speed)
                av_err = float(abs(v_err))
            else:
                target_rel_speed = float("nan")
                final_rel_speed = float("nan")
                v_err = float("nan")
                av_err = float("nan")

            eval_rows.append(
                EvalCase(
                    case_idx=i,
                    lead_target_speed_mps=v_lead,
                    target_final_gap_m=g_target,
                    required_initial_gap_m=d_init,
                    final_gap_m=float(tr.final_gap_m),
                    gap_error_m=e,
                    abs_gap_error_m=ae,
                    final_lead_speed_mps=float(tr.final_lead_speed_mps),
                    final_ego_speed_mps=float(tr.final_ego_speed_mps),
                    target_rel_speed_mps=target_rel_speed,
                    final_rel_speed_mps=final_rel_speed,
                    rel_speed_error_mps=v_err,
                    abs_rel_speed_error_mps=av_err,
                    settled=bool(tr.settled),
                    collision=bool(tr.collision),
                    spawn_ok=bool(tr.spawn_ok),
                    ticks_executed=int(tr.ticks_executed),
                    error=tr.error,
                )
            )

            if i % max(10, max(1, len(cases) // 20)) == 0 or i == len(cases):
                print(f"Eval progress {i}/{len(cases)}")

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

    eval_csv = out_root / "eval_trials.csv"
    with eval_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "case_idx",
                "lead_target_speed_mps",
                "target_final_gap_m",
                "required_initial_gap_m",
                "final_gap_m",
                "gap_error_m",
                "abs_gap_error_m",
                "final_lead_speed_mps",
                "final_ego_speed_mps",
                "target_rel_speed_mps",
                "final_rel_speed_mps",
                "rel_speed_error_mps",
                "abs_rel_speed_error_mps",
                "settled",
                "collision",
                "spawn_ok",
                "ticks_executed",
                "error",
            ],
        )
        writer.writeheader()
        for r in eval_rows:
            writer.writerow(
                {
                    "case_idx": r.case_idx,
                    "lead_target_speed_mps": r.lead_target_speed_mps,
                    "target_final_gap_m": r.target_final_gap_m,
                    "required_initial_gap_m": r.required_initial_gap_m,
                    "final_gap_m": r.final_gap_m,
                    "gap_error_m": r.gap_error_m,
                    "abs_gap_error_m": r.abs_gap_error_m,
                    "final_lead_speed_mps": r.final_lead_speed_mps,
                    "final_ego_speed_mps": r.final_ego_speed_mps,
                    "target_rel_speed_mps": r.target_rel_speed_mps,
                    "final_rel_speed_mps": r.final_rel_speed_mps,
                    "rel_speed_error_mps": r.rel_speed_error_mps,
                    "abs_rel_speed_error_mps": r.abs_rel_speed_error_mps,
                    "settled": int(r.settled),
                    "collision": int(r.collision),
                    "spawn_ok": int(r.spawn_ok),
                    "ticks_executed": r.ticks_executed,
                    "error": r.error,
                }
            )

    good = [r for r in eval_rows if r.settled and (not r.collision) and np.isfinite(r.gap_error_m)]
    errs = np.asarray([r.gap_error_m for r in good], dtype=float)
    abs_errs = np.abs(errs)
    vel_errs = np.asarray([r.rel_speed_error_mps for r in good if np.isfinite(r.rel_speed_error_mps)], dtype=float)
    abs_vel_errs = np.abs(vel_errs)

    summary: Dict[str, Any] = {
        "timestamp_start": started,
        "timestamp_end": datetime.now().isoformat(),
        "inverse_map_csv": str(inv_path),
        "args": vars(args),
        "n_requested": int(args.eval_cases),
        "n_generated": int(len(cases)),
        "n_completed": int(len(eval_rows)),
        "n_good": int(len(good)),
        "n_settled": int(sum(1 for r in eval_rows if r.settled)),
        "n_collision": int(sum(1 for r in eval_rows if r.collision)),
        "n_spawn_fail": int(sum(1 for r in eval_rows if not r.spawn_ok)),
        "mae_m": _nan_to_none(float(np.mean(abs_errs))) if abs_errs.size > 0 else None,
        "rmse_m": _nan_to_none(float(np.sqrt(np.mean(errs * errs)))) if errs.size > 0 else None,
        "bias_m": _nan_to_none(float(np.mean(errs))) if errs.size > 0 else None,
        "median_abs_err_m": _nan_to_none(float(np.median(abs_errs))) if abs_errs.size > 0 else None,
        "p90_abs_err_m": _nan_to_none(float(np.percentile(abs_errs, 90.0))) if abs_errs.size > 0 else None,
        "p95_abs_err_m": _nan_to_none(float(np.percentile(abs_errs, 95.0))) if abs_errs.size > 0 else None,
        "max_abs_err_m": _nan_to_none(float(np.max(abs_errs))) if abs_errs.size > 0 else None,
        "vel_mae_mps": _nan_to_none(float(np.mean(abs_vel_errs))) if abs_vel_errs.size > 0 else None,
        "vel_rmse_mps": _nan_to_none(float(np.sqrt(np.mean(vel_errs * vel_errs)))) if vel_errs.size > 0 else None,
        "vel_bias_mps": _nan_to_none(float(np.mean(vel_errs))) if vel_errs.size > 0 else None,
        "vel_median_abs_err_mps": _nan_to_none(float(np.median(abs_vel_errs))) if abs_vel_errs.size > 0 else None,
        "vel_p90_abs_err_mps": _nan_to_none(float(np.percentile(abs_vel_errs, 90.0))) if abs_vel_errs.size > 0 else None,
        "vel_p95_abs_err_mps": _nan_to_none(float(np.percentile(abs_vel_errs, 95.0))) if abs_vel_errs.size > 0 else None,
        "vel_max_abs_err_mps": _nan_to_none(float(np.max(abs_vel_errs))) if abs_vel_errs.size > 0 else None,
        "eval_csv": str(eval_csv),
    }

    if plt is not None and abs_errs.size > 0 and abs_vel_errs.size > 0:
        good_plot = [r for r in good if np.isfinite(r.rel_speed_error_mps)]
        x_gap = np.asarray([r.target_final_gap_m for r in good_plot], dtype=float)
        x_speed = np.asarray([r.lead_target_speed_mps for r in good_plot], dtype=float)
        gap_err_plot = np.asarray([r.gap_error_m for r in good_plot], dtype=float)
        vel_err_plot = np.asarray([r.rel_speed_error_mps for r in good_plot], dtype=float)

        _plot_hist(out_root / "eval_error_hist.png", gap_err_plot, vel_err_plot)
        _plot_scatter(
            out_root / "eval_error_vs_target.png",
            x_gap,
            gap_err_plot,
            vel_err_plot,
            title="Error vs Target Gap",
            xlabel="Target final gap [m]",
        )
        _plot_scatter(
            out_root / "eval_error_vs_speed.png",
            x_speed,
            gap_err_plot,
            vel_err_plot,
            title="Error vs Lead Target Speed",
            xlabel="Lead target speed [m/s]",
        )

    with (out_root / "eval_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    print("=" * 72)
    print("Evaluation completed")
    print(f"Cases completed: {len(eval_rows)}")
    print(f"Good cases (settled, no collision): {len(good)}")
    if abs_errs.size > 0:
        print(f"MAE: {float(np.mean(abs_errs)):.4f} m")
        print(f"RMSE: {float(np.sqrt(np.mean(errs * errs))):.4f} m")
        print(f"P95 |error|: {float(np.percentile(abs_errs, 95.0)):.4f} m")
    if abs_vel_errs.size > 0:
        print(f"Velocity MAE: {float(np.mean(abs_vel_errs)):.4f} m/s")
        print(f"Velocity RMSE: {float(np.sqrt(np.mean(vel_errs * vel_errs))):.4f} m/s")
        print(f"Velocity P95 |error|: {float(np.percentile(abs_vel_errs, 95.0)):.4f} m/s")
    print(f"Eval CSV: {eval_csv}")
    print(f"Summary: {out_root / 'eval_summary.json'}")


if __name__ == "__main__":
    main()
