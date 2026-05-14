#!/usr/bin/env python3
"""
Standalone CARLA ACC-style car-following runner for IS_PAC workflows.

Goal:
- Reproduce the IS_PAC car-following initial-condition semantics in CARLA.
- Avoid SHARC runtime dependencies (this script does not call SHARC).
- Support rigorous warm-start to target initial gap/speeds.
- Support deterministic synchronous stepping and optional video recording.

Example (single initial condition):
  python carla_acc_warmstart.py \
    --host localhost --port 2010 \
    --h0 20.0 --v-lead0 6.0 --v-ego0 5.0 \
    --t-max 10.0 --dt 0.05 \
    --video

Example (many samples from IS_PAC-style distribution):
  python carla_acc_warmstart.py \
    --host localhost --port 2010 \
    --num-samples 100 --sample-mode is_pac \
    --seed 456 --output-dir runs/is_pac_sampling

Notes:
- Run inside the CARLA container/conda env where `import carla` works.
- For video in headless mode, start CARLA with GPU rendering (xvfb-run) and
  avoid -nullrhi.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import carla  # type: ignore
except ImportError as exc:
    raise SystemExit(
        "Could not import carla Python API. Activate the CARLA env first "
        "(for this repo usually: conda activate carla)."
    ) from exc

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None


# IS_PAC example_carfollowing defaults
VF_INIT_DEFAULT = 5.0
ACCEL_DEFAULT = 4.9
DRAG_B_DEFAULT = 1.0
DT_DEFAULT = 0.05
T_MAX_DEFAULT = 10.0
MU_V_MEAN_DEFAULT = 5.0
MU_V_STD_DEFAULT = 2.0
D_STOP_MEDIAN_DEFAULT = 2.0
TAU_MEDIAN_DEFAULT = 1.5
SIGMA_GAP_DEFAULT = 0.8


@dataclass
class InitialCondition:
    h0: float
    v_lead0: float
    v_ego0: float


@dataclass
class EpisodeResult:
    episode_index: int
    h0_request: float
    v_lead0_request: float
    v_ego0_request: float
    h0_achieved: float
    v_lead0_achieved: float
    v_ego0_achieved: float
    failure_indicator: float
    collision: bool
    min_gap_m: float
    n_steps_executed: int
    trajectory_csv: str
    summary_json: str
    video_mp4: Optional[str]
    recorder_log: Optional[str]


class LongitudinalPID:
    """Simple longitudinal PID for target-speed tracking."""

    def __init__(
        self,
        kp: float = 0.45,
        ki: float = 0.05,
        kd: float = 0.02,
        max_throttle: float = 0.85,
        max_brake: float = 0.95,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_throttle = max_throttle
        self.max_brake = max_brake
        self._integral = 0.0
        self._prev_error = 0.0

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0

    def step(self, target_speed: float, current_speed: float, dt: float) -> Tuple[float, float]:
        error = target_speed - current_speed
        self._integral += error * dt
        deriv = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error

        u = self.kp * error + self.ki * self._integral + self.kd * deriv
        if u >= 0.0:
            throttle = min(u, self.max_throttle)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(-u, self.max_brake)
        return throttle, brake


class CollisionMonitor:
    def __init__(self) -> None:
        self.collided = False
        self.events = 0

    def callback(self, _event: "carla.CollisionEvent") -> None:
        self.collided = True
        self.events += 1


class FrameRecorder:
    """Attach camera to ego, save frames, optionally encode MP4."""

    def __init__(
        self,
        world: "carla.World",
        ego_vehicle: "carla.Vehicle",
        output_dir: Path,
        width: int,
        height: int,
        prefix: str,
        overlay_enabled: bool,
        record_immediately: bool = True,
    ) -> None:
        self.output_dir = output_dir
        self.prefix = prefix
        self.width = int(width)
        self.height = int(height)
        self.frames_dir = output_dir / f"{prefix}_frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._frame_count = 0
        self._overlay_enabled = bool(overlay_enabled and cv2 is not None)
        self._overlay_state: Dict[str, Any] = {}
        self._overlay_by_frame: Dict[int, Dict[str, Any]] = {}
        self._recording_enabled = bool(record_immediately)
        self._capture_after_frame: Optional[int] = None

        if overlay_enabled and cv2 is None:
            print("[video] OpenCV not available; disabling HUD overlay.")

        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(self.width))
        bp.set_attribute("image_size_y", str(self.height))
        bp.set_attribute("fov", "90")

        tf = carla.Transform(
            carla.Location(x=-6.0, z=3.0),
            carla.Rotation(pitch=-15.0),
        )
        self.camera = world.spawn_actor(bp, tf, attach_to=ego_vehicle)
        self.camera.listen(self._on_image)

    def start_capture(self, reset_counter: bool = False, capture_after_frame: Optional[int] = None) -> None:
        if reset_counter:
            for jpg in self.frames_dir.glob(f"{self.prefix}_*.jpg"):
                try:
                    jpg.unlink()
                except Exception:
                    pass
            self._frame_count = 0
        self._capture_after_frame = capture_after_frame
        self._recording_enabled = True

    def update_overlay(self, state: Dict[str, Any], frame_id: Optional[int] = None) -> None:
        payload = dict(state)
        if frame_id is None:
            self._overlay_state = payload
            return
        self._overlay_by_frame[int(frame_id)] = payload

    def _draw_overlay(self, frame_bgr: "np.ndarray", state: Dict[str, Any]) -> None:
        if not self._overlay_enabled or cv2 is None:
            return

        panel_w = min(520, max(300, int(self.width * 0.30)))
        panel_h = min(230, max(170, int(self.height * 0.22)))
        x0, y0 = 20, 20
        x1, y1 = x0 + panel_w, y0 + panel_h

        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (245, 245, 245), thickness=2)
        cv2.addWeighted(overlay, 0.72, frame_bgr, 0.28, 0.0, frame_bgr)

        t_s = state.get("time_s", 0.0)
        gap = state.get("gap_m", float("nan"))
        v_ego = state.get("ego_speed_mps", float("nan"))
        v_lead = state.get("lead_speed_mps", float("nan"))

        lines = [
            f"t      : {t_s:6.2f} s",
            f"gap    : {gap:7.3f} m",
            f"ego v  : {v_ego:6.3f} m/s",
            f"lead v : {v_lead:6.3f} m/s",
        ]

        y_text = y0 + 42
        for text in lines:
            color = (245, 245, 245)
            scale = 0.84
            # Text shadow for readability.
            cv2.putText(
                frame_bgr,
                text,
                (x0 + 22, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (0, 0, 0),
                thickness=3,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                frame_bgr,
                text,
                (x0 + 20, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            y_text += 42

    def _on_image(self, image: "carla.Image") -> None:
        if not self._recording_enabled:
            return
        if self._capture_after_frame is not None and int(image.frame) <= int(self._capture_after_frame):
            return

        path = self.frames_dir / f"{self.prefix}_{self._frame_count:06d}.jpg"
        if self._overlay_enabled and cv2 is not None:
            arr = np.frombuffer(image.raw_data, dtype=np.uint8)
            arr = arr.reshape((self.height, self.width, 4))
            frame_bgr = arr[:, :, :3].copy()
            frame_id = int(image.frame)
            state = self._overlay_by_frame.pop(frame_id, self._overlay_state)
            if self._overlay_by_frame:
                stale = [k for k in self._overlay_by_frame.keys() if k < frame_id - 4]
                for k in stale:
                    self._overlay_by_frame.pop(k, None)
            self._draw_overlay(frame_bgr, state)
            cv2.imwrite(str(path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
        else:
            image.save_to_disk(str(path))
        self._frame_count += 1

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def destroy(self) -> None:
        if self.camera is not None:
            self.camera.stop()
            self.camera.destroy()
            self.camera = None

    def make_video(self, fps: int, keep_frames: bool, crf: int, preset: str) -> Optional[str]:
        output_mp4 = self.output_dir / f"{self.prefix}_video.mp4"
        pattern = self.frames_dir / f"{self.prefix}_%06d.jpg"

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(pattern),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(crf),
            "-preset",
            preset,
            str(output_mp4),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            tail = proc.stderr[-500:] if proc.stderr else ""
            print(f"[video] ffmpeg failed: {tail}")
            return None

        if not keep_frames:
            shutil.rmtree(self.frames_dir, ignore_errors=True)
        return str(output_mp4)


def speed_mps(vehicle: "carla.Vehicle") -> float:
    vel = vehicle.get_velocity()
    return math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z)


def planar_forward_xy(transform: "carla.Transform") -> Tuple[float, float]:
    fwd = transform.get_forward_vector()
    norm_xy = math.hypot(fwd.x, fwd.y)
    if norm_xy <= 1e-8:
        return 1.0, 0.0
    return fwd.x / norm_xy, fwd.y / norm_xy


def forward_speed_mps(vehicle: "carla.Vehicle") -> float:
    # Use scalar speed for car-following semantics; heading-projection can under-report
    # during spawn/snap transients on uneven road geometry.
    return speed_mps(vehicle)


def wrap_angle_rad(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def choose_successor(current_wp: "carla.Waypoint", candidates: List["carla.Waypoint"]) -> Optional["carla.Waypoint"]:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    cur_tf = current_wp.transform
    cur_yaw = math.radians(cur_tf.rotation.yaw)
    cur_loc = cur_tf.location

    def key_fn(candidate: "carla.Waypoint") -> Tuple[float, float, float, float, float, float]:
        tf = candidate.transform
        yaw = math.radians(tf.rotation.yaw)
        heading_error = abs(wrap_angle_rad(yaw - cur_yaw))
        same_lane_penalty = 0.0 if candidate.lane_id == current_wp.lane_id else 1.0
        same_road_penalty = 0.0 if candidate.road_id == current_wp.road_id else 1.0
        lateral_offset = abs(
            -(tf.location.x - cur_loc.x) * math.sin(cur_yaw)
            + (tf.location.y - cur_loc.y) * math.cos(cur_yaw)
        )
        return (
            same_lane_penalty,
            same_road_penalty,
            heading_error,
            lateral_offset,
            tf.location.x,
            tf.location.y,
        )

    return min(candidates, key=key_fn)


def advance_waypoint(start_wp: "carla.Waypoint", distance_m: float, step_m: float = 1.0) -> Optional["carla.Waypoint"]:
    if distance_m <= 0.0:
        return start_wp
    wp = start_wp
    advanced = 0.0
    n_steps = max(1, int(math.ceil(distance_m / step_m)))
    for _ in range(n_steps):
        nxt = wp.next(step_m)
        wp = choose_successor(wp, nxt)
        if wp is None:
            return None
        advanced += step_m
        if advanced + 1e-9 >= distance_m:
            return wp
    return wp


def lane_steer(vehicle: "carla.Vehicle", carla_map: "carla.Map", steer_gain: float = 1.2, steer_limit: float = 0.4) -> float:
    tf = vehicle.get_transform()
    wp = carla_map.get_waypoint(
        tf.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if wp is None:
        return 0.0
    target_yaw = math.radians(wp.transform.rotation.yaw)
    current_yaw = math.radians(tf.rotation.yaw)
    yaw_err = wrap_angle_rad(target_yaw - current_yaw)
    steer = steer_gain * yaw_err
    return max(-steer_limit, min(steer_limit, steer))


def bumper_gap_m(lead: "carla.Vehicle", ego: "carla.Vehicle") -> float:
    lead_tf = lead.get_transform()
    ego_tf = ego.get_transform()
    lead_loc = lead_tf.location
    ego_loc = ego_tf.location

    ego_fwd = ego_tf.get_forward_vector()
    rel_x = lead_loc.x - ego_loc.x
    rel_y = lead_loc.y - ego_loc.y
    center_longitudinal = rel_x * ego_fwd.x + rel_y * ego_fwd.y
    center_distance = math.hypot(rel_x, rel_y)
    sign = 1.0 if center_longitudinal >= 0.0 else -1.0

    ego_front = ego.bounding_box.extent.x
    lead_rear = lead.bounding_box.extent.x
    return sign * center_distance - ego_front - lead_rear


def center_distance_for_bumper_gap(ego: "carla.Vehicle", lead: "carla.Vehicle", desired_gap_m: float) -> float:
    ego_front = ego.bounding_box.extent.x
    lead_rear = lead.bounding_box.extent.x
    return max(0.5, desired_gap_m + ego_front + lead_rear)


def destroy_actor_safe(actor: Optional["carla.Actor"]) -> None:
    if actor is None:
        return
    try:
        if isinstance(actor, carla.Sensor):
            actor.stop()
    except Exception:
        pass
    try:
        actor.destroy()
    except Exception:
        pass


def connect_world(host: str, port: int, timeout_s: float) -> Tuple["carla.Client", "carla.World"]:
    client = carla.Client(host, port)
    client.set_timeout(timeout_s)
    world = client.get_world()
    return client, world


def is_rpc_open(host: str, port: int, timeout_s: float = 1.0) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout_s)
        sock.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def wait_for_rpc(host: str, port: int, timeout_s: float) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        if is_rpc_open(host, port, timeout_s=1.0):
            return True
        time.sleep(1.0)
    return False


def locate_carla_binary(carla_root: str) -> Path:
    root = Path(carla_root)
    candidates = [
        root / "CarlaUE4.sh",
        Path("/home/workspace/carla_0.9.16/CarlaUE4.sh"),
        Path("/workspace/CarlaUE4.sh"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Could not locate CarlaUE4.sh. Set --carla-root or start CARLA manually."
    )


def start_carla_server(
    args: argparse.Namespace,
    run_dir: Path,
) -> Tuple[subprocess.Popen, Path]:
    carla_bin = locate_carla_binary(args.carla_root)
    log_path = run_dir / "carla_server.log"

    cmd = [
        str(carla_bin),
        "-RenderOffScreen",
        "-nosound",
        f"-quality-level={args.carla_quality_level}",
        f"-ResX={args.carla_res_x}",
        f"-ResY={args.carla_res_y}",
        f"-carla-rpc-port={args.port}",
    ]

    if args.carla_prefernvidia:
        cmd.insert(1, "-prefernvidia")

    if not args.video:
        cmd.insert(1, "-nullrhi")

    if args.start_with_xvfb:
        cmd = [
            "xvfb-run",
            "--auto-servernum",
            f"--server-args=-screen 0 {args.carla_res_x}x{args.carla_res_y}x{args.xvfb_screen_depth} +extension GLX",
        ] + cmd

    with log_path.open("w") as log_fh:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

    return proc, log_path


def stop_carla_server(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def load_world_with_town_fallback(client: "carla.Client", requested_town: str) -> Tuple["carla.World", str]:
    candidates = [requested_town]

    normalized = requested_town.lower().replace("_", "")
    if normalized in {"town10hd", "town10hdopt", "town10"}:
        candidates.extend(["Town10HD_Opt", "Town10HD"])

    seen = set()
    ordered: List[str] = []
    for c in candidates:
        if c not in seen:
            ordered.append(c)
            seen.add(c)

    last_exc: Optional[Exception] = None
    for town in ordered:
        try:
            return client.load_world(town), town
        except Exception as exc:
            last_exc = exc

    raise RuntimeError(
        f"Failed to load requested town '{requested_town}'. Tried: {ordered}."
    ) from last_exc


def build_weather(args: argparse.Namespace) -> "carla.WeatherParameters":
    presets: Dict[str, "carla.WeatherParameters"] = {
        "clear_noon": carla.WeatherParameters.ClearNoon,
        "wet_sunset": carla.WeatherParameters.WetSunset,
        "wet_cloudy_sunset": carla.WeatherParameters.WetCloudySunset,
        "mid_rain_sunset": carla.WeatherParameters.MidRainSunset,
        "clear_sunset": carla.WeatherParameters.ClearSunset,
    }

    if args.weather_profile == "custom":
        weather = carla.WeatherParameters.WetSunset
        weather.cloudiness = 20.0
        weather.precipitation = 30.0
        weather.precipitation_deposits = 55.0
        weather.wetness = 28.0
        weather.wind_intensity = 4.0
        weather.sun_altitude_angle = 20.0
        weather.sun_azimuth_angle = 38.0
        weather.fog_density = 1.0
        weather.fog_distance = 180.0
        weather.fog_falloff = 0.2
    else:
        weather = presets.get(args.weather_profile, carla.WeatherParameters.WetSunset)

    if args.sun_altitude_angle is not None:
        weather.sun_altitude_angle = float(args.sun_altitude_angle)
    if args.sun_azimuth_angle is not None:
        weather.sun_azimuth_angle = float(args.sun_azimuth_angle)
    if args.wetness is not None:
        weather.wetness = float(args.wetness)

    return weather


def resolve_vehicle_blueprint(
    bp_lib: "carla.BlueprintLibrary",
    requested_id: str,
    fallback_ids: List[str],
) -> "carla.ActorBlueprint":
    candidates = [requested_id] + [c for c in fallback_ids if c != requested_id]
    last_exc: Optional[Exception] = None
    for cid in candidates:
        try:
            return bp_lib.find(cid)
        except Exception as exc:
            last_exc = exc
            continue
    raise RuntimeError(
        f"Could not resolve any vehicle blueprint from candidates: {candidates}"
    ) from last_exc


def set_sync_mode(world: "carla.World", dt: float, no_rendering: bool = False) -> None:
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = dt
    settings.no_rendering_mode = no_rendering
    if settings.max_substeps < 1 or settings.max_substeps > 16:
        settings.max_substeps = 10
    if settings.max_substep_delta_time <= 0.0 or settings.max_substep_delta_time > 0.05:
        settings.max_substep_delta_time = 0.01
    world.apply_settings(settings)


def set_async_mode(world: "carla.World") -> None:
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    settings.no_rendering_mode = False
    world.apply_settings(settings)


def clear_stale_actors(client: "carla.Client", world: "carla.World") -> int:
    actors = world.get_actors()
    stale = [
        a.id
        for a in actors
        if a.type_id.startswith(("vehicle.", "sensor.", "walker.", "controller."))
    ]
    if stale:
        client.apply_batch_sync([carla.command.DestroyActor(aid) for aid in stale], True)
        world.tick()
    return len(stale)


def run_gpu_warmup(world: "carla.World") -> None:
    bp_lib = world.get_blueprint_library()
    vehicle_bp = resolve_vehicle_blueprint(
        bp_lib,
        "vehicle.lincoln.mkz_2020",
        fallback_ids=["vehicle.tesla.model3", "vehicle.audi.tt"],
    )
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        return

    vehicle = None
    camera = None
    try:
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0])
        if vehicle is None:
            return
        world.tick()

        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "1280")
        cam_bp.set_attribute("image_size_y", "720")
        cam_bp.set_attribute("fov", "110")
        cam_tf = carla.Transform(carla.Location(x=-8.0, z=5.0), carla.Rotation(pitch=-20.0))
        camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)
        camera.listen(lambda _img: None)

        for _ in range(20):
            vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0, brake=0.0))
            world.tick()
    finally:
        destroy_actor_safe(camera)
        destroy_actor_safe(vehicle)


def sample_is_pac_initial_conditions(
    n: int,
    rng: random.Random,
    mu_v_mean: float,
    mu_v_std: float,
    d_stop_median: float,
    tau_median: float,
    sigma_gap: float,
    v_ego0: float,
) -> List[InitialCondition]:
    samples: List[InitialCondition] = []
    while len(samples) < n:
        # Truncated normal v >= 0 via rejection sampling.
        v = rng.gauss(mu_v_mean, mu_v_std)
        if v < 0.0:
            continue
        # Lognormal with median = d_stop + v * tau.
        median_gap = max(0.1, d_stop_median + v * tau_median)
        mu_log = math.log(median_gap)
        h = rng.lognormvariate(mu_log, sigma_gap)
        samples.append(InitialCondition(h0=h, v_lead0=v, v_ego0=v_ego0))
    return samples


def prepare_initial_conditions(args: argparse.Namespace, rng: random.Random) -> List[InitialCondition]:
    if args.sample_mode == "fixed":
        return [InitialCondition(h0=args.h0, v_lead0=args.v_lead0, v_ego0=args.v_ego0) for _ in range(args.num_samples)]
    if args.sample_mode == "is_pac":
        return sample_is_pac_initial_conditions(
            n=args.num_samples,
            rng=rng,
            mu_v_mean=args.mu_v_mean,
            mu_v_std=args.mu_v_std,
            d_stop_median=args.d_stop_median,
            tau_median=args.tau_median,
            sigma_gap=args.sigma_gap,
            v_ego0=args.v_ego0,
        )
    raise ValueError(f"Unsupported sample mode: {args.sample_mode}")


def is_straight_lane_segment(
    start_wp: "carla.Waypoint",
    lookahead_m: float,
    step_m: float,
    max_yaw_delta_deg: float,
) -> bool:
    wp = start_wp
    traversed = 0.0
    yaw0 = math.radians(start_wp.transform.rotation.yaw)
    max_yaw_delta_rad = math.radians(max_yaw_delta_deg)

    while traversed < lookahead_m:
        nxt = wp.next(step_m)
        wp = choose_successor(wp, nxt)
        if wp is None:
            return False
        yaw = math.radians(wp.transform.rotation.yaw)
        if abs(wrap_angle_rad(yaw - yaw0)) > max_yaw_delta_rad:
            return False
        traversed += step_m
    return True


def straight_lane_length_m(
    start_wp: "carla.Waypoint",
    max_scan_m: float,
    step_m: float,
    max_yaw_delta_deg: float,
    require_no_junction: bool,
) -> float:
    wp = start_wp
    traversed = 0.0
    yaw0 = math.radians(start_wp.transform.rotation.yaw)
    max_yaw_delta_rad = math.radians(max_yaw_delta_deg)

    if require_no_junction and start_wp.is_junction:
        return 0.0

    while traversed < max_scan_m:
        nxt = wp.next(step_m)
        wp = choose_successor(wp, nxt)
        if wp is None:
            break
        if require_no_junction and wp.is_junction:
            break
        yaw = math.radians(wp.transform.rotation.yaw)
        if abs(wrap_angle_rad(yaw - yaw0)) > max_yaw_delta_rad:
            break
        traversed += step_m

    return traversed


def find_straight_spawn_indices(
    carla_map: "carla.Map",
    lookahead_m: float,
    step_m: float,
    max_yaw_delta_deg: float,
    min_length_m: float,
    scan_max_m: float,
    require_no_junction: bool,
    sort_by_length_desc: bool,
) -> Tuple[List[int], List[Tuple[int, float]]]:
    spawn_points = carla_map.get_spawn_points()
    candidates: List[Tuple[int, float]] = []
    effective_scan_m = max(scan_max_m, min_length_m, lookahead_m)

    for i, sp in enumerate(spawn_points):
        wp = carla_map.get_waypoint(
            sp.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if wp is None:
            continue

        straight_len = straight_lane_length_m(
            start_wp=wp,
            max_scan_m=effective_scan_m,
            step_m=step_m,
            max_yaw_delta_deg=max_yaw_delta_deg,
            require_no_junction=require_no_junction,
        )
        if straight_len + 1e-9 >= min_length_m:
            candidates.append((i, straight_len))

    if sort_by_length_desc:
        candidates.sort(key=lambda p: p[1], reverse=True)
    else:
        candidates.sort(key=lambda p: p[0])

    return [idx for idx, _ in candidates], candidates


def try_spawn_actor_with_z_sweep(
    world: "carla.World",
    blueprint: "carla.ActorBlueprint",
    base_tf: "carla.Transform",
    base_z_offset: float,
) -> Optional["carla.Actor"]:
    # Small Z sweep keeps actors grounded while recovering from occasional spawn collisions.
    z_offsets = [
        base_z_offset,
        base_z_offset + 0.02,
        base_z_offset + 0.05,
        base_z_offset + 0.10,
    ]
    seen = set()
    for dz in z_offsets:
        if dz in seen:
            continue
        seen.add(dz)
        tf = carla.Transform(
            carla.Location(
                x=base_tf.location.x,
                y=base_tf.location.y,
                z=base_tf.location.z + dz,
            ),
            base_tf.rotation,
        )
        actor = world.try_spawn_actor(blueprint, tf)
        if actor is not None:
            return actor
    return None


def make_vehicle_pair(
    world: "carla.World",
    carla_map: "carla.Map",
    spawn_index: int,
    desired_gap_m: float,
    ego_model: str,
    lead_model: str,
    spawn_z_offset: float,
    candidate_spawn_indices: Optional[List[int]] = None,
) -> Tuple["carla.Vehicle", "carla.Vehicle", "carla.Waypoint", "carla.Waypoint"]:
    bp_lib = world.get_blueprint_library()
    ego_bp = resolve_vehicle_blueprint(
        bp_lib,
        ego_model,
        fallback_ids=["vehicle.lincoln.mkz_2020", "vehicle.tesla.model3", "vehicle.audi.tt"],
    )
    lead_bp = resolve_vehicle_blueprint(
        bp_lib,
        lead_model,
        fallback_ids=["vehicle.lincoln.mkz_2020", "vehicle.audi.tt", "vehicle.tesla.model3"],
    )

    if ego_bp.has_attribute("color"):
        ego_bp.set_attribute("color", "0,0,0")
    if lead_bp.has_attribute("color"):
        lead_bp.set_attribute("color", "255,0,0")

    spawn_points = carla_map.get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points available in current CARLA map.")

    if candidate_spawn_indices:
        search_indices = candidate_spawn_indices
    else:
        search_indices = list(range(len(spawn_points)))

    n = len(search_indices)
    if n == 0:
        raise RuntimeError("No candidate spawn indices available for ego/lead placement.")

    for offset in range(n):
        idx = search_indices[(spawn_index + offset) % n]
        start_tf = spawn_points[idx]
        ego_wp = carla_map.get_waypoint(
            start_tf.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if ego_wp is None:
            continue

        lead_wp = advance_waypoint(ego_wp, max(1.0, desired_gap_m + 5.0), step_m=1.0)
        if lead_wp is None:
            continue

        ego_tf = ego_wp.transform
        lead_tf = lead_wp.transform

        ego = try_spawn_actor_with_z_sweep(
            world=world,
            blueprint=ego_bp,
            base_tf=ego_tf,
            base_z_offset=spawn_z_offset,
        )
        if ego is None:
            continue
        lead = try_spawn_actor_with_z_sweep(
            world=world,
            blueprint=lead_bp,
            base_tf=lead_tf,
            base_z_offset=spawn_z_offset,
        )
        if lead is None:
            destroy_actor_safe(ego)
            continue

        exact_center_distance = center_distance_for_bumper_gap(ego, lead, desired_gap_m)
        lead_wp_exact = advance_waypoint(ego_wp, exact_center_distance, step_m=1.0)
        if lead_wp_exact is not None:
            lead_tf_exact = lead_wp_exact.transform
            lead_tf_exact.location.z += spawn_z_offset
            lead.set_transform(lead_tf_exact)
            lead_wp = lead_wp_exact
            world.tick()

        return ego, lead, ego_wp, lead_wp

    raise RuntimeError("Failed to spawn ego/lead pair on same lane with desired initial gap.")


def set_vehicle_kinematic_state(vehicle: "carla.Vehicle", waypoint: "carla.Waypoint", target_speed_mps: float) -> None:
    tf = waypoint.transform
    # Preserve current grounded Z to avoid visible vertical teleports.
    try:
        tf.location.z = vehicle.get_transform().location.z
    except Exception:
        tf.location.z += 0.05
    vehicle.set_transform(tf)
    fwd_x, fwd_y = planar_forward_xy(tf)
    vehicle.set_target_velocity(
        carla.Vector3D(
            x=fwd_x * target_speed_mps,
            y=fwd_y * target_speed_mps,
            z=0.0,
        )
    )
    vehicle.set_target_angular_velocity(carla.Vector3D(x=0.0, y=0.0, z=0.0))


def warm_start_to_initial_condition(
    world: "carla.World",
    carla_map: "carla.Map",
    ego: "carla.Vehicle",
    lead: "carla.Vehicle",
    ego_wp: "carla.Waypoint",
    desired_h0: float,
    desired_v_ego: float,
    desired_v_lead: float,
    dt: float,
    max_warmup_ticks: int,
    gap_tolerance_m: float,
    speed_tolerance_mps: float,
    settle_ticks_required: int,
    gap_feedback_gain: float,
    on_step: Optional[Any] = None,
) -> Tuple[float, float, float, bool]:
    lead_pid = LongitudinalPID()
    ego_pid = LongitudinalPID()
    ok_count = 0

    for _ in range(max_warmup_ticks):
        current_gap = bumper_gap_m(lead, ego)
        v_lead = max(0.0, forward_speed_mps(lead))
        v_ego = max(0.0, forward_speed_mps(ego))

        ego_target = max(0.0, desired_v_ego + gap_feedback_gain * (current_gap - desired_h0))

        if on_step is not None:
            on_step(
                {
                    "phase": "warmstart",
                    "gap_m": current_gap,
                    "lead_speed_mps": v_lead,
                    "ego_speed_mps": v_ego,
                    "time_s": 0.0,
                }
            )

        lead_throttle, lead_brake = lead_pid.step(desired_v_lead, v_lead, dt)
        ego_throttle, ego_brake = ego_pid.step(ego_target, v_ego, dt)

        lead_ctrl = carla.VehicleControl(
            throttle=lead_throttle,
            brake=lead_brake,
            steer=lane_steer(lead, carla_map),
        )
        ego_ctrl = carla.VehicleControl(
            throttle=ego_throttle,
            brake=ego_brake,
            steer=lane_steer(ego, carla_map),
        )

        lead.apply_control(lead_ctrl)
        ego.apply_control(ego_ctrl)
        world.tick()

        current_gap = bumper_gap_m(lead, ego)
        v_lead = max(0.0, forward_speed_mps(lead))
        v_ego = max(0.0, forward_speed_mps(ego))

        gap_ok = abs(current_gap - desired_h0) <= gap_tolerance_m
        lead_ok = abs(v_lead - desired_v_lead) <= speed_tolerance_mps
        ego_ok = abs(v_ego - desired_v_ego) <= speed_tolerance_mps

        if gap_ok and lead_ok and ego_ok:
            ok_count += 1
            if ok_count >= settle_ticks_required:
                return current_gap, v_lead, v_ego, True
        else:
            ok_count = 0

    # Hard correction step: deterministic exact snap on lane, then settle.
    ego_now_wp = carla_map.get_waypoint(
        ego.get_transform().location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if ego_now_wp is None:
        ego_now_wp = ego_wp

    center_distance = center_distance_for_bumper_gap(ego, lead, desired_h0)
    lead_now_wp = advance_waypoint(ego_now_wp, center_distance, step_m=1.0)
    if lead_now_wp is None:
        lead_now_wp = ego_now_wp

    set_vehicle_kinematic_state(ego, ego_now_wp, desired_v_ego)
    set_vehicle_kinematic_state(lead, lead_now_wp, desired_v_lead)

    lead_pid.reset()
    ego_pid.reset()
    for _ in range(max(2, settle_ticks_required)):
        v_lead = max(0.0, forward_speed_mps(lead))
        v_ego = max(0.0, forward_speed_mps(ego))

        lead_throttle, lead_brake = lead_pid.step(desired_v_lead, v_lead, dt)
        ego_throttle, ego_brake = ego_pid.step(desired_v_ego, v_ego, dt)

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

        if on_step is not None:
            current_gap = bumper_gap_m(lead, ego)
            v_lead = max(0.0, forward_speed_mps(lead))
            v_ego = max(0.0, forward_speed_mps(ego))
            on_step(
                {
                    "phase": "warmstart-correction",
                    "gap_m": current_gap,
                    "lead_speed_mps": v_lead,
                    "ego_speed_mps": v_ego,
                    "time_s": 0.0,
                }
            )

    current_gap = bumper_gap_m(lead, ego)
    v_lead = max(0.0, forward_speed_mps(lead))
    v_ego = max(0.0, forward_speed_mps(ego))
    success = (
        abs(current_gap - desired_h0) <= gap_tolerance_m
        and abs(v_lead - desired_v_lead) <= speed_tolerance_mps
        and abs(v_ego - desired_v_ego) <= speed_tolerance_mps
    )
    return current_gap, v_lead, v_ego, success


def enforce_strict_initial_condition(
    world: "carla.World",
    carla_map: "carla.Map",
    ego: "carla.Vehicle",
    lead: "carla.Vehicle",
    reference_ego_wp: Optional["carla.Waypoint"],
    desired_h0: float,
    desired_v_ego: float,
    desired_v_lead: float,
    dt: float,
    settle_ticks: int,
    gap_tolerance_m: float,
    speed_tolerance_mps: float,
    max_iterations: int,
    ground_z_offset: float,
    on_step: Optional[Any] = None,
) -> Tuple[float, float, float, bool]:
    ego_wp = reference_ego_wp
    if ego_wp is None:
        ego_wp = carla_map.get_waypoint(
            ego.get_transform().location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
    if ego_wp is None:
        raise RuntimeError("Ego vehicle not on a driving lane during strict initialization snap.")

    center_distance = center_distance_for_bumper_gap(ego, lead, desired_h0)
    anchor_tf = ego_wp.transform
    anchor_rot = ego_wp.transform.rotation
    anchor_loc = anchor_tf.location
    anchor_loc.z += ground_z_offset

    zero_angular = carla.Vector3D(x=0.0, y=0.0, z=0.0)
    strict_constvel_active = False

    def apply_exact_state(distance_m: float, cmd_v_ego: float, cmd_v_lead: float) -> None:
        ego_tf = carla.Transform(
            carla.Location(x=anchor_loc.x, y=anchor_loc.y, z=anchor_loc.z),
            anchor_rot,
        )
        fwd_x, fwd_y = planar_forward_xy(ego_tf)

        lead_wp_snap = advance_waypoint(ego_wp, max(0.5, distance_m), step_m=1.0)
        if lead_wp_snap is not None:
            lead_z = lead_wp_snap.transform.location.z + ground_z_offset
        else:
            lead_z = ego_tf.location.z

        lead_tf = carla.Transform(
            carla.Location(
                x=ego_tf.location.x + fwd_x * distance_m,
                y=ego_tf.location.y + fwd_y * distance_m,
                z=lead_z,
            ),
            ego_tf.rotation,
        )

        ego.set_transform(ego_tf)
        lead.set_transform(lead_tf)

        ego_vel = carla.Vector3D(
            x=fwd_x * cmd_v_ego,
            y=fwd_y * cmd_v_ego,
            z=0.0,
        )
        lead_vel = carla.Vector3D(
            x=fwd_x * cmd_v_lead,
            y=fwd_y * cmd_v_lead,
            z=0.0,
        )
        ego.set_target_velocity(ego_vel)
        lead.set_target_velocity(lead_vel)
        try:
            ego.set_velocity(ego_vel)
            lead.set_velocity(lead_vel)
        except Exception:
            pass
        ego.set_target_angular_velocity(zero_angular)
        lead.set_target_angular_velocity(zero_angular)

    # Disable lingering constant-velocity mode from previous operations.
    try:
        ego.disable_constant_velocity()
    except Exception:
        pass
    try:
        lead.disable_constant_velocity()
    except Exception:
        pass

    current_gap = bumper_gap_m(lead, ego)
    v_lead = max(0.0, forward_speed_mps(lead))
    v_ego = max(0.0, forward_speed_mps(ego))
    success = False
    cmd_v_lead = max(0.0, desired_v_lead)
    cmd_v_ego = max(0.0, desired_v_ego)

    total_iters = max(max(1, settle_ticks), max(1, max_iterations))
    for i in range(total_iters):
        apply_exact_state(center_distance, cmd_v_ego, cmd_v_lead)

        # Try to keep exact requested speed over the strict-init correction ticks.
        try:
            ego_tf_now = ego.get_transform()
            ego_fwd_x, ego_fwd_y = planar_forward_xy(ego_tf_now)
            ego.enable_constant_velocity(
                carla.Vector3D(
                    x=ego_fwd_x * cmd_v_ego,
                    y=ego_fwd_y * cmd_v_ego,
                    z=0.0,
                )
            )
            lead.enable_constant_velocity(
                carla.Vector3D(
                    x=ego_fwd_x * cmd_v_lead,
                    y=ego_fwd_y * cmd_v_lead,
                    z=0.0,
                )
            )
            strict_constvel_active = True
        except Exception:
            strict_constvel_active = False

        world.tick()

        current_gap = bumper_gap_m(lead, ego)
        v_lead = max(0.0, forward_speed_mps(lead))
        v_ego = max(0.0, forward_speed_mps(ego))
        gap_err = desired_h0 - current_gap

        if on_step is not None:
            on_step(
                {
                    "phase": "strict-init",
                    "time_s": i * dt,
                    "gap_m": current_gap,
                    "lead_speed_mps": v_lead,
                    "ego_speed_mps": v_ego,
                }
            )

        speed_ok = (
            abs(v_lead - desired_v_lead) <= speed_tolerance_mps
            and abs(v_ego - desired_v_ego) <= speed_tolerance_mps
        )
        gap_ok = abs(gap_err) <= gap_tolerance_m
        if gap_ok and speed_ok:
            success = True
            break

        # Correct placement based on measured bumper-gap error.
        center_distance += gap_err

        # Correct commanded speeds using measured forward-speed error.
        lead_speed_err = desired_v_lead - v_lead
        ego_speed_err = desired_v_ego - v_ego
        cmd_v_lead = max(0.0, min(desired_v_lead + 8.0, cmd_v_lead + lead_speed_err))
        cmd_v_ego = max(0.0, min(desired_v_ego + 8.0, cmd_v_ego + ego_speed_err))

    # Finalize exact t=0 state right before rollout/video start (pre-tick).
    apply_exact_state(center_distance, max(0.0, desired_v_ego), max(0.0, desired_v_lead))
    try:
        ego_tf_now = ego.get_transform()
        ego_fwd_x, ego_fwd_y = planar_forward_xy(ego_tf_now)
        ego.enable_constant_velocity(
            carla.Vector3D(
                x=ego_fwd_x * max(0.0, desired_v_ego),
                y=ego_fwd_y * max(0.0, desired_v_ego),
                z=0.0,
            )
        )
        lead.enable_constant_velocity(
            carla.Vector3D(
                x=ego_fwd_x * max(0.0, desired_v_lead),
                y=ego_fwd_y * max(0.0, desired_v_lead),
                z=0.0,
            )
        )
        strict_constvel_active = True
    except Exception:
        strict_constvel_active = False

    current_gap = bumper_gap_m(lead, ego)
    v_lead = max(0.0, forward_speed_mps(lead))
    v_ego = max(0.0, forward_speed_mps(ego))
    return current_gap, v_lead, v_ego, strict_constvel_active


def run_episode(
    args: argparse.Namespace,
    episode_index: int,
    initial_condition: InitialCondition,
    rng: random.Random,
    client: "carla.Client",
    world: "carla.World",
    traffic_manager: "carla.TrafficManager",
    output_dir: Path,
    candidate_spawn_indices: Optional[List[int]],
) -> EpisodeResult:
    carla_map = world.get_map()

    # Ensure clean world between episodes.
    clear_stale_actors(client, world)

    ego = None
    lead = None
    collision_sensor = None
    recorder = None
    recorder_log_path: Optional[str] = None
    video_path: Optional[str] = None

    episode_dir = output_dir / f"episode_{episode_index:04d}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    trajectory_csv_path = episode_dir / "trajectory.csv"
    summary_json_path = episode_dir / "summary.json"

    monitor = CollisionMonitor()

    try:
        ego, lead, ego_wp, _ = make_vehicle_pair(
            world=world,
            carla_map=carla_map,
            spawn_index=args.spawn_index + episode_index,
            desired_gap_m=initial_condition.h0,
            ego_model=args.ego_model,
            lead_model=args.lead_model,
            spawn_z_offset=args.spawn_z_offset,
            candidate_spawn_indices=candidate_spawn_indices,
        )

        tm_port = traffic_manager.get_port()
        ego.set_autopilot(False, tm_port)
        lead.set_autopilot(False, tm_port)

        if args.enable_vehicle_lights:
            light_bits = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam
            if args.enable_fog_lights:
                light_bits |= carla.VehicleLightState.Fog
            if args.enable_interior_light:
                light_bits |= carla.VehicleLightState.Interior
            lights = carla.VehicleLightState(light_bits)
            ego.set_light_state(lights)
            lead.set_light_state(lights)

        collision_bp = world.get_blueprint_library().find("sensor.other.collision")
        collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=ego)
        collision_sensor.listen(monitor.callback)

        if args.native_recorder:
            recorder_log = episode_dir / "carla_recording.log"
            client.start_recorder(str(recorder_log), True)
            recorder_log_path = str(recorder_log)

        if args.video:
            recorder = FrameRecorder(
                world=world,
                ego_vehicle=ego,
                output_dir=episode_dir,
                width=args.video_width,
                height=args.video_height,
                prefix=f"episode_{episode_index:04d}",
                overlay_enabled=args.video_hud,
                record_immediately=args.include_warmstart_in_video,
            )

        def push_hud(extra: Dict[str, Any], frame_id: Optional[int] = None) -> None:
            if recorder is None:
                return
            payload: Dict[str, Any] = {
                "phase": "run",
                "time_s": 0.0,
            }
            payload.update(extra)
            recorder.update_overlay(payload, frame_id=frame_id)

        world.tick()

        max_warmup_ticks = args.max_warmup_ticks
        if args.warmstart_mode == "hybrid_short":
            max_warmup_ticks = min(max_warmup_ticks, args.hybrid_warmup_ticks)

        if args.warmstart_mode == "strict_only":
            h0_achieved = bumper_gap_m(lead, ego)
            v_lead_achieved = max(0.0, forward_speed_mps(lead))
            v_ego_achieved = max(0.0, forward_speed_mps(ego))
            warm_ok = (
                abs(h0_achieved - initial_condition.h0) <= args.gap_tolerance
                and abs(v_lead_achieved - initial_condition.v_lead0) <= args.speed_tolerance
                and abs(v_ego_achieved - initial_condition.v_ego0) <= args.speed_tolerance
            )
            if not warm_ok:
                print(
                    f"[episode {episode_index}] strict-only warmstart: skipping iterative warm-up and applying strict snap "
                    f"from h={h0_achieved:.3f}, vL={v_lead_achieved:.3f}, vE={v_ego_achieved:.3f}"
                )
        else:
            h0_achieved, v_lead_achieved, v_ego_achieved, warm_ok = warm_start_to_initial_condition(
                world=world,
                carla_map=carla_map,
                ego=ego,
                lead=lead,
                ego_wp=ego_wp,
                desired_h0=initial_condition.h0,
                desired_v_ego=initial_condition.v_ego0,
                desired_v_lead=initial_condition.v_lead0,
                dt=args.dt,
                max_warmup_ticks=max_warmup_ticks,
                gap_tolerance_m=args.gap_tolerance,
                speed_tolerance_mps=args.speed_tolerance,
                settle_ticks_required=args.settle_ticks,
                gap_feedback_gain=args.warm_gap_feedback_gain,
                on_step=push_hud if recorder is not None else None,
            )

        if not warm_ok:
            if args.strict_initial_snap:
                print(
                    f"[episode {episode_index}] warm-start pre-snap mismatch: "
                    f"h={h0_achieved:.3f} (target {initial_condition.h0:.3f}), "
                    f"vL={v_lead_achieved:.3f} (target {initial_condition.v_lead0:.3f}), "
                    f"vE={v_ego_achieved:.3f} (target {initial_condition.v_ego0:.3f}). "
                    "Applying strict initialization snap..."
                )
            else:
                print(
                    f"[episode {episode_index}] warm-start tolerance not fully met: "
                    f"h={h0_achieved:.3f} (target {initial_condition.h0:.3f}), "
                    f"vL={v_lead_achieved:.3f} (target {initial_condition.v_lead0:.3f}), "
                    f"vE={v_ego_achieved:.3f} (target {initial_condition.v_ego0:.3f})"
                )

        strict_constvel_active = False
        if args.strict_initial_snap:
            strict_ok = False
            strict_attempts = max(1, args.strict_retry_attempts)
            for strict_try in range(1, strict_attempts + 1):
                h0_achieved, v_lead_achieved, v_ego_achieved, strict_constvel_active = enforce_strict_initial_condition(
                    world=world,
                    carla_map=carla_map,
                    ego=ego,
                    lead=lead,
                    reference_ego_wp=ego_wp,
                    desired_h0=initial_condition.h0,
                    desired_v_ego=initial_condition.v_ego0,
                    desired_v_lead=initial_condition.v_lead0,
                    dt=args.dt,
                    settle_ticks=args.strict_snap_settle_ticks,
                    gap_tolerance_m=args.gap_tolerance,
                    speed_tolerance_mps=args.speed_tolerance,
                    max_iterations=args.strict_max_iterations,
                    ground_z_offset=args.strict_ground_z_offset,
                    on_step=push_hud if recorder is not None else None,
                )
                strict_ok = (
                    abs(h0_achieved - initial_condition.h0) <= args.gap_tolerance
                    and abs(v_lead_achieved - initial_condition.v_lead0) <= args.speed_tolerance
                    and abs(v_ego_achieved - initial_condition.v_ego0) <= args.speed_tolerance
                )
                if strict_ok:
                    break
                if strict_try < strict_attempts:
                    print(
                        f"[episode {episode_index}] strict snap attempt {strict_try}/{strict_attempts} missed tolerance; "
                        "retrying..."
                    )

            warm_ok = strict_ok
            if not strict_ok:
                msg = (
                    f"[episode {episode_index}] strict snap tolerance not met: "
                    f"h={h0_achieved:.3f} (target {initial_condition.h0:.3f}), "
                    f"vL={v_lead_achieved:.3f} (target {initial_condition.v_lead0:.3f}), "
                    f"vE={v_ego_achieved:.3f} (target {initial_condition.v_ego0:.3f})"
                )
                if args.fail_on_init_mismatch:
                    raise RuntimeError(msg + " | Aborting due to --fail-on-init-mismatch")
                print(msg)

        if args.video and recorder is not None and not args.include_warmstart_in_video:
            frame_before_rollout = world.get_snapshot().frame
            recorder.start_capture(
                reset_counter=True,
                capture_after_frame=frame_before_rollout,
            )

        if recorder is not None:
            push_hud(
                {
                    "phase": "run-init",
                    "time_s": 0.0,
                    "gap_m": h0_achieved,
                    "lead_speed_mps": v_lead_achieved,
                    "ego_speed_mps": v_ego_achieved,
                }
            )

        ego_init_tf = ego.get_transform()
        lead_init_tf = lead.get_transform()
        ego_init_wp = carla_map.get_waypoint(
            ego_init_tf.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        lead_init_wp = carla_map.get_waypoint(
            lead_init_tf.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        ego_init_wp_z = float(ego_init_wp.transform.location.z) if ego_init_wp is not None else None
        lead_init_wp_z = float(lead_init_wp.transform.location.z) if lead_init_wp is not None else None
        ego_init_z = float(ego_init_tf.location.z)
        lead_init_z = float(lead_init_tf.location.z)
        ego_init_z_offset = (ego_init_z - ego_init_wp_z) if ego_init_wp_z is not None else None
        lead_init_z_offset = (lead_init_z - lead_init_wp_z) if lead_init_wp_z is not None else None

        n_steps = int(round(args.t_max / args.dt))

        lead_pid = LongitudinalPID()
        ego_pid = LongitudinalPID()

        # Reference speed states used by "toy_drag_brake" profile.
        v_lead_ref = initial_condition.v_lead0
        v_ego_ref = initial_condition.v_ego0

        min_gap = float("inf")
        failure = False

        current_gap = bumper_gap_m(lead, ego)
        lead_speed = max(0.0, forward_speed_mps(lead))
        ego_speed = max(0.0, forward_speed_mps(ego))
        min_gap = min(min_gap, current_gap)

        with trajectory_csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "step",
                    "time_s",
                    "gap_m",
                    "lead_speed_mps",
                    "ego_speed_mps",
                    "lead_target_speed_mps",
                    "ego_target_speed_mps",
                    "lead_throttle",
                    "lead_brake",
                    "ego_throttle",
                    "ego_brake",
                    "collision",
                ],
            )
            writer.writeheader()

            for k in range(n_steps):
                t = k * args.dt

                if args.scenario_profile == "toy_drag_brake":
                    # Keep the same structure as IS_PAC example_carfollowing dynamics,
                    # but track these target speeds using CARLA's realistic vehicle model.
                    accel_eff_l = args.accel + rng.uniform(-args.disturbance_uniform, args.disturbance_uniform)
                    drag_eff_l = args.drag_b + rng.uniform(-args.disturbance_uniform, args.disturbance_uniform)
                    accel_eff_e = args.accel + rng.uniform(-args.disturbance_uniform, args.disturbance_uniform)
                    drag_eff_e = args.drag_b + rng.uniform(-args.disturbance_uniform, args.disturbance_uniform)

                    noise_l = rng.uniform(0.0, args.speed_noise_uniform)
                    noise_e = rng.uniform(0.0, args.speed_noise_uniform)

                    if v_lead_ref > 0.0:
                        v_lead_ref = max(0.0, v_lead_ref + (-accel_eff_l - drag_eff_l * v_lead_ref * v_lead_ref) * args.dt + noise_l)
                    if v_ego_ref > 0.0:
                        v_ego_ref = max(0.0, v_ego_ref + (-accel_eff_e - drag_eff_e * v_ego_ref * v_ego_ref) * args.dt + noise_e)

                    lead_target_speed = v_lead_ref
                    ego_target_speed = v_ego_ref
                else:
                    # Lead brakes after threshold time; ego keeps nominal speed.
                    lead_target_speed = initial_condition.v_lead0
                    if t >= args.lead_brake_start_s:
                        lead_target_speed = max(0.0, lead_target_speed - args.lead_brake_decel_mps2 * (t - args.lead_brake_start_s))
                    ego_target_speed = initial_condition.v_ego0

                ego_target_speed_with_gap = max(
                    0.0,
                    ego_target_speed + args.run_gap_feedback_gain * (current_gap - initial_condition.h0),
                )

                next_frame_id = world.get_snapshot().frame + 1
                push_hud(
                    {
                        "phase": "run",
                        "time_s": t,
                        "gap_m": current_gap,
                        "lead_speed_mps": lead_speed,
                        "ego_speed_mps": ego_speed,
                    },
                    frame_id=next_frame_id,
                )

                lead_throttle, lead_brake = lead_pid.step(lead_target_speed, lead_speed, args.dt)
                ego_throttle, ego_brake = ego_pid.step(ego_target_speed_with_gap, ego_speed, args.dt)

                is_collision = monitor.collided
                if current_gap <= 0.0 or is_collision:
                    failure = True

                writer.writerow(
                    {
                        "step": k,
                        "time_s": round(t, 6),
                        "gap_m": round(current_gap, 6),
                        "lead_speed_mps": round(lead_speed, 6),
                        "ego_speed_mps": round(ego_speed, 6),
                        "lead_target_speed_mps": round(lead_target_speed, 6),
                        "ego_target_speed_mps": round(ego_target_speed_with_gap, 6),
                        "lead_throttle": round(lead_throttle, 6),
                        "lead_brake": round(lead_brake, 6),
                        "ego_throttle": round(ego_throttle, 6),
                        "ego_brake": round(ego_brake, 6),
                        "collision": int(is_collision),
                    }
                )

                if failure and args.stop_on_failure:
                    n_steps = k + 1
                    break

                if strict_constvel_active:
                    try:
                        ego.disable_constant_velocity()
                    except Exception:
                        pass
                    try:
                        lead.disable_constant_velocity()
                    except Exception:
                        pass
                    strict_constvel_active = False

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

                current_gap = bumper_gap_m(lead, ego)
                lead_speed = max(0.0, forward_speed_mps(lead))
                ego_speed = max(0.0, forward_speed_mps(ego))

                min_gap = min(min_gap, current_gap)

        if recorder is not None:
            # Wait briefly to ensure the last frame callback is flushed.
            time.sleep(0.2)
            output_fps = args.video_fps
            if output_fps <= 0:
                output_fps = max(1, int(round(1.0 / args.dt)))
            video_path = recorder.make_video(
                fps=output_fps,
                keep_frames=args.keep_frames,
                crf=args.video_crf,
                preset=args.video_preset,
            )

        failure_indicator = 0.0 if failure else 1.0

        summary = {
            "episode_index": episode_index,
            "h0_request": initial_condition.h0,
            "v_lead0_request": initial_condition.v_lead0,
            "v_ego0_request": initial_condition.v_ego0,
            "h0_achieved": h0_achieved,
            "v_lead0_achieved": v_lead_achieved,
            "v_ego0_achieved": v_ego_achieved,
            "warm_start_tolerance_met": warm_ok,
            "failure_indicator": failure_indicator,
            "collision": monitor.collided,
            "min_gap_m": min_gap,
            "n_steps_executed": n_steps,
            "ego_init_z_m": ego_init_z,
            "lead_init_z_m": lead_init_z,
            "ego_init_wp_z_m": ego_init_wp_z,
            "lead_init_wp_z_m": lead_init_wp_z,
            "ego_init_z_offset_from_wp_m": ego_init_z_offset,
            "lead_init_z_offset_from_wp_m": lead_init_z_offset,
            "scenario_profile": args.scenario_profile,
            "trajectory_csv": str(trajectory_csv_path),
            "video_mp4": video_path,
            "recorder_log": recorder_log_path,
        }
        with summary_json_path.open("w") as fh:
            json.dump(summary, fh, indent=2)

        return EpisodeResult(
            episode_index=episode_index,
            h0_request=initial_condition.h0,
            v_lead0_request=initial_condition.v_lead0,
            v_ego0_request=initial_condition.v_ego0,
            h0_achieved=h0_achieved,
            v_lead0_achieved=v_lead_achieved,
            v_ego0_achieved=v_ego_achieved,
            failure_indicator=failure_indicator,
            collision=monitor.collided,
            min_gap_m=min_gap,
            n_steps_executed=n_steps,
            trajectory_csv=str(trajectory_csv_path),
            summary_json=str(summary_json_path),
            video_mp4=video_path,
            recorder_log=recorder_log_path,
        )

    finally:
        if recorder is not None:
            recorder.destroy()
        destroy_actor_safe(collision_sensor)
        destroy_actor_safe(lead)
        destroy_actor_safe(ego)

        if args.native_recorder:
            try:
                client.stop_recorder()
            except Exception:
                pass

        # Flush destruction commands in sync mode.
        try:
            world.tick()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone deterministic CARLA car-following runner with warm-start to exact initial conditions."
    )

    # CARLA connectivity
    parser.add_argument("--host", default="localhost", help="CARLA host (default: localhost)")
    parser.add_argument("--port", type=int, default=2010, help="CARLA RPC port (default: 2010)")
    parser.add_argument("--timeout", type=float, default=30.0, help="CARLA client timeout seconds")
    parser.add_argument("--town", default="Town10HD_Opt", help="Town to load (default: Town10HD_Opt)")
    parser.add_argument("--start-carla-server", action="store_true", help="Start/stop CarlaUE4.sh from this script")
    parser.add_argument("--carla-root", default=os.environ.get("CARLA_ROOT", "/home/workspace/carla_0.9.16"))
    parser.add_argument("--carla-start-timeout", type=float, default=180.0, help="Seconds to wait for CARLA RPC after start")
    parser.add_argument("--start-with-xvfb", dest="start_with_xvfb", action="store_true", help="Wrap CarlaUE4.sh in xvfb-run")
    parser.add_argument("--no-xvfb", dest="start_with_xvfb", action="store_false", help="Do not wrap CarlaUE4.sh in xvfb-run")
    parser.add_argument("--xvfb-screen-depth", type=int, default=24)
    parser.add_argument("--carla-quality-level", default="Epic", choices=["Low", "Epic"])
    parser.add_argument("--carla-res-x", type=int, default=1920)
    parser.add_argument("--carla-res-y", type=int, default=1080)
    parser.add_argument("--carla-prefernvidia", dest="carla_prefernvidia", action="store_true")
    parser.add_argument("--no-carla-prefernvidia", dest="carla_prefernvidia", action="store_false")

    # Determinism and stepping
    parser.add_argument("--dt", type=float, default=DT_DEFAULT, help="Fixed delta time seconds")
    parser.add_argument("--t-max", type=float, default=T_MAX_DEFAULT, help="Simulation horizon seconds")
    parser.add_argument("--seed", type=int, default=456, help="Random seed")
    parser.add_argument("--no-rendering", action="store_true", help="Enable CARLA no_rendering_mode (faster, no video)")
    parser.add_argument("--gpu-warmup", action="store_true", help="Run shader/GPU warmup after connecting")
    parser.add_argument(
        "--weather-profile",
        default="professional_wet_golden_hour",
        choices=[
            "custom",
            "professional_wet_golden_hour",
            "wet_sunset",
            "wet_cloudy_sunset",
            "mid_rain_sunset",
            "clear_sunset",
            "clear_noon",
        ],
        help="Scene weather/lighting profile",
    )
    parser.add_argument("--sun-altitude-angle", type=float, default=None)
    parser.add_argument("--sun-azimuth-angle", type=float, default=None)
    parser.add_argument("--wetness", type=float, default=None)

    # Initial conditions and sampling
    parser.add_argument("--sample-mode", choices=["fixed", "is_pac"], default="fixed")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--h0", type=float, default=20.0, help="Initial bumper gap in meters (fixed mode)")
    parser.add_argument("--v-lead0", type=float, default=6.0, help="Initial lead speed in m/s (fixed mode)")
    parser.add_argument("--v-ego0", type=float, default=VF_INIT_DEFAULT, help="Initial ego speed in m/s")

    # IS_PAC distribution params
    parser.add_argument("--mu-v-mean", type=float, default=MU_V_MEAN_DEFAULT)
    parser.add_argument("--mu-v-std", type=float, default=MU_V_STD_DEFAULT)
    parser.add_argument("--d-stop-median", type=float, default=D_STOP_MEDIAN_DEFAULT)
    parser.add_argument("--tau-median", type=float, default=TAU_MEDIAN_DEFAULT)
    parser.add_argument("--sigma-gap", type=float, default=SIGMA_GAP_DEFAULT)

    # Warm-start options
    parser.add_argument("--spawn-index", type=int, default=0, help="Base spawn index for episode 0")
    parser.add_argument("--spawn-z-offset", type=float, default=0.02, help="Spawn Z offset above waypoint road surface [m]")
    parser.add_argument("--max-warmup-ticks", type=int, default=400)
    parser.add_argument("--gap-tolerance", type=float, default=0.35, help="Warm-start acceptable |gap error| [m]")
    parser.add_argument("--speed-tolerance", type=float, default=0.25, help="Warm-start acceptable |speed error| [m/s]")
    parser.add_argument("--settle-ticks", type=int, default=8, help="Consecutive in-tolerance ticks required")
    parser.add_argument(
        "--warmstart-mode",
        choices=["iterative_then_strict", "hybrid_short", "strict_only"],
        default="iterative_then_strict",
        help="Warm-start strategy before strict snap",
    )
    parser.add_argument(
        "--hybrid-warmup-ticks",
        type=int,
        default=40,
        help="Max iterative warmup ticks when --warmstart-mode hybrid_short",
    )
    parser.add_argument("--warm-gap-feedback-gain", type=float, default=0.25, help="Warm-start ego speed correction from gap error")
    parser.add_argument("--strict-initial-snap", dest="strict_initial_snap", action="store_true", help="Apply strict final initialization snap (default)")
    parser.add_argument("--no-strict-initial-snap", dest="strict_initial_snap", action="store_false")
    parser.add_argument("--strict-snap-settle-ticks", type=int, default=4)
    parser.add_argument("--strict-max-iterations", type=int, default=36, help="Max strict-init correction ticks")
    parser.add_argument("--strict-retry-attempts", type=int, default=3, help="Number of strict-snap retries before declaring init mismatch")
    parser.add_argument("--strict-ground-z-offset", type=float, default=0.02, help="Strict snap Z offset above waypoint road surface [m]")
    parser.add_argument("--fail-on-init-mismatch", dest="fail_on_init_mismatch", action="store_true", help="Abort run if strict init cannot meet tolerances (default)")
    parser.add_argument("--allow-init-mismatch", dest="fail_on_init_mismatch", action="store_false", help="Continue run even if strict init mismatch remains")
    parser.add_argument("--straight-road-only", dest="straight_road_only", action="store_true", help="Use only approximately straight spawn lanes (default)")
    parser.add_argument("--allow-curved-spawn", dest="straight_road_only", action="store_false")
    parser.add_argument("--straight-lookahead-m", type=float, default=90.0)
    parser.add_argument("--straight-sample-step-m", type=float, default=2.0)
    parser.add_argument("--straight-max-yaw-delta-deg", type=float, default=6.0)
    parser.add_argument("--straight-min-length-m", type=float, default=90.0, help="Required straight distance from spawn point")
    parser.add_argument("--straight-scan-max-m", type=float, default=220.0, help="Maximum forward scan distance for straight-road scoring")
    parser.add_argument("--straight-require-no-junction", dest="straight_require_no_junction", action="store_true")
    parser.add_argument("--straight-allow-junction", dest="straight_require_no_junction", action="store_false")
    parser.add_argument("--straight-sort-by-length", dest="straight_sort_by_length", action="store_true")
    parser.add_argument("--straight-keep-map-order", dest="straight_sort_by_length", action="store_false")

    # Scenario options
    parser.add_argument(
        "--scenario-profile",
        choices=["toy_drag_brake", "lead_brake_constant_ego"],
        default="toy_drag_brake",
        help="Run profile after warm-start",
    )
    parser.add_argument("--accel", type=float, default=ACCEL_DEFAULT, help="Toy profile accel parameter")
    parser.add_argument("--drag-b", type=float, default=DRAG_B_DEFAULT, help="Toy profile drag parameter")
    parser.add_argument("--disturbance-uniform", type=float, default=0.25, help="Toy profile disturbance half-width")
    parser.add_argument("--speed-noise-uniform", type=float, default=0.2, help="Toy profile additive speed noise upper bound")
    parser.add_argument("--run-gap-feedback-gain", type=float, default=0.0, help="Optional runtime ego speed correction from gap error")
    parser.add_argument("--lead-brake-start-s", type=float, default=2.0, help="Alternative profile brake start")
    parser.add_argument("--lead-brake-decel-mps2", type=float, default=3.5, help="Alternative profile lead braking magnitude")
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop episode as soon as failure is detected")

    # Models
    parser.add_argument("--ego-model", default="vehicle.lincoln.mkz_2020")
    parser.add_argument("--lead-model", default="vehicle.lincoln.mkz_2020")
    parser.add_argument("--enable-vehicle-lights", dest="enable_vehicle_lights", action="store_true")
    parser.add_argument("--no-vehicle-lights", dest="enable_vehicle_lights", action="store_false")
    parser.add_argument("--enable-fog-lights", dest="enable_fog_lights", action="store_true")
    parser.add_argument("--no-fog-lights", dest="enable_fog_lights", action="store_false")
    parser.add_argument("--enable-interior-light", dest="enable_interior_light", action="store_true")
    parser.add_argument("--no-interior-light", dest="enable_interior_light", action="store_false")

    # Recording
    parser.add_argument("--native-recorder", action="store_true", help="Save CARLA native recorder log per episode")
    parser.add_argument("--video", dest="video", action="store_true", help="Capture ego camera video per episode")
    parser.add_argument("--no-video", dest="video", action="store_false", help="Disable ego camera video capture")
    parser.add_argument(
        "--include-warmstart-in-video",
        dest="include_warmstart_in_video",
        action="store_true",
        help="Include warm-start phase in recorded video",
    )
    parser.add_argument(
        "--exclude-warmstart-in-video",
        dest="include_warmstart_in_video",
        action="store_false",
        help="Start video after warm-start phase (default)",
    )
    parser.add_argument("--video-hud", dest="video_hud", action="store_true", help="Render HUD dashboard on video")
    parser.add_argument("--no-video-hud", dest="video_hud", action="store_false", help="Disable HUD dashboard on video")
    parser.add_argument("--video-fps", type=int, default=0, help="Output FPS. Use 0 to auto-match simulation dt")
    parser.add_argument("--video-width", type=int, default=1920)
    parser.add_argument("--video-height", type=int, default=1080)
    parser.add_argument("--video-crf", type=int, default=18, help="Lower is higher quality (typical range 16-28)")
    parser.add_argument("--video-preset", default="slow", help="ffmpeg x264 preset, e.g. fast, medium, slow")
    parser.add_argument("--keep-frames", action="store_true", help="Do not delete JPG frames after MP4 encode")

    # Output
    parser.add_argument("--output-dir", default="", help="Output directory (default: auto timestamped)")

    parser.set_defaults(
        video=True,
        video_hud=True,
        include_warmstart_in_video=False,
        start_with_xvfb=True,
        carla_prefernvidia=True,
        strict_initial_snap=True,
        fail_on_init_mismatch=True,
        straight_road_only=True,
        straight_require_no_junction=True,
        straight_sort_by_length=True,
        enable_vehicle_lights=True,
        enable_fog_lights=False,
        enable_interior_light=False,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.video and args.no_rendering:
        raise SystemExit("--video is incompatible with --no-rendering")

    if args.warmstart_mode == "strict_only" and not args.strict_initial_snap:
        raise SystemExit("--warmstart-mode strict_only requires --strict-initial-snap")

    rng = random.Random(args.seed)
    random.seed(args.seed)

    if args.output_dir:
        out_root = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = Path("runs") / f"carla_is_pac_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("CARLA IS_PAC Warm-Start Car-Following Runner")
    print("=" * 72)
    print(f"Output dir: {out_root.resolve()}")
    print(f"Host/port: {args.host}:{args.port}")
    print(f"Samples: {args.num_samples} ({args.sample_mode})")

    carla_proc: Optional[subprocess.Popen] = None
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

    # Deterministic weather and traffic-light behavior.
    weather = build_weather(args)
    world.set_weather(weather)
    print(
        "Weather:",
        args.weather_profile,
        f"(sun_alt={weather.sun_altitude_angle:.1f}, sun_az={weather.sun_azimuth_angle:.1f}, wetness={weather.wetness:.1f})",
    )
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
        straight, straight_scored = find_straight_spawn_indices(
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
            top_preview = ", ".join([f"{idx}:{dist:.0f}m" for idx, dist in straight_scored[:5]])
            print(
                f"Using straight-lane spawn subset: {len(straight)} / {len(all_spawn_points)} "
                f"(min={args.straight_min_length_m:.0f}m, no_junction={int(args.straight_require_no_junction)})"
            )
            if top_preview:
                print(f"Top straight candidates (spawn_idx:length): {top_preview}")
        else:
            print(
                "WARNING: No straight-lane spawn points detected with current thresholds; "
                "using all spawn points."
            )

    initial_conditions = prepare_initial_conditions(args, rng)

    run_manifest = {
        "args": vars(args),
        "seed": args.seed,
        "output_dir": str(out_root.resolve()),
        "carla_server_log": str(carla_log_path) if carla_log_path else None,
        "episodes": [],
    }

    results: List[EpisodeResult] = []

    try:
        for i, ic in enumerate(initial_conditions):
            print(
                f"[episode {i}] request: h0={ic.h0:.3f} m, "
                f"v_lead0={ic.v_lead0:.3f} m/s, v_ego0={ic.v_ego0:.3f} m/s"
            )
            res = run_episode(
                args=args,
                episode_index=i,
                initial_condition=ic,
                rng=rng,
                client=client,
                world=world,
                traffic_manager=traffic_manager,
                output_dir=out_root,
                candidate_spawn_indices=candidate_spawn_indices,
            )
            results.append(res)
            run_manifest["episodes"].append(
                {
                    "episode_index": res.episode_index,
                    "failure_indicator": res.failure_indicator,
                    "collision": res.collision,
                    "min_gap_m": res.min_gap_m,
                    "h0_request": res.h0_request,
                    "h0_achieved": res.h0_achieved,
                    "v_lead0_request": res.v_lead0_request,
                    "v_lead0_achieved": res.v_lead0_achieved,
                    "v_ego0_request": res.v_ego0_request,
                    "v_ego0_achieved": res.v_ego0_achieved,
                    "summary_json": res.summary_json,
                    "trajectory_csv": res.trajectory_csv,
                    "video_mp4": res.video_mp4,
                    "recorder_log": res.recorder_log,
                }
            )
            print(
                f"[episode {i}] done: indicator={res.failure_indicator:.0f}, "
                f"collision={int(res.collision)}, min_gap={res.min_gap_m:.3f} m"
            )

    finally:
        # Reset world settings.
        try:
            traffic_manager.set_synchronous_mode(False)
        except Exception:
            pass
        try:
            set_async_mode(world)
        except Exception:
            pass
        stop_carla_server(carla_proc)

    manifest_path = out_root / "run_manifest.json"
    with manifest_path.open("w") as fh:
        json.dump(run_manifest, fh, indent=2)

    # Flat summary CSV for quick PAC post-processing.
    summary_csv = out_root / "episodes_summary.csv"
    with summary_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "episode_index",
                "failure_indicator",
                "collision",
                "min_gap_m",
                "h0_request",
                "h0_achieved",
                "v_lead0_request",
                "v_lead0_achieved",
                "v_ego0_request",
                "v_ego0_achieved",
                "n_steps_executed",
                "trajectory_csv",
                "summary_json",
                "video_mp4",
                "recorder_log",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "episode_index": r.episode_index,
                    "failure_indicator": r.failure_indicator,
                    "collision": int(r.collision),
                    "min_gap_m": r.min_gap_m,
                    "h0_request": r.h0_request,
                    "h0_achieved": r.h0_achieved,
                    "v_lead0_request": r.v_lead0_request,
                    "v_lead0_achieved": r.v_lead0_achieved,
                    "v_ego0_request": r.v_ego0_request,
                    "v_ego0_achieved": r.v_ego0_achieved,
                    "n_steps_executed": r.n_steps_executed,
                    "trajectory_csv": r.trajectory_csv,
                    "summary_json": r.summary_json,
                    "video_mp4": r.video_mp4 or "",
                    "recorder_log": r.recorder_log or "",
                }
            )

    safe_count = sum(1 for r in results if r.failure_indicator > 0.5)
    fail_count = len(results) - safe_count

    print("=" * 72)
    print(f"Completed episodes: {len(results)}")
    print(f"Safe: {safe_count} | Failure: {fail_count}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()