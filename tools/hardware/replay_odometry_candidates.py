#!/usr/bin/env python3
"""Reprocesa bags de corridas con cada candidato de odometría de RTAB-Map y los compara contra el tablero.

Fase 3 del plan (``docs/plans/plan_d435i_herramienta_demostraciones.md``): todos los candidatos sobre los MISMOS
datos, así la diferencia es solo del estimador. Por cada bag y candidato (``vio_measurement.yaml``, ``candidates``):
1. ``ros2 bag play`` (reloj simulado, más lento que el tiempo real para no perder frames) de color, profundidad,
   ``camera_info``, IMU y ``/tf_static``. ``/tf_static`` se reproduce con QoS TRANSIENT_LOCAL.
2. ``image_transport republish`` descomprime el color (JPEG) y la profundidad (compressedDepth, PNG) si vienen
   comprimidos.
3. ``imu_filter_madgwick`` y ``rgbd_odometry``: parámetros base (``rtabmap_vio_params.yaml``) más los del candidato,
   sin publicar TF.
4. ``ros2 bag record`` de ``/replay/odom`` y ``/replay/odom_info``.
Todo corre en un dominio DDS aislado (``replay.domain_id``) para no interferir con la cámara ni con el robot. Al
final llama a ``evaluate_pose_estimates.py`` con la odometría en vivo y la de cada candidato.

Uso (contenedor con ROS 2 y RTAB-Map; los bags no se modifican):
  python3 tools/hardware/replay_odometry_candidates.py --run datasets/vio_bags/run_01 --run datasets/vio_bags/run_02 \\
      --candidates baseline,bundle_adjustment,guess_motion --out-dir experiments/vio_replay/2026-11-25
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "hardware"))

from vio_bag_utils import bag_topics  # noqa: E402

DEFAULT_CONFIG = REPO / "visuomotor_bringup" / "config" / "realsense" / "vio_measurement.yaml"
QOS_OVERRIDES = {"/tf_static": {"history": "keep_last", "depth": 1, "reliability": "reliable",
                                "durability": "transient_local"}}


def merged_params(base_yaml: Path, overrides: Dict) -> Dict:
    """Parámetros de ``rgbd_odometry``: base + candidato, con reloj simulado y sin publicar TF."""
    base = yaml.safe_load(base_yaml.read_text())
    params = dict(next(iter(base.values()))["ros__parameters"])
    params.update({k: v for k, v in overrides.items()})
    params["use_sim_time"] = True
    params["publish_tf"] = False
    return {"/**": {"ros__parameters": params}}


class Procs:
    """Procesos de ROS 2 en su propio grupo, detenidos con SIGINT (para que el bag se cierre bien)."""

    def __init__(self, env: Dict[str, str], log_dir: Path):
        self.env, self.log_dir, self.items = env, log_dir, []

    def start(self, name: str, cmd: List[str]) -> subprocess.Popen:
        log = open(self.log_dir / f"{name}.log", "w")
        p = subprocess.Popen(cmd, env=self.env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.items.append((name, p, log))
        return p

    def stop(self, name: str, timeout: float = 10.0) -> None:
        for n, p, log in self.items:
            if n == name and p.poll() is None:
                os.killpg(p.pid, signal.SIGINT)
                try:
                    p.wait(timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                log.close()

    def stop_all(self) -> None:
        for n, _, _ in list(self.items)[::-1]:
            self.stop(n)


def run_candidate(bag: Path, cand: str, overrides: Dict, cfg: Dict, out_dir: Path) -> Path:
    """Reprocesa ``bag`` con un candidato; devuelve la ruta del bag de salida."""
    rp = cfg["replay"]
    topics = bag_topics(bag)
    color = next((t for t in cfg["color_topics"] if t in topics), None)
    depth = next((t for t in cfg["depth_topics"] if t in topics), None)
    info, imu = cfg["camera_info_topic"], cfg["imu_topic"]
    if color is None or depth is None or info not in topics:
        raise SystemExit(f"{bag}: faltan tópicos (color {color}, profundidad {depth}, {info})")
    use_imu = bool(overrides.get("wait_imu_to_init", True)) and imu in topics
    out_dir.mkdir(parents=True, exist_ok=True)
    out_bag = out_dir / "odom"
    if out_bag.exists():
        raise SystemExit(f"{out_bag} ya existe: usar otro --out-dir (no se sobrescriben resultados)")
    env = dict(os.environ, ROS_DOMAIN_ID=str(rp["domain_id"]), ROS_LOCALHOST_ONLY="1",
               ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST", RMW_IMPLEMENTATION="rmw_cyclonedds_cpp")
    procs = Procs(env, out_dir)
    params = out_dir / "rgbd_odometry_params.yaml"
    params.write_text(yaml.safe_dump(merged_params(REPO / rp["base_params"], overrides), sort_keys=False))
    qos = out_dir / "qos_overrides.yaml"
    qos.write_text(yaml.safe_dump(QOS_OVERRIDES))
    sim = ["-p", "use_sim_time:=true"]
    rgb_in, depth_in = color, depth
    try:
        if color.endswith("/compressed"):
            rgb_in = "/replay/color/image_raw"
            procs.start("republish_color", ["ros2", "run", "image_transport", "republish", "--ros-args", "-p",
                                            "in_transport:=compressed", "-p", "out_transport:=raw", *sim,
                                            "-r", f"in/compressed:={color}", "-r", f"out:={rgb_in}"])
        if depth.endswith("/compressedDepth"):
            depth_in = "/replay/depth/image_raw"
            procs.start("republish_depth", ["ros2", "run", "image_transport", "republish", "--ros-args", "-p",
                                            "in_transport:=compressedDepth", "-p", "out_transport:=raw", *sim,
                                            "-r", f"in/compressedDepth:={depth}", "-r", f"out:={depth_in}"])
        remaps = ["-r", f"rgb/image:={rgb_in}", "-r", f"depth/image:={depth_in}", "-r", f"rgb/camera_info:={info}",
                  "-r", f"odom:={rp['output_topic']}", "-r", "odom_info:=/replay/odom_info"]
        if use_imu:
            procs.start("madgwick", ["ros2", "run", "imu_filter_madgwick", "imu_filter_madgwick_node", "--ros-args",
                                     "-p", "use_mag:=false", "-p", "world_frame:=enu", "-p", "publish_tf:=false",
                                     "-p", f"gain:={rp['madgwick_gain']}", *sim,
                                     "-r", f"imu/data_raw:={imu}", "-r", "imu/data:=/replay/imu/data"])
            remaps += ["-r", "imu:=/replay/imu/data"]
        procs.start("rgbd_odometry", ["ros2", "run", "rtabmap_odom", "rgbd_odometry", "--ros-args",
                                      "--params-file", str(params), *remaps])
        procs.start("record", ["ros2", "bag", "record", "-s", "mcap", "-o", str(out_bag), rp["output_topic"],
                               "/replay/odom_info"])
        time.sleep(float(rp["warmup_s"]))
        play_topics = [t for t in (color, depth, info, imu if use_imu else None, "/tf_static") if t and t in topics]
        play = procs.start("play", ["ros2", "bag", "play", str(bag), "--clock", "100", "--rate", str(rp["rate"]),
                                    "--disable-keyboard-controls", "--qos-profile-overrides-path", str(qos),
                                    "--topics", *play_topics])
        play.wait()
        time.sleep(2.0)                                      # que la odometría termine los últimos frames
        procs.stop("record")
    finally:
        procs.stop_all()
    return out_bag


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--candidates", default="", help="lista separada por comas (por defecto, todos los del YAML)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--allow-nominal", action="store_true", help="pasa al evaluador: tablero sin medir")
    ap.add_argument("--ground-truth-topic", default="", help="pasa al evaluador (bags sintéticos)")
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text())["vio_measurement"]
    cands = [c for c in a.candidates.split(",") if c] or list(cfg["candidates"])
    unknown = [c for c in cands if c not in cfg["candidates"]]
    if unknown:
        raise SystemExit(f"candidatos desconocidos: {unknown}")
    out_root = Path(a.out_dir)
    if not out_root.is_absolute():
        out_root = REPO / out_root
    for run in a.run:
        for cand in cands:
            t0 = time.time()
            out = run_candidate(Path(run), cand, cfg["candidates"][cand] or {}, cfg, out_root / Path(run).name / cand)
            print(f"[{Path(run).name}] {cand}: {time.time() - t0:.0f} s -> {out}")
    if a.no_eval:
        return 0
    from evaluate_pose_estimates import main as evaluate
    args = sum((["--run", r] for r in a.run), [])
    args += sum((["--estimate", f"{c}={out_root}/{{run}}/{c}/odom:{cfg['replay']['output_topic']}"] for c in cands), [])
    args += ["--out-dir", str(out_root / "evaluation"), "--config", a.config]
    if a.allow_nominal:
        args.append("--allow-nominal")
    if a.ground_truth_topic:
        args += ["--ground-truth-topic", a.ground_truth_topic]
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
