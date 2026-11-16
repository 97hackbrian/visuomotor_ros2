#!/usr/bin/env python3
# Copyright (c) 2026. Programacion Cientifica, DISC, Antofagasta, Chile.
"""
parse_collection_telemetry.py — Extractor y Analizador de Telemetría de Demostraciones Expertas
=============================================================================================
Parsea archivos de registro de telemetría (/tmp/pickup_approximate_*.log y logs de ROS 2)
generados durante la recolección autónoma de demostraciones para Diffusion Policy.
Extrae todas las métricas por episodio:
- Coordenadas estocásticas del objeto (Halton QMC: X, Y, Z, Yaw)
- Jitter y perturbación inicial de la pose del efector final (EE: dx, dy, dRoll, dPitch, dYaw)
- Tiempos de duración por fase de la FSM (WAIT, APPROACH, ALIGN, DESCEND, SETTLE, GRASP, LIFT, UNTILT, MOVE, HOLD)
- Errores residuales cinemáticos en cada transición (err_xy, err_z, err_rot, err_3d)
- Pasos temporales a 20 Hz, duración simulada, duración de pared y factor RTF
- Intentos y reintentos por fallos de agarre o timeouts

Permite exportar a CSV tabular y persistir de forma fail-safe en la base de datos
SQLite experiments/experiments.db.
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Importar utilidades de visuomotor_core
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
CORE_DIR = REPO_ROOT / "visuomotor_core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

try:
    from visuomotor_core.utils.tracker import record_dataset_episode
except ImportError:
    record_dataset_episode = None


def parse_timestamp(ts_float_str: Optional[str]) -> Tuple[str, str]:
    """Convierte timestamp flotante de ROS a ISO 8601 legible."""
    if not ts_float_str:
        now_dt = datetime.now()
        return str(now_dt.timestamp()), now_dt.isoformat()
    try:
        ts = float(ts_float_str)
        # Si es timestamp simulado muy bajo (ej. 10.5), retornar formato numérico
        if ts < 1e8:
            return ts_float_str, f"sim_time_{ts:.3f}s"
        dt = datetime.fromtimestamp(ts)
        return ts_float_str, dt.isoformat()
    except Exception:
        return ts_float_str, str(ts_float_str)


def parse_single_log(filepath: str) -> List[Dict[str, Any]]:
    """
    Parsea exhaustivamente un archivo de registro de pickup_approximate
    extrayendo todos los episodios completados y sus métricas intermedias.
    """
    if not os.path.isfile(filepath):
        return []

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    episodes = []
    log_name = os.path.basename(filepath)

    # Estado temporal del episodio en curso
    curr = {
        "timestamp_ros": None,
        "obj_x": None,
        "obj_y": None,
        "obj_z": 0.44,  # Altura estándar spawn_z
        "obj_yaw": None,
        "ee_dx": 0.0,
        "ee_dy": 0.0,
        "ee_roll": 0.0,
        "ee_pitch": 0.0,
        "ee_yaw": 0.0,
        "err_approach_xy": None,
        "err_approach_z": None,
        "err_align_rot": None,
        "err_descend_3d": None,
        "err_descend_rot": None,
        "err_move_3d": None,
        "err_move_rot": None,
        "retries": 0,
        "phase_timestamps": {},
    }

    for line in lines:
        # Timestamp de la línea
        ts_m = re.search(r"\[(\d+\.\d+)\]", line)
        line_ts = float(ts_m.group(1)) if ts_m else None
        if line_ts is not None:
            curr["timestamp_ros"] = str(line_ts)

        # 1. Spawn del Objeto (Halton)
        # Objeto re-posicionado (Halton): X=-0.357, Y=-0.859, Yaw=-2.56
        obj_m = re.search(
            r"Objeto re-posicionado \(Halton\):\s*X=([-\d.]+),\s*Y=([-\d.]+),\s*Yaw=([-\d.]+)",
            line,
        )
        if obj_m:
            curr["obj_x"] = float(obj_m.group(1))
            curr["obj_y"] = float(obj_m.group(2))
            curr["obj_yaw"] = float(obj_m.group(3))

        # 2. Inyección de Jitter EE (Robot)
        # Pose aleatoria EE inyectada: Traslación(dx=0.043m, dy=-0.032m) | Rotación(dRoll=-0.061, dPitch=0.054, dYaw=0.053 rad)
        ee_m = re.search(
            r"Traslación\(dx=([-\d.]+)m,\s*dy=([-\d.]+)m\)\s*\|\s*Rotación\(dRoll=([-\d.]+),\s*dPitch=([-\d.]+),\s*dYaw=([-\d.]+)\s*rad\)",
            line,
        )
        if ee_m:
            curr["ee_dx"] = float(ee_m.group(1))
            curr["ee_dy"] = float(ee_m.group(2))
            curr["ee_roll"] = float(ee_m.group(3))
            curr["ee_pitch"] = float(ee_m.group(4))
            curr["ee_yaw"] = float(ee_m.group(5))

        # 3. Transiciones de Fases y Errores Residuales
        if "WAIT terminado → APPROACH_XY" in line:
            if line_ts:
                curr["phase_timestamps"]["approach_start"] = line_ts

        app_m = re.search(
            r"APPROACH_XY completo \(err_xy=([-\d.]+)m,\s*err_z=([-\d.]+)m\)\s*→\s*ALIGN_ROTATION",
            line,
        )
        if app_m:
            curr["err_approach_xy"] = float(app_m.group(1))
            curr["err_approach_z"] = float(app_m.group(2))
            if line_ts:
                curr["phase_timestamps"]["align_start"] = line_ts

        align_m = re.search(
            r"ALIGN_ROTATION completo \(err_rot=([-\d.]+)rad\)\s*→\s*DESCEND",
            line,
        )
        if align_m:
            curr["err_align_rot"] = float(align_m.group(1))
            if line_ts:
                curr["phase_timestamps"]["descend_start"] = line_ts

        desc_m = re.search(
            r"DESCEND completado/forzado \(err_3d=([-\d.]+)m,\s*err_rot=([-\d.]+)rad\)\s*→\s*SETTLE",
            line,
        )
        if desc_m:
            curr["err_descend_3d"] = float(desc_m.group(1))
            curr["err_descend_rot"] = float(desc_m.group(2))
            if line_ts:
                curr["phase_timestamps"]["settle_start"] = line_ts

        if "SETTLE completo → GRASP" in line:
            if line_ts:
                curr["phase_timestamps"]["grasp_start"] = line_ts

        if "GRASP completo → LIFT" in line:
            if line_ts:
                curr["phase_timestamps"]["lift_start"] = line_ts

        if "LIFT completo → UNTILT" in line:
            if line_ts:
                curr["phase_timestamps"]["untilt_start"] = line_ts

        if "UNTILT completo y validado → MOVE_TO_TARGET" in line:
            if line_ts:
                curr["phase_timestamps"]["move_start"] = line_ts

        move_m = re.search(
            r"MOVE_TO_TARGET completo \(err_3d=([-\d.]+)m,\s*err_rot=([-\d.]+)rad\)\s*→\s*HOLD_TARGET",
            line,
        )
        if move_m:
            curr["err_move_3d"] = float(move_m.group(1))
            curr["err_move_rot"] = float(move_m.group(2))
            if line_ts:
                curr["phase_timestamps"]["hold_start"] = line_ts

        if "HOLD_TARGET completo" in line:
            if line_ts:
                curr["phase_timestamps"]["episode_end"] = line_ts

        # 4. Fallos / Reintentos
        if (
            "Fallo de agarre detectado" in line
            or "Timeout en WAIT" in line
            or "Timeout global superado" in line
            or "Secuencia insuficiente" in line
        ):
            curr["retries"] += 1

        # 5. Telemetría de Episodio Guardado
        # ⏱️ TELEMETRÍA EPISODIO 40: Pasos=230 | Duración Sim=11.50s | Grabación Wall=46.80s (0.25x) | Ciclo Total Wall=60.95s
        telem_m = re.search(
            r"TELEMETRÍA EPISODIO (\d+):\s*Pasos=(\d+)\s*\|\s*Duración Sim=([-\d.]+)s\s*\|\s*Grabación Wall=([-\d.]+)s\s*\(([-\d.]+)x\)\s*\|\s*Ciclo Total Wall=([-\d.]+)s",
            line,
        )
        if telem_m:
            ep_batch_idx = int(telem_m.group(1))
            steps = int(telem_m.group(2))
            sim_dur = float(telem_m.group(3))
            wall_rec = float(telem_m.group(4))
            rtf = float(telem_m.group(5))
            wall_cycle = float(telem_m.group(6))

            # Calcular duraciones de fase si los timestamps existen
            pts = curr["phase_timestamps"]
            phase_durations = {}

            def _diff(start_key, end_key):
                if start_key in pts and end_key in pts:
                    return round(max(0.0, pts[end_key] - pts[start_key]), 3)
                return None

            phase_durations["duration_approach_s"] = _diff("approach_start", "align_start")
            phase_durations["duration_align_s"] = _diff("align_start", "descend_start")
            phase_durations["duration_descend_s"] = _diff("descend_start", "settle_start")
            phase_durations["duration_settle_s"] = _diff("settle_start", "grasp_start")
            phase_durations["duration_grasp_s"] = _diff("grasp_start", "lift_start")
            phase_durations["duration_lift_s"] = _diff("lift_start", "untilt_start")
            phase_durations["duration_untilt_s"] = _diff("untilt_start", "move_start")
            phase_durations["duration_move_s"] = _diff("move_start", "hold_start")
            phase_durations["duration_hold_s"] = _diff("hold_start", "episode_end")

            ros_ts, iso_ts = parse_timestamp(curr["timestamp_ros"])

            ep_record = {
                "source_log": log_name,
                "episode_in_batch": ep_batch_idx,
                "timestamp_ros": ros_ts,
                "timestamp_iso": iso_ts,
                "obj_spawn_x": curr["obj_x"] if curr["obj_x"] is not None else 0.0,
                "obj_spawn_y": curr["obj_y"] if curr["obj_y"] is not None else 0.0,
                "obj_spawn_z": curr["obj_z"],
                "obj_spawn_yaw_rad": curr["obj_yaw"] if curr["obj_yaw"] is not None else 0.0,
                "ee_jitter_dx_m": curr["ee_dx"],
                "ee_jitter_dy_m": curr["ee_dy"],
                "ee_jitter_droll_rad": curr["ee_roll"],
                "ee_jitter_dpitch_rad": curr["ee_pitch"],
                "ee_jitter_dyaw_rad": curr["ee_yaw"],
                "err_approach_xy_m": curr["err_approach_xy"],
                "err_approach_z_m": curr["err_approach_z"],
                "err_align_rot_rad": curr["err_align_rot"],
                "err_descend_3d_m": curr["err_descend_3d"],
                "err_descend_rot_rad": curr["err_descend_rot"],
                "err_move_target_3d_m": curr["err_move_3d"],
                "err_move_target_rot_rad": curr["err_move_rot"],
                "steps": steps,
                "duration_sim_s": sim_dur,
                "recording_wall_s": wall_rec,
                "cycle_wall_s": wall_cycle,
                "rtf_speedup": rtf,
                "retries_before_success": curr["retries"],
                "status": "SUCCESS",
                **phase_durations,
            }
            episodes.append(ep_record)

            # Resetear estado para el siguiente episodio
            curr = {
                "timestamp_ros": None,
                "obj_x": None,
                "obj_y": None,
                "obj_z": 0.44,
                "obj_yaw": None,
                "ee_dx": 0.0,
                "ee_dy": 0.0,
                "ee_roll": 0.0,
                "ee_pitch": 0.0,
                "ee_yaw": 0.0,
                "err_approach_xy": None,
                "err_approach_z": None,
                "err_align_rot": None,
                "err_descend_3d": None,
                "err_descend_rot": None,
                "err_move_3d": None,
                "err_move_rot": None,
                "retries": 0,
                "phase_timestamps": {},
            }

    return episodes


def read_zarr_step_counts(zarr_path: str) -> List[int]:
    """Lee la secuencia exacta de longitudes temporales (pasos) de cada episode_i en un Zarr sin dependencias externas."""
    if not os.path.isdir(zarr_path):
        return []

    ep_dirs = []
    for p in glob.glob(os.path.join(zarr_path, "episode_*")):
        m = re.match(r".*episode_(\d+)$", p)
        if m:
            ep_dirs.append((int(m.group(1)), p))
    ep_dirs.sort(key=lambda x: x[0])

    steps = []
    for idx, p in ep_dirs:
        zarray_file = os.path.join(p, "action", ".zarray")
        if os.path.isfile(zarray_file):
            try:
                with open(zarray_file, "r", encoding="utf-8") as fp:
                    meta = json.load(fp)
                    steps.append(int(meta["shape"][0]))
            except Exception:
                pass
    return steps


def fallback_sort_key(filepath: str) -> Tuple[int, int, int]:
    """Ordenamiento determinista por sesión de ejecución, número de batch y timestamp."""
    fname = os.path.basename(filepath)
    parts = fname.replace(".log", "").split("_")
    ts = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else int(os.path.getmtime(filepath) * 1000)

    batch_num = 0
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as fp:
            head = "".join([fp.readline() for _ in range(15)])
        if "_batch_1" in head:
            batch_num = 1
        elif "_batch_2" in head:
            batch_num = 2
    except Exception:
        pass

    session_id = ts // 60000000000 if ts > 1e11 else (ts // 60000 if ts > 1e8 else ts // 60)
    return (session_id, batch_num, ts)


def parse_all_logs(
    log_files: List[str],
    align_zarr: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Parsea una lista de archivos de log. Si se proporciona align_zarr (o se detecta
    el dataset Zarr canónico), alinea matemáticamente los episodios contra la verdad
    fundamental de longitudes de paso de cada episode_{i}.
    """
    valid_files = [f for f in log_files if os.path.isfile(f)]

    # 1. Parsear episodios por cada archivo individual
    parsed_by_file = {}
    for filepath in valid_files:
        eps = parse_single_log(filepath)
        if eps:
            parsed_by_file[filepath] = eps

    # 2. Si hay dataset Zarr para alinear, buscar concordancia exacta de bloques
    if align_zarr and os.path.isdir(align_zarr):
        zarr_steps = read_zarr_step_counts(align_zarr)
        if zarr_steps:
            print(f"🔗 [Alineación Zarr] Detectados {len(zarr_steps)} episodios en '{align_zarr}'.")
            ordered_episodes = []
            remaining = dict(parsed_by_file)
            current_idx = 0
            matched_all = True

            while current_idx < len(zarr_steps) and remaining:
                matched_file = None
                for fpath, eps in remaining.items():
                    n = len(eps)
                    if current_idx + n <= len(zarr_steps):
                        file_steps = [e["steps"] for e in eps]
                        if zarr_steps[current_idx : current_idx + n] == file_steps:
                            matched_file = fpath
                            break

                if matched_file is not None:
                    eps = remaining.pop(matched_file)
                    for ep in eps:
                        ep["global_episode_idx"] = current_idx
                        current_idx += 1
                        ordered_episodes.append(ep)
                else:
                    matched_all = False
                    break

            if matched_all and current_idx == len(zarr_steps):
                print(f"✔ [Alineación Zarr] Concordancia matemática 100% alcanzada ({current_idx}/{len(zarr_steps)} episodios).")
                return ordered_episodes
            else:
                print(f"⚠ [Alineación Zarr] Alineación parcial hasta episodio {current_idx}. Usando fallback determinista.")

    # 3. Fallback: Ordenamiento determinista por sesión y batch
    valid_files.sort(key=fallback_sort_key)
    all_episodes = []
    global_idx = 0

    for filepath in valid_files:
        eps = parsed_by_file.get(filepath, [])
        for ep in eps:
            ep["global_episode_idx"] = global_idx
            global_idx += 1
            all_episodes.append(ep)

    return all_episodes


def export_to_csv(episodes: List[Dict[str, Any]], output_csv_path: str) -> None:
    """Exporta la lista de episodios estructurados a CSV."""
    if not episodes:
        print("[WARN] No hay episodios para exportar.")
        return

    out_path = Path(output_csv_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Orden canónico de columnas
    fieldnames = [
        "global_episode_idx",
        "episode_in_batch",
        "source_log",
        "timestamp_iso",
        "timestamp_ros",
        "obj_spawn_x",
        "obj_spawn_y",
        "obj_spawn_z",
        "obj_spawn_yaw_rad",
        "ee_jitter_dx_m",
        "ee_jitter_dy_m",
        "ee_jitter_droll_rad",
        "ee_jitter_dpitch_rad",
        "ee_jitter_dyaw_rad",
        "err_approach_xy_m",
        "err_approach_z_m",
        "err_align_rot_rad",
        "err_descend_3d_m",
        "err_descend_rot_rad",
        "err_move_target_3d_m",
        "err_move_target_rot_rad",
        "duration_approach_s",
        "duration_align_s",
        "duration_descend_s",
        "duration_settle_s",
        "duration_grasp_s",
        "duration_lift_s",
        "duration_untilt_s",
        "duration_move_s",
        "duration_hold_s",
        "steps",
        "duration_sim_s",
        "recording_wall_s",
        "cycle_wall_s",
        "rtf_speedup",
        "retries_before_success",
        "status",
    ]

    with open(out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(episodes)

    print(f"✔ [CSV Export] {len(episodes)} episodios exportados exitosamente a: {out_path}")


def sync_to_sqlite(
    episodes: List[Dict[str, Any]], dataset_name: str = "demonstrations_v8.zarr"
) -> int:
    """Inserta los episodios parseados en SQLite experiments/experiments.db."""
    if record_dataset_episode is None:
        print("[WARN] Módulo tracker SQLite no disponible.")
        return 0

    synced = 0
    for ep in episodes:
        obj_spawn = {
            "x": ep.get("obj_spawn_x", 0.0),
            "y": ep.get("obj_spawn_y", 0.0),
            "z": ep.get("obj_spawn_z", 0.44),
            "yaw": ep.get("obj_spawn_yaw_rad", 0.0),
        }
        ee_jitter = {
            "dx": ep.get("ee_jitter_dx_m", 0.0),
            "dy": ep.get("ee_jitter_dy_m", 0.0),
            "droll": ep.get("ee_jitter_droll_rad", 0.0),
            "dpitch": ep.get("ee_jitter_dpitch_rad", 0.0),
            "dyaw": ep.get("ee_jitter_dyaw_rad", 0.0),
        }
        phase_durations = {
            k: ep.get(k)
            for k in [
                "duration_approach_s",
                "duration_align_s",
                "duration_descend_s",
                "duration_settle_s",
                "duration_grasp_s",
                "duration_lift_s",
                "duration_untilt_s",
                "duration_move_s",
                "duration_hold_s",
            ]
            if ep.get(k) is not None
        }
        phase_errors = {
            k: ep.get(k)
            for k in [
                "err_approach_xy_m",
                "err_approach_z_m",
                "err_align_rot_rad",
                "err_descend_3d_m",
                "err_descend_rot_rad",
                "err_move_target_3d_m",
                "err_move_target_rot_rad",
            ]
            if ep.get(k) is not None
        }

        ok = record_dataset_episode(
            dataset_name=dataset_name,
            episode_idx=ep.get("global_episode_idx", ep.get("episode_in_batch", 0)),
            obj_spawn=obj_spawn,
            ee_jitter=ee_jitter,
            steps=ep.get("steps", 0),
            sim_duration_s=ep.get("duration_sim_s", 0.0),
            wall_duration_s=ep.get("recording_wall_s", 0.0),
            cycle_wall_s=ep.get("cycle_wall_s", 0.0),
            retry_count=ep.get("retries_before_success", 0),
            phase_durations=phase_durations,
            phase_errors=phase_errors,
            metadata={
                "source_log": ep.get("source_log"),
                "rtf_speedup": ep.get("rtf_speedup"),
                "status": ep.get("status"),
            },
            timestamp=ep.get("timestamp_iso"),
        )
        if ok:
            synced += 1

    print(f"✔ [SQLite Sync] {synced}/{len(episodes)} episodios registrados en SQLite experiments.db")
    return synced


def print_statistical_summary(episodes: List[Dict[str, Any]]) -> None:
    """Imprime un resumen estadístico consolidado de la campaña de recolección."""
    if not episodes:
        return

    import numpy as np

    steps_list = [e["steps"] for e in episodes]
    sim_durs = [e["duration_sim_s"] for e in episodes]
    wall_durs = [e["recording_wall_s"] for e in episodes]
    rtfs = [e["rtf_speedup"] for e in episodes]
    retries = [e["retries_before_success"] for e in episodes]

    err_app_xy = [e["err_approach_xy_m"] for e in episodes if e["err_approach_xy_m"] is not None]
    err_align_rot = [e["err_align_rot_rad"] for e in episodes if e["err_align_rot_rad"] is not None]
    err_descend_3d = [e["err_descend_3d_m"] for e in episodes if e["err_descend_3d_m"] is not None]
    err_move_3d = [e["err_move_target_3d_m"] for e in episodes if e["err_move_target_3d_m"] is not None]

    print("\n" + "=" * 80)
    print("📊 RESUMEN ESTADÍSTICO DE TELEMETRÍA — CAMPAÑA DE DEMOSTRACIONES")
    print("=" * 80)
    print(f"  • Total Episodios Analizados: {len(episodes)}")
    print(
        f"  • Pasos Temporales (20 Hz):  Media={np.mean(steps_list):.1f} | Min={np.min(steps_list)} | Max={np.max(steps_list)}"
    )
    print(
        f"  • Duración Simulada:        Media={np.mean(sim_durs):.2f}s | Min={np.min(sim_durs):.2f}s | Max={np.max(sim_durs):.2f}s"
    )
    print(
        f"  • Duración de Pared (Rec):  Media={np.mean(wall_durs):.2f}s | Total Acumulado={np.sum(wall_durs)/3600:.2f} h"
    )
    print(
        f"  • Aceleración RTF:           Media={np.mean(rtfs):.3f}x | Min={np.min(rtfs):.2f}x | Max={np.max(rtfs):.2f}x"
    )
    print(
        f"  • Reintentos / Fallos:       Media={np.mean(retries):.2f} | Tasa Éxito Limpio={np.mean([1 if r==0 else 0 for r in retries])*100:.1f}%"
    )

    if err_app_xy:
        print(
            f"  • Error Residual Approach:   Media XY={np.mean(err_app_xy)*1000:.2f} mm"
        )
    if err_align_rot:
        print(
            f"  • Error Residual Alineación: Media Rot={np.mean(err_align_rot):.4f} rad ({np.degrees(np.mean(err_align_rot)):.2f}°)"
        )
    if err_descend_3d:
        print(
            f"  • Error Residual Descend:    Media 3D={np.mean(err_descend_3d)*1000:.2f} mm"
        )
    if err_move_3d:
        print(
            f"  • Error Residual Destino:    Media 3D={np.mean(err_move_3d)*1000:.2f} mm"
        )
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Parsea logs de telemetría de recolección de demostraciones a CSV."
    )
    parser.add_argument(
        "--log_files",
        nargs="+",
        default=[],
        help="Rutas a archivos de registro específicos (/tmp/pickup_approximate_*.log)",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Directorio que contiene logs (ej. experiments/logs/expert_collection_v8)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="experiments/telemetry_demonstrations.csv",
        help="Ruta al archivo CSV de salida",
    )
    parser.add_argument(
        "--record_db",
        action="store_true",
        default=False,
        help="Sincronizar episodios extraídos en experiments/experiments.db",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="demonstrations_v8.zarr",
        help="Nombre canónico del dataset asociado",
    )
    parser.add_argument(
        "--align_zarr",
        type=str,
        default=None,
        help="Ruta al dataset Zarr para alinear matemáticamente el orden canónico de episodios (ej. datasets/demonstrations_v8.zarr)",
    )

    args = parser.parse_args()

    # Resolver align_zarr si no se especificó directamente
    align_target = args.align_zarr
    if not align_target:
        possible_paths = [
            os.path.join(REPO_ROOT, "datasets", args.dataset_name),
            os.path.join("datasets", args.dataset_name),
            args.dataset_name,
        ]
        for p in possible_paths:
            if os.path.isdir(p):
                align_target = p
                break

    files_to_parse = []
    if args.log_files:
        for p in args.log_files:
            files_to_parse.extend(glob.glob(p))

    if args.log_dir and os.path.isdir(args.log_dir):
        files_to_parse.extend(glob.glob(os.path.join(args.log_dir, "*.log")))

    # Si no se pasó ninguno, buscar por defecto en /tmp y en experiments/logs
    if not files_to_parse:
        candidates = [
            "/tmp/pickup_approximate_1.log",
            "/tmp/pickup_approximate_2.log",
            "experiments/logs/pickup_approximate_1.log",
            "experiments/logs/pickup_approximate_2.log",
        ]
        files_to_parse = [c for c in candidates if os.path.isfile(c)]
        if not files_to_parse:
            files_to_parse = glob.glob("experiments/logs/expert_collection_v8/*.log")

    if not files_to_parse:
        print("[ERROR FATAL] No se encontraron archivos de registro para parsear.")
        sys.exit(1)

    print(f"🔍 Parseando {len(files_to_parse)} archivo(s) de registro de telemetría...")
    episodes = parse_all_logs(files_to_parse, align_zarr=align_target)

    print(f"📊 Episodios extraídos con éxito: {len(episodes)}")
    export_to_csv(episodes, args.out)
    print_statistical_summary(episodes)

    if args.record_db:
        sync_to_sqlite(episodes, dataset_name=args.dataset_name)


if __name__ == "__main__":
    main()
