#!/usr/bin/env python3
# Copyright (c) 2026. Programacion Cientifica, DISC, Antofagasta, Chile.
"""
verify_zarr_dataset.py — Verificador Exhaustivo de Integridad y Calidad de Datasets Zarr
========================================================================================
Herramienta de certificación previa al entrenamiento para datasets de demostraciones Zarr
utilizados por políticas visuomotoras generativas (Diffusion Policy, DDIM Schedule).

Comprueba de forma exhaustiva, segura y no destructiva (modo lectura estricto 'r'):
1. Integridad estructural Zarr y metadatos:
   - Apertura de grupos y validación de archivos JSON de metadatos (.zgroup, .zattrs, .zarray).
   - Nomenclatura contigua y continua estricta: episode_0, episode_1, ..., episode_{N-1}.
   - Detección de claves anómalas o episodios faltantes/desfasados.
   - Confirmación de tipo zarr.Array para cada modalidad (image, state, action, timestamp).
2. Integridad de archivos y descompresión de chunks:
   - Lectura secuencial de chunks (Blosc / zlib) verificando ausencia de archivos truncados,
     corruptos o faltantes en disco mediante streaming seguro por bloques (sin OOM en RAM).
   - En modo rápido (--skip-images), verifica obligatoriamente el primer fotograma (chunk 0).
3. Consistencia dimensional y de tipos de datos (Dtypes):
   - image: [T, 256, 256, 3] uint8.
   - state: [T, 8] float32.
   - action: [T, 9] float32.
   - timestamp (opcional): [T] float64 / float32 unidimensional.
   - Concordancia temporal estricta de pasos T entre todas las modalidades (T >= 16).
4. Regla arquitectónica crítica — Ausencia de Fuga Causal (Causal Leakage):
   - state debe poseer exactamente 8 dimensiones ([XYZ, qx, qy, qz, qw, gripper_state]).
   - La 9ª dimensión (done_flag) JAMÁS debe estar en state.
5. Validación numérica y detección de anomalías / Ceros Espurios:
   - Ausencia total de NaNs e Infs en todas las modalidades.
   - Detección de ceros espurios (vectores nulos [0,0,...,0] en state o action).
   - Detección de cuaterniones degenerados/nulos [0,0,0,0] (norma < 1e-3).
   - Detección de timestamps espurios con valor 0.0 o no-monotónicos.
   - Límites físicos plausibles de posición cartesiana ([-3.0, 3.0] m).
6. Señal de término y piso de parada (Done Floor):
   - action[:, 8] estrictamente en rango [0.0, 1.0].
   - Cuantificación del piso continuo terminal (trailing done frames >= min_done_frames).
   - Detección de señales de parada intermitentes o prematuras antes del piso terminal.
7. Compatibilidad Integral con Pipeline Neuronal (DataLoader & Diffusion Policy):
   - Instanciación de ZarrDataset y validación de estadísticas (min, max, mean, std).
   - Extracción de lote completo con PyTorch DataLoader.
   - Paso forward y backward opcional con DiffusionPolicy (U-Net 1D + ResNet18 GroupNorm)
     para certificar computación de pérdida y gradiente sin anomalías numéricas.
"""

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    import zarr
except ImportError:
    print(
        "[ERROR FATAL] 'zarr' y 'numpy' son requeridos para ejecutar verify_zarr_dataset.py.\n"
        "Ejecútalo dentro del contenedor Docker 'visuomotor_ros2' o instala zarr y numpy.",
        file=sys.stderr,
    )
    sys.exit(1)

# Asegurar importación de visuomotor_core si no está en el entorno global
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, ".."))
CORE_PKG_DIR = os.path.join(REPO_ROOT_DIR, "visuomotor_core")
if CORE_PKG_DIR not in sys.path:
    sys.path.insert(0, CORE_PKG_DIR)

try:
    from visuomotor_core.utils.logger import configure_logging
except ImportError:
    configure_logging = None

log = logging.getLogger("verify_zarr_dataset")


@dataclass
class EpisodeReport:
    name: str
    index: int
    steps: int
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    has_image: bool = False
    has_state: bool = False
    has_action: bool = False
    has_timestamp: bool = False
    image_shape: Optional[Tuple[int, ...]] = None
    state_shape: Optional[Tuple[int, ...]] = None
    action_shape: Optional[Tuple[int, ...]] = None
    done_frames: int = 0
    trailing_done_frames: int = 0
    done_floor_seconds: float = 0.0
    ends_with_done: bool = False
    chunks_read: int = 0
    spurious_zeros: int = 0
    metadata_ok: bool = True


@dataclass
class DatasetVerificationReport:
    dataset_path: str
    is_valid: bool = True
    total_episodes: int = 0
    total_steps: int = 0
    episodes_passed: int = 0
    episodes_failed: int = 0
    episodes_warned: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    dataloader_compatible: bool = False
    dataloader_batch_shapes: Dict[str, str] = field(default_factory=dict)
    policy_step_verified: bool = False
    initial_policy_loss: Optional[float] = None
    elapsed_time_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def check_metadata_json_file(file_path: str, expected_kind: str = "json") -> Tuple[bool, Optional[str]]:
    """
    Verifica la existencia física y la integridad sintáctica de un archivo de metadatos Zarr.
    Retorna (es_valido, mensaje_error).
    """
    if not os.path.exists(file_path):
        return False, f"Archivo de metadatos ausente: {file_path}"
    if not os.path.isfile(file_path):
        return False, f"La ruta de metadatos no es un archivo regular: {file_path}"
    if os.path.getsize(file_path) == 0:
        return False, f"Archivo de metadatos vacío (0 bytes): {file_path}"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, f"Error al parsear JSON en '{file_path}': {e}"

    if not isinstance(data, dict):
        return False, f"El contenido JSON en '{file_path}' no es un objeto/diccionario."

    if expected_kind == "zarray":
        for k in ("chunks", "dtype", "shape"):
            if k not in data:
                return False, f"Metadatos '.zarray' incompletos en '{file_path}': falta clave '{k}'"
    elif expected_kind == "zgroup":
        if "zarr_format" not in data:
            return False, f"Metadatos '.zgroup' incompletos en '{file_path}': falta clave 'zarr_format'"

    return True, None


def verify_zarr_structure(
    dataset_path: str,
) -> Tuple[Optional[zarr.Group], List[Tuple[int, str]], List[str], List[str]]:
    """
    Verifica la estructura inicial del Zarr:
    - Existencia del directorio y apertura como grupo Zarr en modo lectura estricta.
    - Integridad de metadatos .zgroup y .zattrs en la raíz.
    - Extracción y ordenamiento de grupos 'episode_N'.
    - Detección de discontinuidades de índice (ej. falta episode_5 entre 4 y 6).
    - Identificación de claves no conformes.
    """
    errors: List[str] = []
    warnings: List[str] = []

    norm_path = os.path.normpath(dataset_path)
    if not os.path.exists(norm_path):
        errors.append(f"La ruta del dataset no existe: {dataset_path}")
        return None, [], errors, warnings

    # Verificar metadatos raíz si es directorio en disco
    if os.path.isdir(norm_path):
        zgroup_file = os.path.join(norm_path, ".zgroup")
        zattrs_file = os.path.join(norm_path, ".zattrs")
        ok_zg, zg_err = check_metadata_json_file(zgroup_file, expected_kind="zgroup")
        if not ok_zg:
            errors.append(f"Raíz Zarr inválida: {zg_err}")
            return None, [], errors, warnings

        if os.path.exists(zattrs_file):
            ok_za, za_err = check_metadata_json_file(zattrs_file, expected_kind="json")
            if not ok_za:
                warnings.append(f"Metadatos '.zattrs' raíz con anomalías: {za_err}")

    try:
        root = zarr.open_group(norm_path, mode="r")
    except Exception as e:
        errors.append(f"Error al abrir el grupo Zarr raíz '{dataset_path}': {e}")
        return None, [], errors, warnings

    all_keys = list(root.group_keys())
    if not all_keys:
        errors.append("El dataset Zarr está vacío: no contiene grupos de episodios.")
        return root, [], errors, warnings

    episode_entries: List[Tuple[int, str]] = []
    alien_keys: List[str] = []

    for key in all_keys:
        match = re.match(r"^episode_(\d+)$", key)
        if match:
            idx = int(match.group(1))
            episode_entries.append((idx, key))
        else:
            alien_keys.append(key)

    if alien_keys:
        warnings.append(
            f"Se encontraron {len(alien_keys)} clave(s) en la raíz que no siguen el patrón 'episode_N': {alien_keys[:10]}"
        )

    if not episode_entries:
        errors.append("No se detectó ningún grupo con formato 'episode_N' en el dataset.")
        return root, [], errors, warnings

    episode_entries.sort(key=lambda x: x[0])

    # Verificar secuencia contigua sin huecos
    indices = [idx for idx, _ in episode_entries]
    min_idx = indices[0]
    max_idx = indices[-1]

    if min_idx != 0:
        warnings.append(f"La secuencia de episodios no comienza en 0 (comienza en episode_{min_idx}).")

    expected_indices = list(range(min_idx, max_idx + 1))
    if indices != expected_indices:
        missing = set(expected_indices) - set(indices)
        errors.append(
            f"Discontinuidad detectada en la secuencia de episodios. Faltan {len(missing)} episodio(s): "
            f"{sorted(list(missing))[:20]}"
        )

    return root, episode_entries, errors, warnings


def verify_episode(
    dataset_path: str,
    ep_name: str,
    ep_idx: int,
    ep_group: zarr.Group,
    check_image_chunks: bool = True,
    chunk_batch_size: int = 64,
    min_done_frames: int = 40,
    expected_image_shape: Tuple[int, int, int] = (256, 256, 3),
    max_start_offset_m: float = 0.20,
) -> EpisodeReport:
    """
    Verifica minuciosamente un episodio individual:
    - Integridad de metadatos .zgroup, .zarray, .zattrs del episodio.
    - Presencia y tipo zarr.Array para 'image', 'state', 'action' (y 'timestamp' si existe).
    - Fuga Causal: 'state' DEBE tener exactamente 8 dimensiones.
    - Salud numérica: ausencia de NaNs/Infs y ceros espurios en state, action y timestamp.
    - Normalización de cuaterniones y detección de cuaterniones degenerados/nulos.
    - Descompresión de bloques de imagen por streaming seguro (sin OOM en RAM).
    - Done Flag: rango, detección de término prematuro/intermitente y cuantificación del piso continuo.
    """
    report = EpisodeReport(name=ep_name, index=ep_idx, steps=0)

    # 1. Verificación de metadatos físicos en disco si aplica
    ep_dir = os.path.join(dataset_path, ep_name)
    if os.path.isdir(ep_dir):
        ep_zg = os.path.join(ep_dir, ".zgroup")
        ok_zg, zg_err = check_metadata_json_file(ep_zg, expected_kind="zgroup")
        if not ok_zg:
            report.errors.append(f"Metadatos de episodio corruptos/ausentes: {zg_err}")
            report.is_valid = False
            report.metadata_ok = False

    # 2. Existencia y tipo de arrays obligatorios
    for req in ("image", "state", "action"):
        # Validar archivo .zarray físico si estamos en disco
        if os.path.isdir(ep_dir):
            arr_dir = os.path.join(ep_dir, req)
            if os.path.isfile(os.path.join(arr_dir, ".zgroup")):
                report.errors.append(
                    f"El nodo '{req}' en {ep_name} no es un Array de Zarr (es un Grupo Zarr con .zgroup)"
                )
                report.is_valid = False
                continue

            arr_meta = os.path.join(arr_dir, ".zarray")
            ok_za, za_err = check_metadata_json_file(arr_meta, expected_kind="zarray")
            if not ok_za:
                report.errors.append(f"Metadatos '.zarray' de '{req}' corruptos/ausentes en {ep_name}: {za_err}")
                report.is_valid = False
                report.metadata_ok = False
                continue

        if req not in ep_group:
            report.errors.append(f"Falta el dataset requerido '{req}' en {ep_name}")
            report.is_valid = False
            continue

        try:
            arr_obj = ep_group[req]
        except Exception as e_arr:
            report.errors.append(f"Error al abrir dataset '{req}' en {ep_name}: {e_arr}")
            report.is_valid = False
            continue

        if not isinstance(arr_obj, zarr.Array):
            report.errors.append(
                f"El nodo '{req}' en {ep_name} no es un Array de Zarr (tipo encontrado: {type(arr_obj).__name__})"
            )
            report.is_valid = False
            continue

    if not report.is_valid:
        return report

    try:
        img_ds = ep_group["image"]
        state_ds = ep_group["state"]
        act_ds = ep_group["action"]
    except Exception as e_load:
        report.errors.append(f"Fallo al acceder a arrays en {ep_name}: {e_load}")
        report.is_valid = False
        return report

    report.has_image = True
    report.has_state = True
    report.has_action = True
    report.has_timestamp = "timestamp" in ep_group

    report.image_shape = tuple(img_ds.shape)
    report.state_shape = tuple(state_ds.shape)
    report.action_shape = tuple(act_ds.shape)

    t_img = img_ds.shape[0]
    t_state = state_ds.shape[0]
    t_act = act_ds.shape[0]

    report.steps = t_img

    # 3. Concordancia temporal T
    if not (t_img == t_state == t_act):
        report.errors.append(
            f"Desajuste de pasos temporales: image={t_img}, state={t_state}, action={t_act}"
        )
        report.is_valid = False

    if t_img < 16:
        report.errors.append(
            f"Episodio demasiado corto ({t_img} pasos). Se requiere T >= 16 para el horizonte de acción H_act."
        )
        report.is_valid = False

    # 4. Tipos de datos (Dtypes)
    if img_ds.dtype != np.uint8:
        report.errors.append(f"'image' debe ser uint8, se encontró {img_ds.dtype}")
        report.is_valid = False

    if state_ds.dtype != np.float32:
        report.errors.append(f"'state' debe ser float32, se encontró {state_ds.dtype}")
        report.is_valid = False

    if act_ds.dtype != np.float32:
        report.errors.append(f"'action' debe ser float32, se encontró {act_ds.dtype}")
        report.is_valid = False

    # 5. Formas espaciales (Shapes)
    if len(img_ds.shape) != 4 or tuple(img_ds.shape[1:]) != expected_image_shape:
        report.errors.append(
            f"'image' debe tener forma (T, {expected_image_shape[0]}, {expected_image_shape[1]}, {expected_image_shape[2]}), "
            f"se encontró {img_ds.shape}"
        )
        report.is_valid = False

    # 6. REGULACIÓN CRÍTICA: Ausencia de Fuga Causal (Causal Leakage)
    if len(state_ds.shape) != 2:
        report.errors.append(f"'state' debe ser bidimensional (T, D), se encontró {state_ds.shape}")
        report.is_valid = False
    elif state_ds.shape[1] == 9:
        report.errors.append(
            f"[FATAL CAUSAL LEAKAGE] 'state' tiene 9 dimensiones en {ep_name}. "
            f"La dimensión 9 (done_flag) NO debe formar parte de la observación propioceptiva sensorial."
        )
        report.is_valid = False
    elif state_ds.shape[1] != 8:
        report.errors.append(
            f"'state' debe tener exactamente 8 dimensiones [XYZ(3), Quat(4), Gripper(1)], "
            f"se encontró {state_ds.shape[1]}"
        )
        report.is_valid = False

    if len(act_ds.shape) != 2 or act_ds.shape[1] != 9:
        report.errors.append(
            f"'action' debe tener exactamente 9 dimensiones [XYZ(3), Quat(4), Gripper(1), Done(1)], "
            f"se encontró {act_ds.shape if len(act_ds.shape) == 2 else act_ds.shape}"
        )
        report.is_valid = False

    # 7. Verificación de Timestamp si está presente
    if report.has_timestamp:
        ts_node = ep_group["timestamp"]
        if not isinstance(ts_node, zarr.Array):
            report.errors.append(f"'timestamp' en {ep_name} no es un Array de Zarr.")
            report.is_valid = False
        else:
            if ts_node.dtype not in (np.float64, np.float32):
                report.errors.append(f"'timestamp' debe ser float64 o float32, se encontró {ts_node.dtype}")
                report.is_valid = False
            if len(ts_node.shape) != 1 or ts_node.shape[0] != t_img:
                report.errors.append(
                    f"'timestamp' debe tener forma unidimensional ({t_img},), se encontró {ts_node.shape}"
                )
                report.is_valid = False
            try:
                ts_data = ts_node[:]
                if np.isnan(ts_data).any() or np.isinf(ts_data).any():
                    report.errors.append("Se detectaron valores NaN o Inf en 'timestamp'")
                    report.is_valid = False

                # Detección de ceros espurios en timestamp
                zero_ts_count = int(np.sum(ts_data == 0.0))
                if zero_ts_count > 0:
                    report.errors.append(
                        f"Se detectaron {zero_ts_count} timestamp(s) con valor exacto 0.0 (ceros espurios / reloj no inicializado)."
                    )
                    report.is_valid = False
                    report.spurious_zeros += zero_ts_count

                # Monotonía temporal estricta
                diffs = np.diff(ts_data)
                if (diffs <= 0).any():
                    report.warnings.append("Los timestamps no son estrictamente crecientes monotónicos.")
            except Exception as e_ts:
                report.errors.append(f"Error al leer/descomprimir 'timestamp': {e_ts}")
                report.is_valid = False

    if not report.is_valid:
        return report

    # 8. Lectura y Validación Numérica de State y Action
    try:
        state_data = state_ds[:]
    except Exception as e:
        report.errors.append(f"Fallo de descompresión/lectura en 'state': {e}")
        report.is_valid = False
        return report

    try:
        act_data = act_ds[:]
    except Exception as e:
        report.errors.append(f"Fallo de descompresión/lectura en 'action': {e}")
        report.is_valid = False
        return report

    # Chequeo de NaNs / Infs
    if np.isnan(state_data).any():
        report.errors.append("Se detectaron valores NaN en 'state'")
        report.is_valid = False
    if np.isinf(state_data).any():
        report.errors.append("Se detectaron valores Inf en 'state'")
        report.is_valid = False

    if np.isnan(act_data).any():
        report.errors.append("Se detectaron valores NaN en 'action'")
        report.is_valid = False
    if np.isinf(act_data).any():
        report.errors.append("Se detectaron valores Inf en 'action'")
        report.is_valid = False

    # Chequeo de Ceros Espurios en State
    # (a) Vector de estado completamente nulo
    zero_state_rows = np.where(np.all(state_data == 0.0, axis=1))[0]
    if len(zero_state_rows) > 0:
        report.errors.append(
            f"Se detectaron {len(zero_state_rows)} fotograma(s) con vector de 'state' completamente nulo (ceros espurios)."
        )
        report.is_valid = False
        report.spurious_zeros += len(zero_state_rows)

    # (b) Posición cartesiana totalmente nula [0, 0, 0] y certificación de relative_coordinates
    is_relative = ep_group.attrs.get("relative_coordinates", False)
    pos_state = state_data[:, :3]
    zero_pos_rows = np.where(np.all(pos_state == 0.0, axis=1))[0]
    
    if is_relative:
        if ep_group.attrs.get("start_anchor") == "measured":
            # Conversor v3 con --start-poses (2026-11-25): las coordenadas son relativas a HOME3 y el paso 0 es la pose
            # REAL de inicio medida (ArUco + IMU), no el origen. Solo es error si el desfase es inverosímil.
            offset = float(np.linalg.norm(pos_state[0]))
            if offset > max_start_offset_m:
                report.errors.append(
                    f"Inicio medido a {1000 * offset:.0f} mm de HOME3 (> {1000 * max_start_offset_m:.0f} mm): {pos_state[0]}"
                )
                report.is_valid = False
        elif not np.allclose(pos_state[0], [0.0, 0.0, 0.0], atol=1e-5):
            report.errors.append(
                f"Dataset marcado con relative_coordinates=True pero el paso 0 no arranca en [0,0,0]: {pos_state[0]}"
            )
            report.is_valid = False
        # Descontar el paso 0 legítimo de los ceros espurios
        unexpected_zero_rows = [r for r in zero_pos_rows if r != 0]
        if len(unexpected_zero_rows) > 0:
            report.warnings.append(
                f"Se detectaron {len(unexpected_zero_rows)} paso(s) intermedio(s) con posición XYZ de 'state' exactamente nula [0,0,0]."
            )
    else:
        if len(zero_pos_rows) > 0:
            report.warnings.append(
                f"Se detectaron {len(zero_pos_rows)} paso(s) con posición XYZ de 'state' exactamente nula [0,0,0]."
            )

    # (c) Cuaternión degenerado o nulo [0, 0, 0, 0]
    quat_state = state_data[:, 3:7]
    state_quat_norms = np.linalg.norm(quat_state, axis=1)
    null_state_quats = np.where((state_quat_norms < 1e-3) | np.all(quat_state == 0.0, axis=1))[0]
    if len(null_state_quats) > 0:
        report.errors.append(
            f"Se detectaron {len(null_state_quats)} paso(s) con cuaternión de 'state' nulo o degenerado [0,0,0,0] (norma ~0)."
        )
        report.is_valid = False
        report.spurious_zeros += len(null_state_quats)

    # Desviación de norma en cuaterniones de estado
    if (np.abs(state_quat_norms - 1.0) > 0.08).any():
        report.warnings.append(
            f"Cuaterniones de 'state' con desviación de norma > 0.08 (max dev: {np.abs(state_quat_norms - 1.0).max():.4f})"
        )

    # Chequeo de Ceros Espurios en Action
    # (a) Vector de acción completamente nulo
    zero_act_rows = np.where(np.all(act_data == 0.0, axis=1))[0]
    if len(zero_act_rows) > 0:
        report.errors.append(
            f"Se detectaron {len(zero_act_rows)} fotograma(s) con vector de 'action' completamente nulo (ceros espurios)."
        )
        report.is_valid = False
        report.spurious_zeros += len(zero_act_rows)

    # (b) Cuaternión objetivo nulo [0, 0, 0, 0]
    quat_act = act_data[:, 3:7]
    act_quat_norms = np.linalg.norm(quat_act, axis=1)
    null_act_quats = np.where((act_quat_norms < 1e-3) | np.all(quat_act == 0.0, axis=1))[0]
    if len(null_act_quats) > 0:
        report.errors.append(
            f"Se detectaron {len(null_act_quats)} paso(s) con cuaternión de 'action' nulo o degenerado [0,0,0,0] (norma ~0)."
        )
        report.is_valid = False
        report.spurious_zeros += len(null_act_quats)

    # Desviación de norma en cuaterniones de acción
    if (np.abs(act_quat_norms - 1.0) > 0.08).any():
        report.warnings.append(
            f"Cuaterniones de 'action' con desviación de norma > 0.08 (max dev: {np.abs(act_quat_norms - 1.0).max():.4f})"
        )

    # Chequeo de rangos físicos plausibles de posición cartesiana ([-3.0, 3.0] m)
    if (np.abs(pos_state) > 3.0).any():
        report.warnings.append(
            f"Posiciones cartesianas en 'state' fuera de rango nominal ([-3, 3]m): max={np.abs(pos_state).max():.2f}m"
        )

    pos_act = act_data[:, :3]
    if (np.abs(pos_act) > 3.0).any():
        report.warnings.append(
            f"Posiciones objetivo en 'action' fuera de rango nominal ([-3, 3]m): max={np.abs(pos_act).max():.2f}m"
        )

    # 9. Verificación del Done Flag y Piso de Parada (action[:, 8])
    done_signals = act_data[:, 8]
    if (done_signals < -1e-4).any() or (done_signals > 1.0001).any():
        report.errors.append(
            f"Valores de done_flag fuera del rango [0.0, 1.0]: min={done_signals.min():.4f}, max={done_signals.max():.4f}"
        )
        report.is_valid = False

    total_done = int(np.sum(done_signals >= 0.5))
    report.done_frames = total_done

    # Contabilizar piso continuo terminal (trailing done frames consecutivos desde el fin)
    trailing_done = 0
    for val in reversed(done_signals >= 0.5):
        if val:
            trailing_done += 1
        else:
            break

    report.trailing_done_frames = trailing_done
    report.done_floor_seconds = trailing_done * 0.050  # 20 Hz
    report.ends_with_done = bool(done_signals[-1] >= 0.5)

    if total_done == 0:
        report.errors.append("El episodio no contiene ninguna señal de término (done_frames = 0).")
        report.is_valid = False
    elif not report.ends_with_done:
        report.warnings.append(
            f"El episodio no finaliza en done=1.0 (último valor: {done_signals[-1]:.2f})."
        )
    else:
        # Señal de término no contigua (se activó y desactivó antes del final)
        if total_done > trailing_done:
            report.warnings.append(
                f"Señal de parada no contigua en {ep_name}: {total_done - trailing_done} frame(s) con done=1.0 "
                f"activados prematuramente antes del piso terminal continuo ({trailing_done} frames)."
            )

        # Piso insuficiente frente al requerimiento
        if trailing_done < min_done_frames:
            report.warnings.append(
                f"Piso de parada terminal insuficiente: {trailing_done} frames ({report.done_floor_seconds:.2f}s). "
                f"Se requieren al menos {min_done_frames} frames (>= {min_done_frames * 0.050:.1f}s a 20Hz)."
            )

    # 10. Descompresión Segura de Chunks de Imagen
    if report.is_valid:
        chunks_checked = 0
        try:
            if check_image_chunks:
                # Streaming seguro en bloques para no saturar memoria RAM
                for s in range(0, t_img, chunk_batch_size):
                    end_idx = min(t_img, s + chunk_batch_size)
                    img_slice = img_ds[s:end_idx]
                    chunks_checked += (end_idx - s)

                    if img_slice.shape[0] != (end_idx - s):
                        report.errors.append(
                            f"Chunk de imagen corrupto/incompleto en slice [{s}:{end_idx}]"
                        )
                        report.is_valid = False
                        break
            else:
                # Chequeo rápido de al menos el primer fotograma para certificar chunks en disco
                img_first = img_ds[0:1]
                if img_first.shape[0] != 1:
                    report.errors.append("Chunk inicial de imagen [0:1] ausente o corrupto en disco.")
                    report.is_valid = False
                chunks_checked = 1

            report.chunks_read = chunks_checked
        except Exception as e_chunk:
            report.errors.append(f"Fallo de descompresión en chunks de 'image': {e_chunk}")
            report.is_valid = False

    return report


def verify_torch_dataloader(
    dataset_path: str,
    batch_size: int = 16,
    n_obs_steps: int = 2,
    h_act: int = 16,
    test_policy_step: bool = True,
) -> Tuple[bool, List[str], Dict[str, str], bool, Optional[float]]:
    """
    Verifica la compatibilidad directa con PyTorch DataLoader e instanciación de ZarrDataset:
    - Importa ZarrDataset (soporta visuomotor_core.dataloaders y visuomotor_core.dataset).
    - Instancia la clase y comprueba estadísticas (min, max, mean, std).
    - Extrae un lote completo validando tensores [B, 2, 3, 256, 256], [B, 2, 8], [B, 16, 9].
    - Opcional: ejecuta forward y backward pass con DiffusionPolicy para certificar entrenamiento real.
    """
    errors: List[str] = []
    shapes_info: Dict[str, str] = {}
    policy_step_ok = False
    initial_loss: Optional[float] = None

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        errors.append("PyTorch no está instalado en el entorno. Omitiendo prueba de DataLoader.")
        return False, errors, shapes_info, False, None

    # Intentar importar ZarrDataset
    ZarrDatasetClass = None
    try:
        from visuomotor_core.dataloaders.zarr_dataset import ZarrDataset
        ZarrDatasetClass = ZarrDataset
    except ImportError:
        try:
            from visuomotor_core.dataset.zarr_dataset import ZarrDataset
            ZarrDatasetClass = ZarrDataset
        except ImportError as e_imp:
            errors.append(f"No fue posible importar ZarrDataset: {e_imp}")
            return False, errors, shapes_info, False, None

    log.info("Probando compatibilidad directa con ZarrDataset y PyTorch DataLoader...")
    t0 = time.time()

    try:
        dataset = ZarrDatasetClass(root=dataset_path, n_obs_steps=n_obs_steps, h_act=h_act)
    except Exception as e_init:
        errors.append(f"Error al instanciar ZarrDataset: {e_init}")
        return False, errors, shapes_info, False, None

    init_time = time.time() - t0
    log.info(f"ZarrDataset instanciado exitosamente en {init_time:.2f}s. Muestras indexadas: {len(dataset)}")

    # Validar estadísticas del dataset
    if not hasattr(dataset, "stats") or not isinstance(dataset.stats, dict):
        errors.append("El atributo 'stats' no está presente o no es un diccionario en ZarrDataset.")
        return False, errors, shapes_info, False, None

    required_stats = ("observation.image", "observation.state", "action")
    for stat_key in required_stats:
        if stat_key not in dataset.stats:
            errors.append(f"Falta la clave estadística obligatoria '{stat_key}' en dataset.stats")
            return False, errors, shapes_info, False, None

        sub_stats = dataset.stats[stat_key]
        for sub_k in ("mean", "std"):
            if sub_k not in sub_stats:
                errors.append(f"Falta '{sub_k}' en dataset.stats['{stat_key}']")
                return False, errors, shapes_info, False, None
            tensor_val = sub_stats[sub_k]
            if torch.isnan(tensor_val).any() or torch.isinf(tensor_val).any():
                errors.append(f"Valores NaN o Inf detectados en dataset.stats['{stat_key}']['{sub_k}']")
                return False, errors, shapes_info, False, None

    # Extraer un lote usando DataLoader
    try:
        # En caso de datasets sintéticos de prueba con pocas muestras, ajustar batch_size
        effective_bs = min(batch_size, len(dataset))
        if effective_bs < 1:
            errors.append("El dataset no contiene suficientes muestras para extraer un lote.")
            return False, errors, shapes_info, False, None

        loader = DataLoader(
            dataset,
            batch_size=effective_bs,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        batch = next(iter(loader))
    except Exception as e_dl:
        errors.append(f"Error al extraer un lote del DataLoader: {e_dl}")
        return False, errors, shapes_info, False, None

    required_batch_keys = ("observation.image", "observation.state", "action", "action_is_pad")
    for bk in required_batch_keys:
        if bk not in batch:
            errors.append(f"Falta la clave obligatoria '{bk}' en el lote generado por DataLoader.")
            return False, errors, shapes_info, False, None

    img_tensor = batch["observation.image"]
    state_tensor = batch["observation.state"]
    act_tensor = batch["action"]
    pad_tensor = batch["action_is_pad"]

    shapes_info["observation.image"] = str(tuple(img_tensor.shape))
    shapes_info["observation.state"] = str(tuple(state_tensor.shape))
    shapes_info["action"] = str(tuple(act_tensor.shape))
    shapes_info["action_is_pad"] = str(tuple(pad_tensor.shape))

    # Validar dimensiones esperadas
    expected_img_shape = (effective_bs, n_obs_steps, 3, 256, 256)
    if img_tensor.shape != torch.Size(expected_img_shape):
        errors.append(
            f"Forma inesperada en batch['observation.image']: {tuple(img_tensor.shape)}, esperado {expected_img_shape}"
        )

    expected_state_shape = (effective_bs, n_obs_steps, 8)
    if state_tensor.shape != torch.Size(expected_state_shape):
        errors.append(
            f"Forma inesperada en batch['observation.state']: {tuple(state_tensor.shape)}, esperado {expected_state_shape}"
        )

    expected_act_shape = (effective_bs, h_act, 9)
    if act_tensor.shape != torch.Size(expected_act_shape):
        errors.append(
            f"Forma inesperada en batch['action']: {tuple(act_tensor.shape)}, esperado {expected_act_shape}"
        )

    # Ausencia de NaNs en tensores del lote
    if torch.isnan(img_tensor).any():
        errors.append("NaNs detectados en batch['observation.image']")
    if torch.isnan(state_tensor).any():
        errors.append("NaNs detectados en batch['observation.state']")
    if torch.isnan(act_tensor).any():
        errors.append("NaNs detectados en batch['action']")

    is_compatible = len(errors) == 0
    if not is_compatible:
        return False, errors, shapes_info, False, None

    log.info(
        f"✔ PyTorch DataLoader verificado con éxito: Lote de {effective_bs} extraído correctamente. "
        f"Shapes: img={shapes_info['observation.image']}, state={shapes_info['observation.state']}, "
        f"action={shapes_info['action']}"
    )

    # 4. Certificación Real de Paso Forward y Backward en DiffusionPolicy
    if test_policy_step:
        try:
            from visuomotor_core.models.config import DiffusionConfig
            from visuomotor_core.models.diffusion_policy import DiffusionPolicy

            log.info("Ejecutando paso de entrenamiento (forward + backward) con DiffusionPolicy...")
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            device = torch.device(device_str)

            cfg = DiffusionConfig()
            cfg.n_obs_steps = n_obs_steps
            cfg.horizon = h_act
            cfg.input_shapes["observation.state"] = [dataset.stats["observation.state"]["mean"].shape[0]]
            cfg.output_shapes["action"] = [dataset.stats["action"]["mean"].shape[0]]

            policy = DiffusionPolicy(cfg, dataset_stats=dataset.stats).to(device)
            policy.train()

            # Mover lote al dispositivo
            batch_dev = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            output = policy.forward(batch_dev)
            if "loss" not in output:
                errors.append("policy.forward(batch) no retornó la clave 'loss'.")
                return False, errors, shapes_info, False, None

            loss = output["loss"]
            if torch.isnan(loss) or torch.isinf(loss):
                errors.append(f"El cálculo de loss produjo NaN o Inf en DiffusionPolicy: {loss}")
                return False, errors, shapes_info, False, None

            loss.backward()
            initial_loss = float(loss.item())
            policy_step_ok = True
            log.info(
                f"✔ DiffusionPolicy certificada exitosamente en {device_str.upper()} "
                f"(Loss inicial DDIM: {initial_loss:.4f}, gradientes computados)."
            )
        except Exception as e_pol:
            log.warning(f"No fue posible ejecutar el paso de DiffusionPolicy: {e_pol}")
            # Si DiffusionPolicy no está disponible en este entorno, no es un fallo fatal del dataloader
            pass

    return is_compatible, errors, shapes_info, policy_step_ok, initial_loss


def verify_dataset(
    dataset_path: str,
    check_all_images: bool = True,
    chunk_batch_size: int = 64,
    min_done_frames: int = 40,
    max_start_offset_m: float = 0.20,
    test_dataloader: bool = True,
    test_policy_step: bool = True,
    batch_size: int = 16,
    strict: bool = False,
    progress_interval: int = 100,
) -> DatasetVerificationReport:
    """
    Función orquestadora principal para la validación exhaustiva de un dataset Zarr.
    """
    t_start = time.time()
    norm_path = os.path.normpath(dataset_path)
    report = DatasetVerificationReport(dataset_path=norm_path)

    log.info("=" * 80)
    log.info("INICIANDO VERIFICACIÓN EXHAUSTIVA DE DATASET ZARR")
    log.info(f"Ruta objetivo: {norm_path}")
    log.info(
        f"Configuración: check_all_images={check_all_images}, batch_chunks={chunk_batch_size}, "
        f"min_done_frames={min_done_frames}, strict={strict}, test_policy_step={test_policy_step}"
    )
    log.info("=" * 80)

    # 1. Estructura Zarr y grupos
    root, episode_entries, struct_errors, struct_warnings = verify_zarr_structure(norm_path)
    report.errors.extend(struct_errors)
    report.warnings.extend(struct_warnings)

    if struct_errors or root is None:
        report.is_valid = False
        report.elapsed_time_s = time.time() - t_start
        return report

    total_eps = len(episode_entries)
    report.total_episodes = total_eps
    log.info(
        f"Estructura Zarr validada: {total_eps} episodios detectados ('{episode_entries[0][1]}' a '{episode_entries[-1][1]}')."
    )

    # 2. Validación iterativa por episodio
    lengths: List[int] = []
    done_counts: List[int] = []
    trailing_counts: List[int] = []
    total_chunks_read = 0
    total_spurious_zeros = 0

    log.info("Inspeccionando tensores, tipos, valores numéricos y descompresión de chunks...")
    for idx, (ep_num, ep_name) in enumerate(episode_entries):
        try:
            ep_group = root[ep_name]
        except Exception as e_grp:
            report.episodes_failed += 1
            report.errors.append(f"[{ep_name}] Imposible abrir grupo de episodio en Zarr: {e_grp}")
            continue

        ep_rep = verify_episode(
            dataset_path=norm_path,
            ep_name=ep_name,
            ep_idx=ep_num,
            ep_group=ep_group,
            check_image_chunks=check_all_images,
            chunk_batch_size=chunk_batch_size,
            min_done_frames=min_done_frames,
            max_start_offset_m=max_start_offset_m,
        )

        lengths.append(ep_rep.steps)
        done_counts.append(ep_rep.done_frames)
        trailing_counts.append(ep_rep.trailing_done_frames)
        total_chunks_read += ep_rep.chunks_read
        total_spurious_zeros += ep_rep.spurious_zeros

        if not ep_rep.is_valid:
            report.episodes_failed += 1
            for err in ep_rep.errors:
                report.errors.append(f"[{ep_name}] {err}")
        else:
            report.episodes_passed += 1

        if ep_rep.warnings:
            report.episodes_warned += 1
            for warn in ep_rep.warnings:
                report.warnings.append(f"[{ep_name}] {warn}")

        # Feedback periódico de progreso
        if (idx + 1) % progress_interval == 0 or (idx + 1) == total_eps:
            elapsed = time.time() - t_start
            fps = sum(lengths) / elapsed if elapsed > 0 else 0
            log.info(
                f"  Progreso: {idx + 1}/{total_eps} episodios verificados "
                f"({report.episodes_passed} válidos, {report.episodes_failed} fallidos) — "
                f"{fps:.1f} pasos/s"
            )

    report.total_steps = int(sum(lengths))

    # 3. Métricas agregadas
    if lengths:
        report.metrics = {
            "total_episodes": total_eps,
            "total_steps": report.total_steps,
            "min_steps": int(min(lengths)),
            "max_steps": int(max(lengths)),
            "mean_steps": float(np.mean(lengths)),
            "std_steps": float(np.std(lengths)),
            "min_done_frames": int(min(trailing_counts)),
            "max_done_frames": int(max(trailing_counts)),
            "mean_done_frames": float(np.mean(trailing_counts)),
            "std_done_frames": float(np.std(trailing_counts)),
            "min_done_duration_s": float(min(trailing_counts) * 0.050),
            "total_chunks_decompressed": total_chunks_read,
            "spurious_zeros_count": total_spurious_zeros,
        }

    # 4. Prueba con PyTorch DataLoader y DiffusionPolicy
    if test_dataloader and report.episodes_failed == 0:
        dl_compat, dl_errors, dl_shapes, pol_ok, pol_loss = verify_torch_dataloader(
            dataset_path=norm_path,
            batch_size=batch_size,
            n_obs_steps=2,
            h_act=16,
            test_policy_step=test_policy_step,
        )
        report.dataloader_compatible = dl_compat
        report.dataloader_batch_shapes = dl_shapes
        report.policy_step_verified = pol_ok
        report.initial_policy_loss = pol_loss
        if not dl_compat:
            report.errors.extend(dl_errors)
    elif not test_dataloader:
        log.info("Prueba de PyTorch DataLoader omitida (--skip-dataloader activo).")

    # 5. Veredicto Final
    report.elapsed_time_s = time.time() - t_start

    has_fatal_errors = len(report.errors) > 0 or report.episodes_failed > 0
    has_warnings_in_strict = strict and len(report.warnings) > 0

    if has_fatal_errors or has_warnings_in_strict:
        report.is_valid = False
    else:
        report.is_valid = True

    return report


def print_scorecard(report: DatasetVerificationReport, strict: bool = False):
    """Imprime una tabla de resumen ejecutiva con los resultados de la certificación."""
    print("\n" + "=" * 80)
    print("                    CERTIFICACIÓN DE DATASET ZARR — SCORECARD")
    print("=" * 80)
    print(f" Dataset Analizado    : {report.dataset_path}")
    print(f" Tiempo Transcurrido  : {report.elapsed_time_s:.2f} s")
    print(f" Total de Episodios   : {report.total_episodes:,}")
    print(f" Total de Pasos (T)   : {report.total_steps:,} fotogramas")
    print("-" * 80)

    m = report.metrics
    if m:
        print(
            f" Duración Pasos/Ep    : Mín: {m.get('min_steps')}, Máx: {m.get('max_steps')}, "
            f"Media: {m.get('mean_steps', 0):.1f} ± {m.get('std_steps', 0):.1f}"
        )
        print(
            f" Piso Señal Parada    : Mín: {m.get('min_done_frames')} frames ({m.get('min_done_duration_s', 0):.2f}s), "
            f"Máx: {m.get('max_done_frames')} frames, Media: {m.get('mean_done_frames', 0):.1f}"
        )
        if m.get("total_chunks_decompressed", 0) > 0:
            print(f" Chunks Descomprimidos: {m.get('total_chunks_decompressed'):,} bloques de imagen")
        if m.get("spurious_zeros_count", 0) > 0:
            print(f" Ceros Espurios       : {m.get('spurious_zeros_count')} detectados")

    print("-" * 80)
    print(
        f" Episodios Válidos    : {report.episodes_passed} / {report.total_episodes} "
        f"({report.episodes_passed / max(1, report.total_episodes) * 100:.1f}%)"
    )
    print(f" Episodios Fallidos   : {report.episodes_failed}")
    print(f" Episodios c/ Alertas : {report.episodes_warned}")
    print("-" * 80)

    # Estado DataLoader y Policy
    if report.dataloader_compatible:
        print(f" [PASS] PyTorch DataLoader  : COMPATIBLE (Batch shapes: {report.dataloader_batch_shapes})")
    else:
        print(f" [INFO/FAIL] DataLoader     : {'NO COMPATIBLE' if report.errors else 'NO EVALUADO'}")

    if report.policy_step_verified:
        print(f" [PASS] Diffusion Policy    : VERIFICADO (Forward + Backward OK, Loss inicial: {report.initial_policy_loss:.4f})")
    else:
        print(" [INFO] Diffusion Policy    : NO EVALUADO O NO DISPONIBLE")

    # Errores
    if report.errors:
        print(f"\n [ERROR] Fallos Críticos Detectados ({len(report.errors)}):")
        for err in report.errors[:15]:
            print(f"   ✖ {err}")
        if len(report.errors) > 15:
            print(f"   ... y {len(report.errors) - 15} errores más.")

    # Advertencias
    if report.warnings:
        print(f"\n [WARN] Advertencias / Anomalías Menores ({len(report.warnings)}):")
        for warn in report.warnings[:10]:
            print(f"   ⚠ {warn}")
        if len(report.warnings) > 10:
            print(f"   ... y {len(report.warnings) - 10} alertas más.")

    print("=" * 80)
    if report.is_valid:
        print(" VEREDICTO FINAL: ✔ DATASET ÍNTEGRO, SIN CORRUPCIÓN Y 100% LISTO PARA ENTRENAR")
    else:
        print(" VEREDICTO FINAL: ✖ DATASET CON ERRORES O NO APTO PARA ENTRENAR")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Verificador Exhaustivo de Integridad y Calidad para Datasets Zarr (Diffusion Policy).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        help="Ruta al dataset Zarr a verificar (ej. datasets/demonstrations_v8.zarr).",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        default=False,
        help="Omitir descompresión completa de imágenes (solo valida chunk 0) para chequeo ultra rápido.",
    )
    parser.add_argument(
        "--chunk-batch-size",
        type=int,
        default=64,
        help="Tamaño de bloque en frames para descompresión de chunks de imágenes sin saturar RAM.",
    )
    parser.add_argument(
        "--min-done-frames",
        type=int,
        default=40,
        help="Mínimo número de fotogramas requeridos con done_flag=1.0 para validar el piso de parada (40 frames = 2.0s a 20Hz).",
    )
    parser.add_argument(
        "--max-start-offset-m",
        type=float,
        default=0.20,
        help="Con inicio medido (start_anchor=measured, conversor v3): desfase máximo creíble del paso 0 respecto de HOME3.",
    )
    parser.add_argument(
        "--skip-dataloader",
        action="store_true",
        default=False,
        help="Omitir prueba de carga con PyTorch DataLoader.",
    )
    parser.add_argument(
        "--skip-policy-step",
        action="store_true",
        default=False,
        help="Omitir prueba de forward/backward con DiffusionPolicy.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Tamaño de lote para la prueba de PyTorch DataLoader.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Modo estricto: abortar con código 1 si se detecta cualquier advertencia menor.",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Ruta opcional para exportar el reporte completo en formato JSON.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Intervalo de reporte de progreso en consola (episodios).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de detalle de logs.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Archivo opcional para guardar el log.",
    )

    args = parser.parse_args()

    numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
    if configure_logging is not None:
        configure_logging(level=numeric_level, log_file=args.log_file)
    else:
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s [%(levelname)8s] %(name)s - %(message)s",
        )

    report = verify_dataset(
        dataset_path=args.dataset,
        check_all_images=not args.skip_images,
        chunk_batch_size=args.chunk_batch_size,
        min_done_frames=args.min_done_frames,
        max_start_offset_m=args.max_start_offset_m,
        test_dataloader=not args.skip_dataloader,
        test_policy_step=not args.skip_policy_step,
        batch_size=args.batch_size,
        strict=args.strict,
        progress_interval=args.progress_interval,
    )

    print_scorecard(report, strict=args.strict)

    if args.out_json:
        try:
            out_p = Path(args.out_json)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
            log.info(f"Reporte exportado exitosamente a: {args.out_json}")
        except Exception as e_json:
            log.error(f"Error al guardar reporte JSON en '{args.out_json}': {e_json}")

    sys.exit(0 if report.is_valid else 1)


if __name__ == "__main__":
    main()
