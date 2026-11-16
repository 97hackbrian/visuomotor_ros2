#!/usr/bin/env python3
# Copyright (c) 2026. Programacion Cientifica, DISC, Antofagasta, Chile.
"""
merge_zarr_batches.py — Herramienta de Fusión y Aplanado de Lotes Zarr (Off-ROS)
================================================================================
Consolida múltiples datasets temporales de demostraciones generados en paralelo
(ej. shards _batch_1.zarr, _batch_2.zarr, ...) en un único dataset destino canónico
(ej. demonstrations_v8.zarr).

Preserva estrictamente:
- Nombres de grupos: episode_0, episode_1, ..., episode_N (secuencia contigua continua).
- Estructura interna por episodio: 'image' (uint8), 'state' (float32, 8D), 'action' (float32, 9D).
- Chunking original por paso (1, H, W, C), compresores y metadatos de atributos.
- Si el destino ya contiene episodios (ej. 179 episodios), detecta el último índice
  (episode_178) y anexa a partir de episode_179 en adelante sin sobrescribir nada.
- Verificación exhaustiva post-fusión y opción segura --remove-sources.
- Actualización opcional no intrusiva en base de datos SQLite (experiments.db).
"""

import argparse
import glob
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import zarr
    import numpy as np
except ImportError:
    print(
        "[ERROR FATAL] 'zarr' y 'numpy' son requeridos para ejecutar merge_zarr_batches.py.\n"
        "Ejecútalo dentro del contenedor Docker 'visuomotor_ros2' o instala zarr en el entorno.",
        file=sys.stderr,
    )
    sys.exit(1)

# Asegurar importación de visuomotor_core si no está en el entorno global
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, ".."))
CORE_PKG_DIR = os.path.join(REPO_ROOT_DIR, "visuomotor_core")
if CORE_PKG_DIR not in sys.path:
    sys.path.insert(0, CORE_PKG_DIR)

# Importar logger científico off-ROS si está disponible
try:
    from visuomotor_core.utils.logger import configure_logging
except ImportError:
    configure_logging = None

log = logging.getLogger("merge_zarr_batches")


def natural_sort_key(s: str) -> List:
    """Clave de ordenamiento natural para que 'batch_2' anteceda a 'batch_10'."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", str(s))]


def find_input_paths(input_patterns: List[str], output_path: str) -> List[str]:
    """
    Expande patrones glob y resuelve rutas reales, excluyendo el dataset destino
    y cualquier directorio ancestro del destino para prevenir auto-fusiones circulares
    o eliminaciones accidentales de carpetas raíz.
    """
    matched_paths = set()
    norm_output = os.path.realpath(os.path.abspath(output_path))

    for item in input_patterns:
        # Expandir glob si contiene comodines
        matches = glob.glob(item)
        if not matches:
            if os.path.exists(item):
                matches = [item]
            else:
                log.warning(f"Ruta o patrón no encontrado: {item}")
                continue

        for p in matches:
            abs_p = os.path.realpath(os.path.abspath(p))
            # Excluir coincidencia exacta o ancestros de output_path
            if abs_p == norm_output or norm_output.startswith(abs_p + os.sep):
                log.warning(f"Ignorando archivo o directorio de entrada que contiene/coincide con el destino: {p}")
                continue
            if os.path.isdir(p) or os.path.exists(p):
                matched_paths.add(p)

    return sorted(list(matched_paths), key=natural_sort_key)


def get_sorted_episode_groups(zarr_root) -> List[Tuple[int, str]]:
    """
    Retorna la lista de pares (indice_numerico, nombre_grupo) ordenados numéricamente.
    Ignora grupos que no cumplan el patrón '^episode_(\\d+)$'.
    """
    episodes = []
    for key in zarr_root.group_keys():
        match = re.match(r"^episode_(\d+)$", key)
        if match:
            idx = int(match.group(1))
            episodes.append((idx, key))
        else:
            log.debug(f"Ignorando grupo no-episodio: '{key}'")
    episodes.sort(key=lambda x: x[0])
    return episodes


def get_next_episode_index(dest_root) -> int:
    """
    Determina el próximo índice disponible en el dataset destino.
    Si ya existen episode_0..episode_178, retorna 179.
    Si está vacío o no contiene episodios, retorna 0.
    """
    existing_eps = get_sorted_episode_groups(dest_root)
    if not existing_eps:
        return 0
    max_idx = max(idx for idx, _ in existing_eps)
    return max_idx + 1


def validate_episode(ep_group, ep_name: str) -> Tuple[bool, str]:
    """
    Valida la consistencia estructural del episodio:
    - Existencia de 'image', 'state', 'action'.
    - Coherencia de longitud temporal T entre los 3 tensores.
    - Tipos de datos correctos: image=uint8, state=float32, action=float32.
    - Dimensiones propioceptivas: state=(T, 8), action=(T, 9).
    """
    for req in ("image", "state", "action"):
        if req not in ep_group:
            return False, f"{ep_name}: Falta el dataset requerido '{req}'"

    img_ds = ep_group["image"]
    state_ds = ep_group["state"]
    act_ds = ep_group["action"]

    t_img = img_ds.shape[0]
    t_state = state_ds.shape[0]
    t_act = act_ds.shape[0]

    if not (t_img == t_state == t_act):
        return (
            False,
            f"{ep_name}: Desajuste de pasos temporales: image={t_img}, state={t_state}, action={t_act}",
        )

    if t_img < 1:
        return False, f"{ep_name}: Episodio con 0 pasos temporales"

    if img_ds.dtype != np.uint8:
        return False, f"{ep_name}: 'image' debe ser uint8, encontrado {img_ds.dtype}"

    if state_ds.dtype != np.float32:
        return False, f"{ep_name}: 'state' debe ser float32, encontrado {state_ds.dtype}"

    if act_ds.dtype != np.float32:
        return False, f"{ep_name}: 'action' debe ser float32, encontrado {act_ds.dtype}"

    if len(state_ds.shape) != 2 or state_ds.shape[1] != 8:
        return False, f"{ep_name}: 'state' debe tener dimensiones (T, 8), encontrado {state_ds.shape}"

    if len(act_ds.shape) != 2 or act_ds.shape[1] != 9:
        return False, f"{ep_name}: 'action' debe tener dimensiones (T, 9), encontrado {act_ds.shape}"

    return True, "OK"


def merge_batches(
    input_paths: List[str],
    output_path: str,
    remove_sources: bool = False,
    verify_only: bool = False,
    strict: bool = False,
) -> bool:
    """
    Ejecuta el proceso completo de fusión, aplanado y validación de shards Zarr.
    En modo normal (strict=False), detecta y omite con advertencia episodios corruptos
    o incompletos (útil ante cortes con Ctrl+C), consolidando todos los episodios válidos.
    En modo estricto (strict=True), aborta antes de modificar el destino si algún lote está corrupto.
    """
    if not input_paths:
        log.warning("No se proporcionaron lotes de entrada para fusionar.")
        return True

    log.info("=" * 70)
    log.info(f"Iniciando consolidación de {len(input_paths)} lote(s) Zarr")
    log.info(f"Dataset destino: {output_path}")
    log.info(f"Lotes de entrada:")
    for idx, p in enumerate(input_paths, 1):
        log.info(f"  [{idx}/{len(input_paths)}] {p}")
    log.info("=" * 70)

    # 1. Inspección y validación previa de todos los lotes de entrada
    total_episodes_to_merge = 0
    batch_inventories = []
    skipped_episodes_count = 0

    for p in input_paths:
        try:
            src_root = zarr.open_group(p, mode="r")
            all_eps = get_sorted_episode_groups(src_root)
            log.info(f"Lote '{p}': {len(all_eps)} episodio(s) detectado(s).")

            valid_eps = []
            for src_idx, src_ep_name in all_eps:
                src_ep_group = src_root[src_ep_name]
                valid, reason = validate_episode(src_ep_group, f"{p}/{src_ep_name}")
                if valid:
                    valid_eps.append((src_idx, src_ep_name))
                else:
                    if strict:
                        log.error(f"[MODO ESTRICTO] Episodio corrupto o incompleto en '{p}/{src_ep_name}': {reason}")
                        return False
                    else:
                        log.warning(f"Omitiendo episodio corrupto/incompleto '{p}/{src_ep_name}': {reason}")
                        skipped_episodes_count += 1

            batch_inventories.append((p, valid_eps))
            total_episodes_to_merge += len(valid_eps)
        except Exception as e:
            log.error(f"Error al abrir el lote fuente '{p}': {e}")
            return False

    if total_episodes_to_merge == 0:
        if skipped_episodes_count > 0:
            log.warning(f"Se omitieron {skipped_episodes_count} episodios inválidos/incompletos. No quedan episodios válidos.")
        else:
            log.warning("No hay episodios en ninguno de los lotes fuente. Nada que fusionar.")
        return True

    log.info(f"Total de episodios válidos a fusionar: {total_episodes_to_merge} (omitidos: {skipped_episodes_count})")

    if verify_only:
        log.info("Modo --verify-only activo. Inspección concluida exitosamente sin escribir cambios.")
        return True

    # 2. Apertura del dataset destino
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    dest_root = zarr.open_group(output_path, mode="a")

    initial_existing_eps = get_sorted_episode_groups(dest_root)
    initial_count = len(initial_existing_eps)
    next_idx = get_next_episode_index(dest_root)

    if initial_count > 0:
        log.info(
            f"Dataset destino preexistente detectado: {initial_count} episodios. "
            f"Último episodio: 'episode_{next_idx - 1}'. Anexando desde 'episode_{next_idx}'."
        )
    else:
        log.info(f"Dataset destino nuevo/vacío. Comenzando desde 'episode_0'.")

    # 3. Transferencia secuencial de cada episodio válido
    copied_episodes = 0

    for src_path, valid_eps in batch_inventories:
        if not valid_eps:
            continue

        src_root = zarr.open_group(src_path, mode="r")
        log.info(f"Procesando lote: {src_path} ({len(valid_eps)} episodios válidos)...")

        for src_idx, src_ep_name in valid_eps:
            src_ep_group = src_root[src_ep_name]
            dest_ep_name = f"episode_{next_idx}"
            dest_ep_group = dest_root.create_group(dest_ep_name)

            # Copiar datasets del episodio
            for name, src_ds in src_ep_group.arrays():
                # Preservar chunks, compressor y atributos exactos
                dest_ds = dest_ep_group.create_dataset(
                    name,
                    data=src_ds[:],
                    chunks=src_ds.chunks,
                    dtype=src_ds.dtype,
                    compressor=src_ds.compressor,
                )
                if hasattr(src_ds, "attrs") and len(src_ds.attrs) > 0:
                    dest_ds.attrs.update(dict(src_ds.attrs))

            # Copiar atributos del grupo de episodio si existieran
            if hasattr(src_ep_group, "attrs") and len(src_ep_group.attrs) > 0:
                dest_ep_group.attrs.update(dict(src_ep_group.attrs))

            # Validar episodio destino recién escrito
            valid_dest, reason_dest = validate_episode(dest_ep_group, f"{output_path}/{dest_ep_name}")
            if not valid_dest:
                log.error(f"Fallo de verificación en destino tras escribir {dest_ep_name}: {reason_dest}")
                return False

            next_idx += 1
            copied_episodes += 1

            if copied_episodes % 50 == 0 or copied_episodes == total_episodes_to_merge:
                log.info(
                    f"  Progreso: {copied_episodes}/{total_episodes_to_merge} episodios copiados "
                    f"(actual: {dest_ep_name})."
                )

    # 4. Verificación exhaustiva post-fusión
    log.info("Verificando consistencia final del dataset destino...")
    dest_root_check = zarr.open_group(output_path, mode="r")
    final_eps = get_sorted_episode_groups(dest_root_check)
    expected_total = initial_count + copied_episodes

    if len(final_eps) != expected_total:
        log.error(
            f"Inconsistencia en conteo final: se esperaban {expected_total} episodios, "
            f"pero se encontraron {len(final_eps)}."
        )
        return False

    log.info(
        f"✔ Consolidación exitosa: {copied_episodes} episodios añadidos. "
        f"Total actual en '{output_path}': {len(final_eps)} episodios."
    )

    # 5. Registro Relacional SQLite Fail-Safe
    try:
        from visuomotor_core.utils.tracker import record_dataset
        dataset_name = os.path.basename(output_path)
        record_dataset(
            dataset_id=dataset_name,
            path_str=str(output_path),
            demonstrator_type="PID_PILOT_PARALLEL_MERGE",
            total_episodes=len(final_eps),
            duration_s=0.0,
            has_done_flag=True,
            metadata={
                "source_batches_count": len(input_paths),
                "merged_episodes": copied_episodes,
                "previous_episodes": initial_count,
            },
        )
        log.info(f"✔ Registro relacional SQLite actualizado en experiments.db para '{dataset_name}'.")
    except Exception as e_db:
        log.debug(f"[DB Fail-Safe] Error no crítico al registrar en SQLite: {e_db}")

    # 6. Limpieza segura de lotes fuente si fue solicitada
    if remove_sources:
        log.info("Removiendo lotes fuente temporales (--remove-sources)...")
        for src_path in input_paths:
            try:
                if os.path.isdir(src_path):
                    shutil.rmtree(src_path)
                    log.info(f"  ✔ Removido: {src_path}")
                elif os.path.isfile(src_path):
                    os.remove(src_path)
                    log.info(f"  ✔ Removido: {src_path}")
            except Exception as e_rm:
                log.warning(f"  Aviso al eliminar lote fuente '{src_path}': {e_rm}")

    log.info("=" * 70)
    log.info(f"Fusión completada con éxito. Dataset consolidado en: {output_path}")
    log.info("=" * 70)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Fusión y Aplanado Automático de Lotes Zarr para Recolección Paralela.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--inputs",
        nargs="+",
        required=True,
        help="Rutas a lotes/shards Zarr o patrones glob (ej. datasets/*_batch_*.zarr).",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Ruta al dataset Zarr consolidado destino (ej. datasets/demonstrations_v8.zarr).",
    )
    parser.add_argument(
        "--remove-sources",
        action="store_true",
        default=False,
        help="Eliminar de forma segura los lotes temporales una vez verificada la fusión exitosa.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        default=False,
        help="Solo inspeccionar e informar la consistencia de los lotes sin escribir en el destino.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Modo estricto: abortar inmediatamente si algún lote contiene episodios corruptos o incompletos.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de verbosidad de logs.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Archivo opcional para guardar el log de la operación.",
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

    resolved_inputs = find_input_paths(args.inputs, args.output)
    if not resolved_inputs:
        log.error(f"No se encontraron datasets válidos con los patrones: {args.inputs}")
        sys.exit(1)

    success = merge_batches(
        input_paths=resolved_inputs,
        output_path=args.output,
        remove_sources=args.remove_sources,
        verify_only=args.verify_only,
        strict=args.strict,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
