#!/usr/bin/env python3
"""Dataset v8 con las poses de DROID-SLAM (offline) en lugar de las de la odometría en vivo.

El recolector guarda, por demo, el episodio Zarr (imágenes de 256 px, pose de la odometría en vivo, done) y el bag
crudo. La odometría en vivo solo guía: se pierde y salta cerca de la mesa. Las poses del dataset salen de DROID-SLAM
RGB-D, el mejor estimador medido contra el robot (``.agents/knowledge/18_vio_estimator_selection_droid_slam.md``),
corrido sobre el bag crudo por ``tools/xarm5/validate_v8_batch.sh``.

Por episodio:
1. **Marco:** DROID da el sistema óptico relativo a su primer frame. Se pasa a ``camera_link`` con ``T_link_opt``
   (calibración exportada del bag), que es lo que guarda el recolector.
2. **Referencia de inicio robusta:** la MEDIANA de las poses de DROID en el tramo quieto del inicio (pasada su
   inicialización y antes de que la mano se mueva; ver ``build_episode``), no el primer frame: DROID puede oscilar
   al inicializarse (episode_1: 42 mm en 0,45 s). Solo los frames de esa inicialización se fijan a la referencia.
3. **Escala:** las traslaciones se multiplican por ``--depth-scale`` (0,986: sesgo de escala de la profundidad de la
   D435i medido con el robot, validado dejando una corrida fuera).
4. **Interpolación:** a los sellos de los frames del episodio (posición lineal, rotación SLERP).
5. **Done:** el tramo final quieto según DROID (``detect_hold_start``) si dura al menos ``--min-hold-s``; si no, se
   conserva el del recolector (queda registrado). En los dos casos se recorta su comienzo para que no incluya frames
   a más de ``--done-radius-m`` (3 cm) de la pose final (versión 3, ``trim_done_to_radius``).
6. **Escritura:** episodio nuevo, con el mismo formato que el recolector (``state`` y ``action`` con coordenadas
   relativas). Imágenes, sellos y el resto se copian tal cual; los atributos se conservan y se agrega la
   trazabilidad. NUNCA sobrescribe: ``--dst`` no debe existir.

Se omiten los episodios descartados y los que el informe de validación marca ``--skip-verdicts`` (por defecto
REPETIR). Si a una demo NO descartada le falta la trayectoria de DROID, se detiene sin escribir nada (2026-11-26, a
pedido del usuario: el dataset que se entrena lleva SIEMPRE las poses de DROID; ``--allow-missing-droid`` la omite).
Cada episodio guarda ``raw_dataset`` y ``raw_episode`` (su demo en el crudo), que el conversor copia al dataset final.

Normalmente lo corre ``tools/xarm5/make_v8_training_set.sh`` (armado, conversión al xArm5 con la pose real de inicio,
verificación y reemplazo del dataset de entrenamiento). A mano:
    python3 tools/xarm5/build_dataset_from_bags.py --src datasets/demonstrations_real_v8.zarr --dst <nuevo>.zarr
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "visuomotor_data_collection"))

from visuomotor_data_collection.vo_episode_logic import detect_hold_start  # noqa: E402

BUILD_VERSION = 3   # 2: referencia en la ventana quieta medida (no la del giróscopo, que borraba movimiento)
                    # 3: el done nunca incluye frames a más de --done-radius-m de la pose final (2026-11-27)


def episode_keys(g):
    return sorted([k for k in g.group_keys() if k.startswith("episode_")], key=lambda s: int(s.split("_")[1]))


def load_tum(path: Path):
    tum = np.loadtxt(path, ndmin=2)
    return tum[:, 0], tum[:, 1:4], tum[:, 4:8]                    # t, xyz, quat xyzw


def trim_done_to_radius(done, xyz, radius_m: float, n_final: int):
    """(Versión 3, 2026-11-27) El tramo done (llegada) nunca incluye frames a más de ``radius_m`` de la pose final
    (media de los últimos ``n_final`` frames): solo RECORTA el comienzo del bloque done final, nunca lo alarga.

    Motivo: en 44 de 230 demos el TCP se movía más de 3 cm dentro del done (peor caso 30 cm), sobre todo cuando la
    llegada venía del recolector (aproximación lenta tomada por quietud). Con esas etiquetas la red aprende a decir
    "llegué" antes de llegar. Devuelve (done corregido, frames recortados)."""
    done = np.asarray(done, dtype=bool).copy()
    xyz = np.asarray(xyz, dtype=np.float64)
    if not done.any() or radius_m <= 0 or len(xyz) == 0:
        return done, 0
    i_done = len(done)                                     # comienzo del bloque done contiguo al final
    while i_done > 0 and done[i_done - 1]:
        i_done -= 1
    d = np.linalg.norm(xyz - xyz[-max(1, n_final):].mean(0), axis=1)
    far = np.where(d > radius_m)[0]
    i_new = max(i_done, int(far[-1]) + 1 if len(far) else 0)
    out = np.zeros(len(done), dtype=bool)
    out[i_new:] = True
    return out, int(i_new - i_done)


def build_episode(ts, t_d, p_d, q_d, T_link_opt, depth_scale: float, init_s: float, ref_end_after_start_s: float):
    """Poses de ``camera_link`` relativas a la referencia de inicio, en los sellos ``ts``. Devuelve
    (xyz [N,3], quat xyzw [N,4], info).

    Referencia = mediana de las poses de DROID entre ``t_bag + init_s`` (pasada su inicialización) y
    ``ts[0] + ref_end_after_start_s``. Medido en las 114 demos del 2026-11-25: el ARMING dura ~0,35 s; la mano empieza
    a moverse al menos 0,53 s después del primer frame grabado (mediana 1,07 s), y DROID ya está estable a los 0,5 s
    del bag (dispersión p95 2,5 mm). Es el mismo tramo quieto en el que ``estimate_start_poses.py`` mide la pose real
    de inicio (ARMING).
    Solo se fijan a la referencia los frames de la demo que caen en la inicialización de DROID (antes de
    ``t_bag + init_s``, la cámara quieta por protocolo). La versión 1 usaba una ventana "quieta" según el giróscopo,
    pero la mano puede trasladarse sin girar: borraba hasta 17 cm de movimiento real al inicio (episode_77)."""
    from scipy.spatial.transform import Rotation, Slerp
    R_opt = Rotation.from_quat(q_d)
    T_opt_link = np.linalg.inv(T_link_opt)
    R_ol, t_ol = Rotation.from_matrix(T_opt_link[:3, :3]), T_opt_link[:3, 3]
    # T_W_link = T_W_opt · T_opt_link
    R_link = R_opt * R_ol
    p_link = p_d + R_opt.apply(t_ol)
    t_init = float(t_d[0]) + init_s
    ref_sel = (t_d >= t_init) & (t_d <= float(ts[0]) + ref_end_after_start_s)
    method = "median_still_window"
    if ref_sel.sum() < 3:                                          # ventana corta: los 5 primeros frames estables
        ref_sel = (t_d >= t_init) & (np.cumsum(t_d >= t_init) <= 5)
        method = "first_stable_frames"
    p_ref = np.median(p_link[ref_sel], axis=0)
    R_ref = R_link[ref_sel].mean()
    spread = float(np.linalg.norm(p_link[ref_sel] - p_ref, axis=1).max())
    # Relativas a la referencia (marco de camera_link al inicio) y con la escala corregida
    R_rel = R_ref.inv() * R_link
    p_rel = depth_scale * R_ref.inv().apply(p_link - p_ref)
    # Interpolación a los sellos del episodio (fuera del rango de DROID: el extremo más cercano)
    tq = np.clip(ts, t_d[0], t_d[-1])
    xyz = np.stack([np.interp(tq, t_d, p_rel[:, c]) for c in range(3)], axis=1)
    rot = Slerp(t_d, R_rel)(tq)
    pin = ts < t_init                                              # inicialización de DROID: cámara quieta
    xyz[pin] = 0.0
    q = rot.as_quat()
    q[pin] = [0.0, 0.0, 0.0, 1.0]
    for i in range(1, len(q)):                                     # signo continuo del cuaternión
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    info = {"reference": method, "reference_frames": int(ref_sel.sum()),
            "reference_window_s": [round(init_s, 3), round(float(ts[0] - t_d[0] + ref_end_after_start_s), 3)],
            "reference_spread_mm": round(1000 * spread, 1), "frames_pinned_to_start": int(pin.sum())}
    return xyz, q, info


def main(argv=None) -> int:
    import zarr
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="dataset crudo del recolector (datasets/<nombre>.zarr)")
    ap.add_argument("--dst", required=True, help="dataset nuevo (no debe existir)")
    ap.add_argument("--droid-dir", default="", help="defecto: experiments/v8_validation/<src>/droid")
    ap.add_argument("--frames-dir", default="", help="defecto: datasets/vio_frames/<src>")
    ap.add_argument("--report", default="", help="report.csv de validate_v8_episodes.py (defecto: el de <src>)")
    ap.add_argument("--skip-verdicts", default="REPETIR", help="veredictos que no entran (coma); vacío = ninguno")
    ap.add_argument("--depth-scale", type=float, default=0.986)
    ap.add_argument("--init-s", type=float, default=0.5, help="inicialización de DROID que no se usa de referencia")
    ap.add_argument("--ref-end-after-start-s", type=float, default=0.4,
                    help="fin de la ventana de referencia, desde el primer frame grabado (la mano se mueve >= 0,53 s)")
    ap.add_argument("--min-hold-s", type=float, default=2.0)
    ap.add_argument("--static-speed", type=float, default=0.02)
    ap.add_argument("--static-window-s", type=float, default=0.5)
    ap.add_argument("--done-radius-m", type=float, default=0.03,
                    help="el done no incluye frames a más de esto de la pose final (0 = sin recorte)")
    ap.add_argument("--done-final-window-s", type=float, default=0.5, help="ventana final que define la pose final")
    ap.add_argument("--episodes", default="", help="solo estos (coma), p. ej. episode_41,episode_55")
    ap.add_argument("--allow-missing-droid", action="store_true",
                    help="omitir (en vez de detenerse) las demos no descartadas sin trayectoria de DROID")
    a = ap.parse_args(argv)

    src = REPO / a.src if not Path(a.src).is_absolute() else Path(a.src)
    dst = REPO / a.dst if not Path(a.dst).is_absolute() else Path(a.dst)
    if dst.exists():
        print(f"ERROR: {dst} ya existe (nunca se sobrescribe un dataset)")
        return 1
    ds = src.stem
    droid_dir = Path(a.droid_dir) if a.droid_dir else REPO / "experiments/v8_validation" / ds / "droid"
    frames_dir = Path(a.frames_dir) if a.frames_dir else REPO / "datasets/vio_frames" / ds
    report = Path(a.report) if a.report else REPO / "experiments/v8_validation" / ds / "report.csv"
    verdicts = {}
    if report.is_file():
        with report.open() as f:
            verdicts = {r["episode"]: r for r in csv.DictReader(f)}
    skip_v = {v.strip() for v in a.skip_verdicts.split(",") if v.strip()}
    only = {e.strip() for e in a.episodes.split(",") if e.strip()}

    g = zarr.open_group(str(src), mode="r")
    # Poses de DROID obligatorias: una demo válida sin trayectoria se corrige ANTES (DROID o descarte), no se pierde
    missing = [k for k in episode_keys(g) if (not only or k in only) and not g[k].attrs.get("discarded")
               and not (droid_dir / f"{k}.tum").is_file()]
    if missing and not a.allow_missing_droid:
        print(f"ERROR: {len(missing)} demos no descartadas sin trayectoria de DROID: {', '.join(missing)}\n"
              "  Correr DROID (tools/xarm5/validate_v8_batch.sh) o marcarlas descartadas (con el OK del usuario). "
              "--allow-missing-droid las omite.")
        return 2
    out = zarr.open_group(str(dst), mode="w-")
    summary = {"built": [], "skipped": {}}
    for k in episode_keys(g):
        if only and k not in only:
            continue
        e = g[k]
        at = dict(e.attrs)
        tum = droid_dir / f"{k}.tum"
        v = verdicts.get(k, {}).get("verdict")
        why = ("descartada" if at.get("discarded") else "sin trayectoria de DROID" if not tum.is_file()
               else f"validación: {v}" if v in skip_v else None)
        if why:
            summary["skipped"][k] = why
            continue
        ts = np.asarray(e["timestamp"], dtype=np.float64)
        st, act = np.asarray(e["state"]), np.asarray(e["action"])
        rate = float(at.get("sampling_rate_hz", 20.0))
        t_d, p_d, q_d = load_tum(tum)
        calib = json.loads((frames_dir / k / "calib.json").read_text())
        T_link_opt = np.asarray(calib["T_link_opt"], dtype=np.float64)
        xyz, q, info = build_episode(ts, t_d, p_d, q_d, T_link_opt, a.depth_scale, a.init_s, a.ref_end_after_start_s)
        # Done: tramo final quieto según DROID; si es corto, el del recolector
        done_old = act[:, 8] > 0.5 if act.shape[1] > 8 else np.zeros(len(ts), bool)
        i_hold = detect_hold_start(xyz, rate, a.static_speed, a.static_window_s)
        n_min = int(round(a.min_hold_s * rate))
        if i_hold is not None and len(ts) - i_hold >= n_min:
            done = np.zeros(len(ts), bool)
            done[i_hold:] = True
            done_src = "droid"
        else:
            done, done_src = done_old, "collector"
        done, n_cut = trim_done_to_radius(done, xyz, a.done_radius_m, int(round(a.done_final_window_s * rate)))
        state = np.zeros_like(st, dtype=np.float32)
        state[:, 0:3], state[:, 3:7], state[:, 7] = xyz, q, st[:, 7]
        action = np.zeros((len(ts), act.shape[1]), dtype=np.float32)
        action[:, 0:8] = state
        if act.shape[1] > 8:
            action[:, 8] = done.astype(np.float32)
        ep = out.create_group(k)
        for name in e.array_keys():
            if name in ("state", "action"):
                arr = state if name == "state" else action
                ep.create_dataset(name, data=arr, chunks=e[name].chunks, compressor=e[name].compressor)
            else:
                zarr.copy(e[name], ep, name=name)
        new_at = dict(at)
        new_at.update({"raw_dataset": str(src.relative_to(REPO)) if src.is_relative_to(REPO) else str(src), "raw_episode": k})
        new_at.update({"pose_source": "droid_rgbd", "build_version": BUILD_VERSION, "droid_tum": str(tum.relative_to(REPO)),
                       "depth_scale": a.depth_scale, "start_reference": info, "done_source_build": done_src,
                       "done_frames": int(done.sum()), "done_frames_collector": int(done_old.sum()),
                       "done_radius_m": a.done_radius_m, "done_trimmed_frames": n_cut,
                       "validation_verdict": v, "validation_reasons": verdicts.get(k, {}).get("reasons", ""),
                       "relative_coordinates": True, "pose_frame": "camera_link relativa a la referencia de inicio"})
        ep.attrs.update(new_at)
        summary["built"].append(k)
        print(f"{k}: {len(ts)} frames | referencia {info['reference_frames']} frames (dispersión "
              f"{info['reference_spread_mm']} mm, {info['frames_pinned_to_start']} fijados) | done {done_src} "
              f"{int(done.sum())} (recolector {int(done_old.sum())}) | recorrido final {np.linalg.norm(xyz[-1]):.3f} m"
              + (f" | validación {v}" if v else ""))
    out.attrs.update({"source": str(src.relative_to(REPO)) if src.is_relative_to(REPO) else str(src),
                      "builder": "tools/xarm5/build_dataset_from_bags.py", "build_version": BUILD_VERSION,
                      "built_at": datetime.now().isoformat(timespec="seconds"), "pose_source": "droid_rgbd",
                      "depth_scale": a.depth_scale, "skip_verdicts": sorted(skip_v), "episodes": len(summary["built"]),
                      "skipped": summary["skipped"]})
    print(f"\n{len(summary['built'])} episodios -> {dst} | omitidos {len(summary['skipped'])}: "
          + ", ".join(f"{k} ({w})" for k, w in summary["skipped"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
