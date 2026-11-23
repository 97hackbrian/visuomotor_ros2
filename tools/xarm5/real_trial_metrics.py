#!/usr/bin/env python3
"""Métricas de los ensayos de la política en el xArm5 real (aproximación a las tijeras desde HOME3).

Cada intento deja una carpeta ``trial_NNN/`` con:
- ``pre.png``: vista en HOME3 antes de empezar;
- ``final.png``: vista tras la parada;
- ``monitor.json``: resumen del vigilante;
- ``tcp.npy``: traza del TCP en link_base;
- ``policy.log``.

Este script calcula sus métricas y actualiza el resumen acumulado de la sesión: ``trials.jsonl``, ``trials.csv`` y
``summary.md``.

Éxito automático (``success_auto``): las tres condiciones deben cumplirse.
1. La política se detuvo SOLA (parada neuronal), sin vigilante, sin tiempo máximo y sin errores.
2. En la vista final las tijeras aparecen donde aparecían en las demostraciones al llegar: el centroide (u, v) a
   ≤ ``--max-z`` desviaciones estándar de la media de los últimos frames de las 76 demos.
3. El tamaño aparente de las tijeras (que equivale a la distancia) queda entre los percentiles 5 y 95 de esas demos.
El juicio del usuario (``success_user``) se puede añadir con ``--set-user TRIAL true|false``.

Las tijeras se segmentan por color (naranja: H < 20 y S > 110 en HSV de OpenCV), en la imagen que ve la política
(640x480 a 256x256 con INTER_AREA).

Uso (contenedor del PC):
    python3 tools/xarm5/real_trial_metrics.py --session experiments/xarm5_real_trials/2026-11-23 --trial 2 --label centro
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "datasets" / "demonstrations_real_v7_xarm5_v2.zarr"


def to_policy_view(img_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (256, 256):
        rgb = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
    return rgb


def scissors_stats(rgb256: np.ndarray):
    """Centroide normalizado (u, v) y fracción de área de los píxeles naranjas (mangos de las tijeras)."""
    hsv = cv2.cvtColor(rgb256, cv2.COLOR_RGB2HSV)
    mask = ((hsv[..., 0] < 20) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 110) & (hsv[..., 2] > 60)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    n = int(mask.sum())
    if n < 30:
        return None
    ys, xs = np.nonzero(mask)
    return {"u": float(xs.mean() / 256), "v": float(ys.mean() / 256), "area": float(n / mask.size)}


def demo_reference(cache: Path) -> dict:
    if cache.exists():
        return json.loads(cache.read_text())
    import zarr
    root = zarr.open_group(str(DATASET), mode="r")
    home = np.array([0.15009, 0.0, 0.75])
    first, last, ends, lens = [], [], [], []
    for k in root.group_keys():
        ep = root[k]
        f = scissors_stats(np.asarray(ep["image"][0]))
        l = scissors_stats(np.asarray(ep["image"][-1]))
        if f:
            first.append([f["u"], f["v"], f["area"]])
        if l:
            last.append([l["u"], l["v"], l["area"]])
        a = np.asarray(ep["action"])
        ends.append((a[-1, :3] + home).tolist())
        lens.append(len(a) / 20.0)
    first, last, ends = np.array(first), np.array(last), np.array(ends)
    ref = {
        "n_episodes": len(ends),
        "last_u_mean": float(last[:, 0].mean()), "last_u_std": float(last[:, 0].std()),
        "last_v_mean": float(last[:, 1].mean()), "last_v_std": float(last[:, 1].std()),
        "last_area_p05": float(np.percentile(last[:, 2], 5)), "last_area_p95": float(np.percentile(last[:, 2], 95)),
        "last_area_median": float(np.median(last[:, 2])),
        "first_u": first[:, 0].tolist(), "first_v": first[:, 1].tolist(),
        "end_xyz_mean": ends.mean(0).tolist(), "end_xyz_std": ends.std(0).tolist(),
        "duration_s_median": float(np.median(lens)),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(ref, indent=2))
    return ref


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def evaluate(trial_dir: Path, ref: dict, max_z: float, label: str) -> dict:
    mon = json.loads((trial_dir / "monitor.json").read_text())
    out = {"trial": trial_dir.name, "label": label, **{k: mon.get(k) for k in (
        "reason", "inference_s", "autostop_s", "tcp_start", "tcp_end", "displacement_mm", "path_mm", "z_min",
        "speed_p95_mm_s", "done_max", "warnings")}}
    out["autostop"] = mon.get("autostop_s") is not None and "parada automática" in str(mon.get("reason"))
    pre = cv2.imread(str(trial_dir / "pre.png")) if (trial_dir / "pre.png").exists() else None
    fin = cv2.imread(str(trial_dir / "final.png")) if (trial_dir / "final.png").exists() else None
    out["scissors_pre"] = scissors_stats(to_policy_view(pre)) if pre is not None else None
    out["scissors_final"] = scissors_stats(to_policy_view(fin)) if fin is not None else None
    sf = out["scissors_final"]
    if sf:
        zu = (sf["u"] - ref["last_u_mean"]) / ref["last_u_std"]
        zv = (sf["v"] - ref["last_v_mean"]) / ref["last_v_std"]
        out["final_view_z"] = [round(zu, 2), round(zv, 2)]
        out["final_scale_vs_demo_median"] = round(sf["area"] / ref["last_area_median"], 2)
        view_ok = abs(zu) <= max_z and abs(zv) <= max_z and ref["last_area_p05"] <= sf["area"] <= ref["last_area_p95"]
    else:
        view_ok = False
    out["final_view_ok"] = bool(view_ok)
    out["success_auto"] = bool(out["autostop"] and view_ok and not any("ERROR" in w for w in (out["warnings"] or [])))
    out["success_user"] = None
    return out


def write_summary(session: Path, rows: list, ref: dict):
    flat_keys = ["trial", "label", "success_auto", "success_user", "autostop", "autostop_s", "reason", "final_view_ok",
                 "final_view_z", "final_scale_vs_demo_median", "tcp_end", "path_mm", "z_min", "speed_p95_mm_s",
                 "done_max"]
    with open(session / "trials.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(flat_keys)
        for r in rows:
            w.writerow([json.dumps(r.get(k)) if isinstance(r.get(k), (list, dict)) else r.get(k) for k in flat_keys])
    n = len(rows)
    k_auto = sum(bool(r["success_auto"]) for r in rows)
    judged = [r for r in rows if r.get("success_user") is not None]
    k_user = sum(bool(r["success_user"]) for r in judged)
    lo, hi = wilson(k_auto, n)
    times = [r["autostop_s"] for r in rows if r.get("autostop_s") is not None]
    md = ["# Ensayos de la política en el xArm5 real", "",
          f"Modelo: época 200 de `train_real_v7x5v2_relative_ee_e300` (`deploy/jetson/model`), 5 pasos DDIM + CUDA "
          f"Graph, puente a 5 cm/s. Referencia: últimos frames de {ref['n_episodes']} demos.", "",
          f"* **Tasa de éxito (automática): {k_auto}/{n} = {100 * k_auto / max(n, 1):.0f} %** "
          f"(IC Wilson 95 %: {100 * lo:.0f}–{100 * hi:.0f} %).",
          f"* Confirmada por el usuario: {k_user}/{len(judged)}." if judged else "* Sin confirmación del usuario todavía.",
          f"* Parada automática: {sum(bool(r['autostop']) for r in rows)}/{n}; tiempo hasta la parada: mediana "
          f"{np.median(times):.1f} s (demos: {ref['duration_s_median']:.1f} s de grabación)." if times else
          "* Sin paradas automáticas.", "",
          "| Intento | Posición | Éxito | Parada (s) | Vista final (z u, z v; escala) | Final TCP (m) | Recorrido (mm) |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['trial']} | {r['label']} | {'sí' if r['success_auto'] else 'no'} | "
                  f"{r['autostop_s'] if r['autostop_s'] is None else round(r['autostop_s'], 1)} | "
                  f"{r.get('final_view_z')}; {r.get('final_scale_vs_demo_median')} | {r.get('tcp_end')} | "
                  f"{r.get('path_mm')} |")
    (session / "summary.md").write_text("\n".join(md) + "\n")
    return k_auto, n, lo, hi


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--session", required=True)
    ap.add_argument("--trial", type=int, help="número del intento a evaluar (carpeta trial_NNN)")
    ap.add_argument("--label", default="", help="posición de las tijeras (p. ej. centro, izquierda-lejos)")
    ap.add_argument("--max-z", type=float, default=2.5)
    ap.add_argument("--set-user", nargs=2, metavar=("TRIAL", "OK"), help="juicio del usuario: 3 true")
    args = ap.parse_args(argv)
    session = Path(args.session)
    ref = demo_reference(session / "demo_reference.json")
    rows_path = session / "trials.jsonl"
    rows = [json.loads(l) for l in rows_path.read_text().splitlines()] if rows_path.exists() else []
    if args.trial is not None:
        name = f"trial_{args.trial:03d}"
        row = evaluate(session / name, ref, args.max_z, args.label)
        rows = [r for r in rows if r["trial"] != name] + [row]
        rows.sort(key=lambda r: r["trial"])
        print(json.dumps(row, ensure_ascii=False))
    if args.set_user:
        name = f"trial_{int(args.set_user[0]):03d}"
        for r in rows:
            if r["trial"] == name:
                r["success_user"] = args.set_user[1].lower() in ("1", "true", "si", "sí", "yes")
    rows_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""))
    k, n, lo, hi = write_summary(session, rows, ref)
    print(f"RESUMEN: éxito automático {k}/{n} ({100 * k / max(n, 1):.0f} %, IC95 {100 * lo:.0f}-{100 * hi:.0f} %)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
