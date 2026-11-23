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

Dos medidas de éxito (no hay posición real de las tijeras en el marco del robot):

* ``success_over_tool`` (principal: llegó y se detuvo sobre la herramienta; es el criterio del usuario en los
  ensayos del 2026-11-23). Las dos condiciones deben cumplirse:
  1. Parada neuronal sin vigilante, sin tiempo máximo y sin errores.
  2. El centro de la imagen final cae dentro de la caja de las tijeras (naranja + partes oscuras).
* ``success_auto`` (estricta: terminó como en las demos). Sus tres condiciones van abajo.

La altura final se compara con las demos que terminaron en el mismo lugar de la mesa (6 vecinas en x-y), solo como
dato informativo. NO es un criterio: en las demos, la altura de parada sobre un mismo lugar varía ±7 cm (por ejemplo,
entre 0,188 y 0,340 m), así que la mediana de las vecinas no define un "punto de aproximación".

Éxito automático ESTRICTO (``success_auto``): las tres condiciones deben cumplirse.
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


def tool_under_center(img_bgr: np.ndarray) -> bool:
    """True si el centro de la imagen cae dentro de la caja de la herramienta (naranja + partes oscuras, la mayor
    componente conexa) en la vista final a resolución completa."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    orange = ((hsv[..., 0] < 20) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 110) & (hsv[..., 2] > 60)
    tool = cv2.morphologyEx((orange | (hsv[..., 2] < 90)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(tool)
    if n < 2:
        return False
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = stats[k, :4]
    cy, cx = img_bgr.shape[0] / 2, img_bgr.shape[1] / 2
    return bool(x <= cx <= x + w and y <= cy <= y + h)


def demo_reference(cache: Path) -> dict:
    if cache.exists():
        ref = json.loads(cache.read_text())
        if "end_xyz" in ref:
            return ref
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
        "end_xyz_mean": ends.mean(0).tolist(), "end_xyz_std": ends.std(0).tolist(), "end_xyz": ends.tolist(),
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
    # Protocolo: cámara sobre la herramienta + altura a <= 5 cm del final de las demos en el mismo lugar de la mesa
    over = tool_under_center(fin) if fin is not None else None
    out["camera_over_tool"] = over
    ends = np.array(ref["end_xyz"])
    end = np.array(out["tcp_end"]) if out.get("tcp_end") else None
    if end is not None:
        d = np.linalg.norm(ends[:, :2] - end[:2], axis=1)
        idx = np.argsort(d)[:6]
        z_ref = float(np.median(ends[idx, 2]))
        out["approach_height_ref_m"] = round(z_ref, 3)
        out["approach_height_error_m"] = round(float(end[2] - z_ref), 3)
        out["demo_height_range_m"] = [round(float(ends[idx, 2].min()), 3), round(float(ends[idx, 2].max()), 3)]
        out["nearest_demo_end_xy_mm"] = round(float(d[idx].max() * 1000), 0)
    out.pop("success_protocol", None)
    out["success_over_tool"] = bool(out["autostop"] and bool(over)
                                    and not any("ERROR" in w for w in (out["warnings"] or [])))
    out["success_auto"] = bool(out["autostop"] and view_ok and not any("ERROR" in w for w in (out["warnings"] or [])))
    out["success_user"] = None
    return out


def write_summary(session: Path, rows: list, ref: dict):
    flat_keys = ["trial", "label", "success_over_tool", "success_auto", "success_user", "autostop", "autostop_s",
                 "camera_over_tool", "demo_height_range_m", "approach_height_ref_m", "reason", "final_view_ok",
                 "final_view_z", "final_scale_vs_demo_median", "tcp_end", "path_mm", "z_min", "speed_p95_mm_s",
                 "done_max"]
    with open(session / "trials.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(flat_keys)
        for r in rows:
            w.writerow([json.dumps(r.get(k)) if isinstance(r.get(k), (list, dict)) else r.get(k) for k in flat_keys])
    n = len(rows)
    k_proto = sum(bool(r.get("success_over_tool")) for r in rows)
    lo_p, hi_p = wilson(k_proto, n)
    k_auto = sum(bool(r["success_auto"]) for r in rows)
    judged = [r for r in rows if r.get("success_user") is not None]
    k_user = sum(bool(r["success_user"]) for r in judged)
    lo, hi = wilson(k_auto, n)
    times = [r["autostop_s"] for r in rows if r.get("autostop_s") is not None]
    md = ["# Ensayos de la política en el xArm5 real", "",
          f"Modelo: época 200 de `train_real_v7x5v2_relative_ee_e300` (`deploy/jetson/model`), 5 pasos DDIM + CUDA "
          f"Graph, puente a 5 cm/s. Referencia: últimos frames de {ref['n_episodes']} demos.", "",
          f"* **Llegó y se detuvo sobre la herramienta (parada automática + cámara sobre las tijeras): "
          f"{k_proto}/{n} = {100 * k_proto / max(n, 1):.0f} %** (IC Wilson 95 %: {100 * lo_p:.0f}–{100 * hi_p:.0f} %).",
          f"* **Terminó como en las demos (criterio estricto de vista final: encuadre y distancia): {k_auto}/{n} = "
          f"{100 * k_auto / max(n, 1):.0f} %** (IC 95 %: {100 * lo:.0f}–{100 * hi:.0f} %).",
          "* La altura frente a las demos del mismo lugar es solo informativa: en las demos la altura de parada sobre un "
          "mismo lugar varía ±7 cm.",
          f"* Confirmada por el usuario: {k_user}/{len(judged)}." if judged else "* Sin confirmación del usuario todavía.",
          f"* Parada automática: {sum(bool(r['autostop']) for r in rows)}/{n}; tiempo hasta la parada: mediana "
          f"{np.median(times):.1f} s (demos: {ref['duration_s_median']:.1f} s de grabación)." if times else
          "* Sin paradas automáticas.", "",
          "| Intento | Posición | Sobre la herramienta | Como en las demos | Usuario | Parada (s) | "
          "Altura final / rango de las demos vecinas (m) | Vista final (z u, z v; escala) | Final TCP (m) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        user = {True: "sí", False: "no"}.get(r.get("success_user"), "—")
        rng = r.get("demo_height_range_m")
        zf = (r.get("tcp_end") or [None, None, None])[2]
        md.append(f"| {r['trial']} | {r['label']} | {'sí' if r.get('success_over_tool') else 'no'} | "
                  f"{'sí' if r['success_auto'] else 'no'} | {user} | "
                  f"{r['autostop_s'] if r['autostop_s'] is None else round(r['autostop_s'], 1)} | "
                  f"{zf} / {rng} | {r.get('final_view_z')}; {r.get('final_scale_vs_demo_median')} | {r.get('tcp_end')} |")
    (session / "summary.md").write_text("\n".join(md) + "\n")
    return k_auto, n, lo, hi


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--session", required=True)
    ap.add_argument("--trial", type=int, help="número del intento a evaluar (carpeta trial_NNN)")
    ap.add_argument("--label", default="", help="posición de las tijeras (p. ej. centro, izquierda-lejos)")
    ap.add_argument("--max-z", type=float, default=2.5)
    ap.add_argument("--set-user", nargs=2, metavar=("TRIAL", "OK"), help="juicio del usuario: 3 true")
    ap.add_argument("--reevaluate", action="store_true", help="recalcular todos los intentos (conserva etiquetas y juicio)")
    args = ap.parse_args(argv)
    session = Path(args.session)
    ref = demo_reference(session / "demo_reference.json")
    rows_path = session / "trials.jsonl"
    rows = [json.loads(l) for l in rows_path.read_text().splitlines()] if rows_path.exists() else []
    if args.reevaluate:
        args_trials = [int(r["trial"].split("_")[1]) for r in rows]
    else:
        args_trials = [args.trial] if args.trial is not None else []
    for num in args_trials:
        name = f"trial_{num:03d}"
        prev = next((r for r in rows if r["trial"] == name), None)
        label = args.label or (prev or {}).get("label", "")
        row = evaluate(session / name, ref, args.max_z, label)
        if prev is not None:
            row["success_user"] = prev.get("success_user")          # se conserva el juicio del usuario
        rows = [r for r in rows if r["trial"] != name] + [row]
        rows.sort(key=lambda r: r["trial"])
        print(json.dumps({k: row.get(k) for k in ("trial", "label", "success_over_tool", "success_auto",
                                                  "camera_over_tool", "demo_height_range_m",
                                                  "final_view_z", "final_scale_vs_demo_median")}, ensure_ascii=False))
    if args.set_user:
        name = f"trial_{int(args.set_user[0]):03d}"
        for r in rows:
            if r["trial"] == name:
                r["success_user"] = args.set_user[1].lower() in ("1", "true", "si", "sí", "yes")
    rows_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""))
    k, n, lo, hi = write_summary(session, rows, ref)
    k_over = sum(bool(r.get("success_over_tool")) for r in rows)
    lo_o, hi_o = wilson(k_over, n)
    print(f"RESUMEN: sobre la herramienta {k_over}/{n} ({100 * k_over / max(n, 1):.0f} %, IC95 {100 * lo_o:.0f}-"
          f"{100 * hi_o:.0f} %) | como en las demos {k}/{n} ({100 * k / max(n, 1):.0f} %, IC95 {100 * lo:.0f}-{100 * hi:.0f} %)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
