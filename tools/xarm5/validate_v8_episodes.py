#!/usr/bin/env python3
"""Validación offline de las demos v8: poses de DROID-SLAM desde el bag crudo, comparadas con lo grabado.

Por episodio (con su trayectoria de DROID en ``experiments/v8_validation/<dataset>/droid/<episodio>.tum``, que
produce ``tools/xarm5/validate_v8_batch.sh``):
* DROID completo: una pose por imagen del bag;
* saltos imposibles para una mano: pasos entre frames consecutivos > ``--jump-mm`` (a 30 Hz, 20 mm = 0,6 m/s). Se
  contrastan con el IMU: si el acelerómetro no muestra ese movimiento (desviación < ``--jump-acc``), es un error de
  DROID (2026-11-25, episode_71: 6 cm en 0,13 s mientras la cámara giraba sobre el mantel blanco);
* coincidencia con la odometría en vivo en los frames que sí rastreaba (alineación rígida, sin escala);
* tramo final (done) quieto según DROID (velocidad suavizada en 0,5 s);
* recorrido y distancia inicio-fin;
* pose real de inicio (``estimate_start_poses.py --zarr``, si se pasa ``--start-poses``).
Veredicto: REPETIR (salto no físico, DROID incompleto o sin pose de inicio), REVISAR (desacuerdo grande o final que
se mueve) u OK. Solo lectura del dataset.

    python3 tools/xarm5/validate_v8_episodes.py list --first 0 --last 120        # para el lote: "episodio bag"
    python3 tools/xarm5/validate_v8_episodes.py report --start-poses experiments/v8_validation/<ds>/start_poses.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def episode_keys(g):
    return sorted([k for k in g.group_keys() if k.startswith("episode_")], key=lambda s: int(s.split("_")[1]))


def kabsch_residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Residuo por punto tras alinear ``a`` sobre ``b`` con una transformación rígida (sin escala)."""
    ca, cb = a.mean(0), b.mean(0)
    u, _s, vt = np.linalg.svd((b - cb).T @ (a - ca))
    r = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
    return np.linalg.norm((a - ca) @ r.T + cb - b, axis=1)


def smoothed_speed(t: np.ndarray, p: np.ndarray, window_s: float = 0.5) -> np.ndarray:
    out = np.zeros(len(t))
    for i in range(len(t)):
        lo, hi = np.searchsorted(t, t[i] - window_s / 2), np.searchsorted(t, t[i] + window_s / 2) - 1
        hi = min(max(hi, i), len(t) - 1)
        if hi > lo:
            out[i] = np.linalg.norm(p[hi] - p[lo]) / (t[hi] - t[lo])
    return out


def jumps_vs_imu(t: np.ndarray, p: np.ndarray, imu_csv: Path, jump_m: float, acc_thr: float):
    """Pasos > ``jump_m`` entre poses consecutivas y, para cada uno, si el IMU lo respalda."""
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    idx = np.nonzero(step > jump_m)[0]
    if len(idx) == 0:
        return [], float(step.max()) if len(step) else 0.0
    imu = np.genfromtxt(imu_csv, delimiter=",", names=True) if imu_csv.is_file() else None
    out = []
    for i in idx:
        tj = 0.5 * (t[i] + t[i + 1])
        physical = None
        if imu is not None:
            ti = imu["t"]
            acc = np.c_[imu["ax"], imu["ay"], imu["az"]]
            ctx = (ti > tj - 0.5) & (ti < tj + 0.5)
            near = (ti > tj - 0.1) & (ti < tj + 0.1)
            if ctx.sum() > 20 and near.sum() > 5:
                dev = np.linalg.norm(acc[near] - acc[ctx].mean(0), axis=1).max()
                physical = bool(dev >= acc_thr)
        out.append({"t_s": float(tj - t[0]), "step_mm": float(1000 * step[i]), "imu_supports": physical})
    return out, float(step.max())


def cmd_list(a) -> int:
    import zarr
    g = zarr.open_group(str(REPO / a.dataset), mode="r")
    for k in episode_keys(g):
        n = int(k.split("_")[1])
        if a.first <= n <= a.last:
            at = dict(g[k].attrs)
            if not at.get("discarded") and at.get("raw_bag"):
                print(k, at["raw_bag"])
    return 0


def cmd_report(a) -> int:
    import zarr
    ds = Path(a.dataset).stem
    vdir = REPO / "experiments/v8_validation" / ds
    g = zarr.open_group(str(REPO / a.dataset), mode="r")
    starts = {}
    if a.start_poses and Path(a.start_poses).is_file():
        starts = json.loads(Path(a.start_poses).read_text()).get("episodes", {})
    rows = []
    for k in episode_keys(g):
        tum_path = vdir / "droid" / f"{k}.tum"
        at = dict(g[k].attrs)
        if at.get("discarded") or not tum_path.is_file():
            continue
        e = g[k]
        s = at.get("schedule") or {}
        ts, st, lost = np.asarray(e["timestamp"]), np.asarray(e["state"]), np.asarray(e["lost"])
        done = np.asarray(e["action"])[:, 8] > 0.5
        tum = np.loadtxt(tum_path)
        t_d, p_d = tum[:, 0], tum[:, 1:4]
        frames_dir = REPO / "datasets/vio_frames" / ds / k
        n_img = sum(1 for _ in open(frames_dir / "rgb.txt")) if (frames_dir / "rgb.txt").is_file() else len(t_d)
        # Inicialización de DROID (primeros frames, cámara quieta en ARMING): se mide aparte. Los saltos cuentan solo
        # dentro de la demo, pasada esa inicialización (2026-11-25: episode_1 oscilaba 55 mm en los primeros 0,45 s).
        t_init = max(float(ts[0]), float(t_d[0]) + a.init_s)
        early = t_d <= max(float(ts[0]) + 0.2, float(t_d[0]) + a.init_s)
        init_spread = (float(np.linalg.norm(p_d[early] - np.median(p_d[early], axis=0), axis=1).max())
                       if early.sum() > 2 else 0.0)
        gyro_early = float("nan")
        if (frames_dir / "imu.csv").is_file() and early.any():
            imu = np.genfromtxt(frames_dir / "imu.csv", delimiter=",", names=True)
            m = imu["t"] <= t_d[early][-1]
            if m.any():
                gyro_early = float(np.linalg.norm(np.c_[imu["wx"][m], imu["wy"][m], imu["wz"][m]], axis=1).max())
        win = t_d >= t_init
        jumps, max_step = jumps_vs_imu(t_d[win], p_d[win], frames_dir / "imu.csv", a.jump_mm / 1000.0, a.jump_acc)
        bad_jumps = [j for j in jumps if j["imu_supports"] is not True]
        inside = (ts >= t_d[0]) & (ts <= t_d[-1])
        P = np.stack([np.interp(ts[inside], t_d, p_d[:, c]) for c in range(3)], 1)
        ok = (~lost)[inside]
        res = kabsch_residual(P[ok], st[inside][ok, :3]) if ok.sum() >= 10 else np.array([np.nan])
        v = smoothed_speed(ts[inside], P)
        v_done = v[done[inside]] if done[inside].any() else np.array([np.nan])
        sp = starts.get(k, {})
        row = {"episode": k, "index": s.get("index"), "cell": s.get("cell"), "yaw_deg": s.get("yaw_deg"),
               "frames": int(len(ts)), "lost_live_pct": round(100 * float(lost.mean()), 1),
               "droid_poses": int(len(t_d)), "images": int(n_img), "max_step_mm": round(1000 * max_step, 1),
               "init_spread_mm": round(1000 * init_spread, 1), "gyro_early_max_rad_s": round(gyro_early, 3),
               "jumps": len(jumps), "nonphysical_jumps": len(bad_jumps),
               "first_bad_jump": (f"{bad_jumps[0]['t_s']:.1f}s/{bad_jumps[0]['step_mm']:.0f}mm" if bad_jumps else ""),
               "live_vs_droid_p50_mm": round(1000 * float(np.nanmedian(res)), 1),
               "live_vs_droid_p90_mm": round(1000 * float(np.nanpercentile(res, 90)), 1),
               "path_m": round(float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))), 3),
               "start_end_m": round(float(np.linalg.norm(P[-1] - P[0])), 3),
               "max_speed_cm_s": round(100 * float(v.max()), 1),
               "done_speed_p90_cm_s": round(100 * float(np.nanpercentile(v_done, 90)), 1),
               "start_usable": sp.get("usable"), "start_offset_mm": sp.get("offset_from_home_mm"),
               "start_offset_deg": sp.get("offset_from_home_deg")}
        why = []
        if row["droid_poses"] < 0.98 * row["images"]:
            why.append("DROID incompleto")
        if bad_jumps:
            why.append(f"salto no físico ({row['first_bad_jump']})")
        if starts and not sp.get("usable"):
            why.append("sin pose de inicio")
        rev = []
        if row["init_spread_mm"] > a.max_init_spread_mm:
            rev.append(f"inicio inestable de DROID ({row['init_spread_mm']:.0f} mm; "
                       f"giro {row['gyro_early_max_rad_s']:.2f} rad/s)")
        if row["live_vs_droid_p90_mm"] > a.max_disagreement_mm:
            rev.append(f"desacuerdo con la odometría en vivo (p90 {row['live_vs_droid_p90_mm']:.0f} mm)")
        if row["done_speed_p90_cm_s"] > a.max_done_speed_cm_s:
            rev.append(f"final en movimiento ({row['done_speed_p90_cm_s']:.1f} cm/s)")
        row["verdict"] = "REPETIR" if why else ("REVISAR" if rev else "OK")
        row["reasons"] = "; ".join(why + rev)
        rows.append(row)
    if not rows:
        print("sin episodios con DROID todavía")
        return 1
    out = vdir / "report.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{'episodio':11s} {'#':>4} {'c':>2} {'yaw':>5} {'perd%':>6} {'ini':>5} {'salto':>7} {'p90':>5} {'fin':>5} "
          f"{'inicio':>8}  veredicto")
    for r in rows:
        st_txt = "-" if r["start_usable"] is None else (f"{r['start_offset_mm']:.0f}mm" if r["start_usable"] else "NO")
        print(f"{r['episode']:11s} {str(r['index']):>4} {str(r['cell']):>2} {r['yaw_deg'] or 0:+5.0f} "
              f"{r['lost_live_pct']:6.1f} {r['init_spread_mm']:5.0f} {r['max_step_mm']:7.1f} {r['live_vs_droid_p90_mm']:5.0f} "
              f"{r['done_speed_p90_cm_s']:5.1f} {st_txt:>8}  {r['verdict']} {r['reasons']}")
    n = {v: sum(r["verdict"] == v for r in rows) for v in ("OK", "REVISAR", "REPETIR")}
    print(f"\n{len(rows)} validadas: {n['OK']} OK, {n['REVISAR']} a revisar, {n['REPETIR']} a repetir -> {out}")
    (vdir / "summary.json").write_text(json.dumps({"validated": len(rows), **n,
                                                   "repeat": [r["episode"] for r in rows if r["verdict"] == "REPETIR"],
                                                   "review": [r["episode"] for r in rows if r["verdict"] == "REVISAR"]},
                                                  indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("list", "report"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", default="datasets/demonstrations_real_v8.zarr")
        if name == "list":
            p.add_argument("--first", type=int, default=0)
            p.add_argument("--last", type=int, default=10**6)
        else:
            p.add_argument("--start-poses", default="")
            p.add_argument("--jump-mm", type=float, default=20.0, help="paso entre poses de DROID (30 Hz) sospechoso")
            p.add_argument("--jump-acc", type=float, default=2.0, help="m/s²: el IMU respalda un salto real")
            p.add_argument("--max-disagreement-mm", type=float, default=60.0)
            p.add_argument("--max-done-speed-cm-s", type=float, default=3.0)
            p.add_argument("--init-s", type=float, default=0.5, help="inicialización de DROID desde el inicio del bag")
            p.add_argument("--max-init-spread-mm", type=float, default=15.0)
    a = ap.parse_args(argv)
    return cmd_list(a) if a.cmd == "list" else cmd_report(a)


if __name__ == "__main__":
    sys.exit(main())
