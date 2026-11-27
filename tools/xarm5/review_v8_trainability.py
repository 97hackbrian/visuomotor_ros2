#!/usr/bin/env python3
"""¿Sirve el dataset v8 para entrenar? Revisión demo por demo de lo que las métricas globales no muestran
(2026-11-27, pedido del usuario: "revisa y verifica que las métricas obtenidas son entrenables").

Con la pose REAL de la cámara al final (pose de inicio medida con ArUco + IMU, compuesta con DROID) y el dataset de
entrenamiento, por cada demo:
  1. Destino: dónde corta el eje óptico al plano de la mesa frente a la celda asignada por el programa.
  2. Altura final de la cámara sobre la mesa (protocolo: 25-30 cm sobre la herramienta).
  3. Orientación final: la dirección "arriba" de la imagen frente a la dirección de las hojas
     (protocolo: hojas hacia arriba en la imagen al final).
  4. Etiqueta de llegada (done): duración y cuánto se mueve el TCP dentro del tramo done (si se mueve, el tramo
     incluye parte de la aproximación y la red aprende a decir "llegué" antes de tiempo).
  5. Alcance por fase (tercios de la demo): dónde quedan los tramos que el robot no puede seguir.
  6. Imagen: nitidez (varianza del laplaciano) al final y en el frame más rápido, comparada con la referencia.
  7. Giro final de las demos a 180° (+177 / -177: dos grupos en el estado) y continuidad del cuaternión.
Salida (solo lectura sobre los datasets): ``trainability_review.json``, ``fig6_final_views_contact_sheet.png``
(vista final de todas las demos) y ``fig7_protocol_checks`` (PNG + PDF), en ``--out-dir``.

    python3 tools/xarm5/review_v8_trainability.py --raw datasets/demonstrations_real_v8.zarr \\
        --droid <armado con DROID>.zarr --train datasets/demonstrations_real_v8_xarm5.zarr \\
        --reference datasets/demonstrations_real_v7_xarm5_v2.zarr --out-dir datasets/evaluations_real_v8_xarm5
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_vio_dataset_quality import C1, C2, C3, GRAY, INK, episode_keys, plt, save  # noqa: E402
from convert_vio_dataset_to_xarm5 import DEFAULT_BRIDGE_PARAMS  # noqa: E402
from visuomotor_bringup.xarm5_interface import ReachLimit, arm_azimuth, arm_plane_params  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CELLS = REPO / "visuomotor_bringup/config/realsense/v8_cells.yaml"


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def pose(xyz, q):
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.asarray(q, dtype=float)).as_matrix()
    T[:3, 3] = xyz
    return T


def sharpness(img) -> float:
    import cv2
    g = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def pct(v, q):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=float)
    return float(np.percentile(v, q)) if len(v) else float("nan")


def main(argv=None) -> int:
    import zarr
    from scipy.spatial.transform import Rotation
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--droid", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--reference", default="", help="dataset de referencia para comparar la nitidez (opcional)")
    ap.add_argument("--start-poses", default="", help="defecto: experiments/v8_validation/<raw>/start_poses.json")
    ap.add_argument("--cells-config", default=str(DEFAULT_CELLS))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--final-frames", type=int, default=10, help="frames finales promediados para la vista final")
    ap.add_argument("--cell-margin-m", type=float, default=0.03, help="tolerancia del eje óptico fuera de la celda")
    ap.add_argument("--align-tol-deg", type=float, default=30.0, help="tolerancia 'hojas arriba' en la imagen")
    ap.add_argument("--done-motion-m", type=float, default=0.03, help="movimiento máximo del TCP dentro del done")
    ap.add_argument("--blur-ratio", type=float, default=0.35, help="imagen borrosa: nitidez < esto × la mediana")
    ap.add_argument("--straight-elbow-m", type=float, default=0.6406)
    a = ap.parse_args(argv)

    raw_name = Path(a.raw).stem
    start_json = Path(a.start_poses) if a.start_poses else Path("experiments/v8_validation") / raw_name / "start_poses.json"
    starts = json.load(open(start_json))
    starts = starts.get("episodes", starts)
    cells_cfg = yaml.safe_load(open(a.cells_config))["v8_cells"]
    table_z = float(cells_cfg["table_z_m"])
    check_h = [float(h) for h in cells_cfg.get("check_heights_m", [0.25, 0.30])]
    bridge = yaml.safe_load(open(DEFAULT_BRIDGE_PARAMS))["xarm5_bridge"]["ros__parameters"]
    reach = ReachLimit(bridge["workspace_shoulder_xyz_m"], bridge["workspace_wrist_offset_m"],
                       bridge["workspace_max_reach_m"], bridge["workspace_min_reach_m"])
    pitch_lim = (float(bridge["arm_plane_pitch_min"]), float(bridge["arm_plane_pitch_max"]))
    raw = zarr.open_group(a.raw, mode="r")
    droid = zarr.open_group(a.droid, mode="r")
    train = zarr.open_group(a.train, mode="r")
    home = np.asarray(dict(train.attrs)["xarm5_conversion"]["home_xyz_m"], dtype=float)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    thumbs = []
    for tk in episode_keys(train):
        at = dict(train[tk].attrs)
        rk = at.get("raw_episode") or at.get("source_episode")
        ra = dict(raw[rk].attrs)
        sch = ra.get("schedule") or {}
        rate = float(ra.get("sampling_rate_hz", 20.0))
        row = {"episode": tk, "raw_episode": rk, "day": str(Path(str(ra.get("raw_bag", ""))).name)[3:11],
               "cell": sch.get("cell"), "yaw_deg": sch.get("yaw_deg")}
        # --- 1-3: vista final con la pose real de la cámara
        sp = starts.get(rk) or {}
        if sp.get("T_base_camera_link") is not None and rk in droid:
            d = np.asarray(droid[rk]["state"][-a.final_frames:], dtype=float)
            T_rel = pose(d[:, :3].mean(0), Rotation.from_quat(d[:, 3:7]).mean().as_quat())
            T = np.asarray(sp["T_base_camera_link"], dtype=float) @ T_rel
            c, R = T[:3, 3], T[:3, :3]
            ax_dir, up = R[:, 0], R[:, 2]                     # camera_link: x = eje óptico, z = arriba de la imagen
            row["camera_height_m"] = float(c[2] - table_z)
            row["optical_tilt_deg"] = float(math.degrees(math.acos(np.clip(-ax_dir[2], -1, 1))))
            if ax_dir[2] < -0.05:
                hit = c + (table_z - c[2]) / ax_dir[2] * ax_dir
                row["axis_hit_xy"] = [float(hit[0]), float(hit[1])]
                b = sch.get("cell_bounds_m")
                if b is not None:
                    row["cell_bounds_m"] = [float(v) for v in b]
                    x0, x1, y0, y1 = row["cell_bounds_m"]
                    dx = max(x0 - hit[0], 0.0, hit[0] - x1)
                    dy = max(y0 - hit[1], 0.0, hit[1] - y1)
                    row["axis_outside_cell_m"] = float(math.hypot(dx, dy))
                    row["axis_to_cell_center_m"] = float(math.hypot(hit[0] - (x0 + x1) / 2, hit[1] - (y0 + y1) / 2))
            if sch.get("yaw_deg") is not None and np.hypot(up[0], up[1]) > 0.2:
                up_az = math.degrees(math.atan2(up[1], up[0]))
                blade_az = -float(sch["yaw_deg"])              # 0 = +x (frente), +90 = derecha (-y), 180 = hacia la base
                row["blade_alignment_err_deg"] = float(wrap_deg(up_az - blade_az))
        # --- 4: etiqueta de llegada
        st = np.asarray(train[tk]["state"][:], dtype=float)
        act = np.asarray(train[tk]["action"][:], dtype=float)
        p = st[:, :3] + home
        done = act[:, 8] > 0.5
        row["frames"] = int(len(p))
        row["done_s"] = float(done.sum() / rate)
        row["done_source"] = at.get("done_source_build")
        if done.any():
            pd_ = p[done]
            row["done_motion_m"] = float(np.linalg.norm(pd_ - p[-1], axis=1).max())
            row["done_start_frac"] = float(np.argmax(done) / len(p))
        # --- 5: alcance por tercios
        q = st[:, 3:7]
        dist = np.array([reach.reach(p[i], min(max(arm_plane_params(q[i], arm_azimuth(p[i]))[0], pitch_lim[0]),
                                                    pitch_lim[1])) for i in range(len(p))])
        beyond = dist > reach.max_reach
        thirds = np.array_split(np.arange(len(p)), 3)
        row["beyond_reach_by_third"] = [float(beyond[t].mean()) for t in thirds]
        row["beyond_reach_frac"] = float(beyond.mean())
        row["beyond_elbow_frac"] = float((dist > a.straight_elbow_m).mean())
        row["max_elbow_utilization"] = float(dist.max() / a.straight_elbow_m)
        # --- 7: giro final y continuidad del cuaternión
        row["final_yaw_deg"] = float(math.degrees(arm_plane_params(q[-1], arm_azimuth(p[-1]))[1]))
        qn = q / np.linalg.norm(q, axis=1, keepdims=True)
        dots = np.abs(np.sum(qn[1:] * qn[:-1], axis=1)).clip(0, 1)
        row["max_rot_step_deg"] = float(np.degrees(2 * np.arccos(dots)).max()) if len(dots) else 0.0
        row["sign_flips"] = int((np.sum(qn[1:] * qn[:-1], axis=1) < 0).sum())
        # --- 6: nitidez (final y frame más rápido)
        img = train[tk]["image"]
        row["sharp_final"] = float(np.mean([sharpness(img[i]) for i in range(max(0, len(p) - 5), len(p))]))
        v = np.linalg.norm(np.gradient(p, axis=0), axis=1) * rate
        i_fast = int(np.argmax(v))
        row["sharp_fastest"] = sharpness(img[i_fast])
        row["speed_fastest_m_s"] = float(v[i_fast])
        row["brightness_final"] = float(np.asarray(img[len(p) - 1]).mean())
        thumbs.append((tk, row, np.asarray(img[len(p) - 1])))
        rows.append(row)

    # --- referencia (nitidez)
    ref = {}
    if a.reference:
        rg = zarr.open_group(a.reference, mode="r")
        rh = np.asarray(dict(rg.attrs).get("xarm5_conversion", {}).get("home_xyz_m", home), dtype=float)
        sf, sx = [], []
        for k in episode_keys(rg):
            img = rg[k]["image"]
            n = len(img)
            pr = np.asarray(rg[k]["state"][:, :3], dtype=float) + rh
            v = np.linalg.norm(np.gradient(pr, axis=0), axis=1) * 20.0
            sf.append(np.mean([sharpness(img[i]) for i in range(max(0, n - 5), n)]))
            sx.append(sharpness(img[int(np.argmax(v))]))
        ref = {"dataset": Path(a.reference).name, "sharp_final_median": float(np.median(sf)),
               "sharp_fastest_median": float(np.median(sx))}

    # ---------------- resumen y veredictos por criterio ----------------
    n = len(rows)
    sched = [r for r in rows if r.get("axis_outside_cell_m") is not None]
    off_cell = [r for r in sched if r["axis_outside_cell_m"] > a.cell_margin_m]
    aligned = [r for r in rows if r.get("blade_alignment_err_deg") is not None]
    misaligned = [r for r in aligned if abs(r["blade_alignment_err_deg"]) > a.align_tol_deg]
    done_moving = [r for r in rows if r.get("done_motion_m", 0) > a.done_motion_m]
    sharp_med = float(np.median([r["sharp_final"] for r in rows]))
    blurry = [r for r in rows if r["sharp_final"] < a.blur_ratio * sharp_med]
    by_third = np.array([r["beyond_reach_by_third"] for r in rows])
    by_yaw = defaultdict(list)
    for r in aligned:
        by_yaw[r["yaw_deg"]].append(abs(r["blade_alignment_err_deg"]))
    y180 = [r for r in rows if r.get("yaw_deg") == 180.0]
    heights = [r["camera_height_m"] for r in rows if "camera_height_m" in r]
    summary = {
        "train_dataset": Path(a.train).name, "demonstrations": n,
        "destination": {"checked": len(sched), "axis_outside_cell_beyond_margin": len(off_cell),
                        "margin_m": a.cell_margin_m, "axis_to_cell_center_m_p50_p90": [pct([r["axis_to_cell_center_m"] for r in sched], 50),
                                                                                         pct([r["axis_to_cell_center_m"] for r in sched], 90)],
                        "episodes_off_cell": [(r["episode"], r["raw_episode"], r["cell"], r["yaw_deg"],
                                               round(r["axis_outside_cell_m"], 3)) for r in off_cell]},
        "final_camera_height_m": {"p5": pct(heights, 5), "p50": pct(heights, 50), "p95": pct(heights, 95),
                                  "protocol_above_tool_m": check_h},
        "blade_alignment": {"checked": len(aligned), "beyond_tol": len(misaligned), "tol_deg": a.align_tol_deg,
                            "abs_err_deg_p50_p90": [pct([abs(r["blade_alignment_err_deg"]) for r in aligned], 50),
                                                    pct([abs(r["blade_alignment_err_deg"]) for r in aligned], 90)],
                            "median_abs_err_by_yaw": {str(k): float(np.median(v)) for k, v in sorted(by_yaw.items())},
                            "episodes_misaligned": [(r["episode"], r["raw_episode"], r["yaw_deg"],
                                                     round(r["blade_alignment_err_deg"], 1)) for r in misaligned]},
        "done_label": {"duration_s_p5_p50_p95": [pct([r["done_s"] for r in rows], q) for q in (5, 50, 95)],
                       "motion_inside_done_m_p50_p90_max": [pct([r.get("done_motion_m") for r in rows], q) for q in (50, 90, 100)],
                       "episodes_done_moving_beyond": len(done_moving), "threshold_m": a.done_motion_m,
                       "done_source_counts": dict(Counter(str(r.get("done_source")) for r in rows))},
        "reach": {"beyond_bridge_reach_by_third_mean": [float(x) for x in by_third.mean(0)],
                  "episodes_with_any_beyond_elbow": int(sum(r["beyond_elbow_frac"] > 0 for r in rows)),
                  "by_day_mean_beyond_reach": {d: float(np.mean([r["beyond_reach_frac"] for r in rows if r["day"] == d]))
                                               for d in sorted({r["day"] for r in rows})}},
        "images": {"sharp_final_median": sharp_med, "sharp_fastest_median": float(np.median([r["sharp_fastest"] for r in rows])),
                   "blurry_final": len(blurry), "blur_ratio": a.blur_ratio, "reference": ref,
                   "brightness_final_p5_p95": [pct([r["brightness_final"] for r in rows], 5), pct([r["brightness_final"] for r in rows], 95)],
                   "episodes_blurry_final": [r["episode"] for r in blurry]},
        "yaw_180": {"demos": len(y180), "final_yaw_positive": int(sum(r["final_yaw_deg"] > 0 for r in y180)),
                    "final_yaw_negative": int(sum(r["final_yaw_deg"] < 0 for r in y180))},
        "quaternion": {"sign_flips_total": int(sum(r["sign_flips"] for r in rows)),
                       "max_rot_step_deg_p99": pct([r["max_rot_step_deg"] for r in rows], 99),
                       "max_rot_step_deg_max": float(max(r["max_rot_step_deg"] for r in rows))},
        "episodes": rows,
    }
    (out / "trainability_review.json").write_text(json.dumps(summary, indent=1, default=float))

    # ---------------- figura 6: vista final de todas las demos ----------------
    import cv2
    cols, tw = 23, 96
    rows_n = int(math.ceil(len(thumbs) / cols))
    sheet = np.full((rows_n * (tw + 14), cols * tw, 3), 255, dtype=np.uint8)
    for i, (tk, r, im) in enumerate(thumbs):
        y, x = (i // cols) * (tw + 14), (i % cols) * tw
        sheet[y:y + tw, x:x + tw] = cv2.resize(im, (tw, tw), interpolation=cv2.INTER_AREA)
        bad = (r in off_cell) or (r in misaligned) or (r in blurry)
        label = f"{tk.split('_')[1]} c{r['cell'] if r['cell'] is not None else '-'} {int(r['yaw_deg']) if r['yaw_deg'] is not None else ''}"
        cv2.putText(sheet, label, (x + 2, y + tw + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (200, 40, 40) if bad else (40, 40, 40), 1,
                    cv2.LINE_AA)
        if bad:
            cv2.rectangle(sheet, (x, y), (x + tw - 1, y + tw - 1), (220, 60, 30), 2)
    cv2.imwrite(str(out / "fig6_final_views_contact_sheet.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    # ---------------- figura 7: controles del protocolo ----------------
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Per-demonstration protocol and label checks ({Path(a.train).stem}, {n} demonstrations)",
                 fontsize=14, fontweight="bold")
    ax = axs[0, 0]
    hits = np.array([r["axis_hit_xy"] for r in sched])
    for c in sorted({r["cell"] for r in sched}):
        b = next(r["cell_bounds_m"] for r in sched if r["cell"] == c)
        ax.add_patch(plt.Rectangle((b[0], b[2]), b[1] - b[0], b[3] - b[2], fill=False, ec=GRAY, lw=1.0))
        ax.text((b[0] + b[1]) / 2, (b[2] + b[3]) / 2, str(c), ha="center", va="center", fontsize=12, color=GRAY)
    ok = [i for i, r in enumerate(sched) if r not in off_cell]
    ax.scatter(hits[ok, 0], hits[ok, 1], s=10, color=C1, label=f"Inside its cell (±{100 * a.cell_margin_m:.0f} cm): {len(ok)}")
    bad = [i for i, r in enumerate(sched) if r in off_cell]
    if bad:
        ax.scatter(hits[bad, 0], hits[bad, 1], s=22, color=C2, marker="x", label=f"Outside its cell: {len(bad)}")
    ax.set_title("A) Final optical axis on the table vs the assigned cell", fontweight="bold")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)
    ax.legend(fontsize=8.5, loc="lower left")
    ax = axs[0, 1]
    ax.hist(heights, bins=30, color=C1, edgecolor="white")
    for h in check_h:
        ax.axvline(h, color=INK, ls="--", lw=1.0)
    ax.set_title(f"B) Final camera height above the table (protocol {check_h[0]:.2f}–{check_h[-1]:.2f} m, dashed)",
                 fontweight="bold", fontsize=10)
    ax.set_xlabel("Height (m)")
    ax.set_ylabel("Demonstrations")
    ax.grid(True)
    ax = axs[0, 2]
    yaws = sorted(by_yaw)
    ax.boxplot([by_yaw[y] for y in yaws], labels=[f"{y:+.0f}°" for y in yaws], showfliers=True)
    ax.axhline(a.align_tol_deg, color=C2, ls="--", lw=1.0, label=f"Tolerance ({a.align_tol_deg:.0f}°)")
    ax.set_title("C) 'Blades up in the image' error at the end, by orientation", fontweight="bold", fontsize=10)
    ax.set_xlabel("Blade orientation")
    ax.set_ylabel("|image-up − blade direction| (deg)")
    ax.grid(True, axis="y")
    ax.legend(fontsize=8.5)
    ax = axs[1, 0]
    dm = [100 * r.get("done_motion_m", 0) for r in rows]
    ax.hist(dm, bins=np.linspace(0, max(10, max(dm)), 41), color=C1, edgecolor="white")
    ax.axvline(100 * a.done_motion_m, color=C2, ls="--", lw=1.0, label=f"{100 * a.done_motion_m:.0f} cm")
    ax.set_title("D) TCP motion inside the 'done' segment", fontweight="bold")
    ax.set_xlabel("Max distance from the final pose (cm)")
    ax.set_ylabel("Demonstrations")
    ax.grid(True)
    ax.legend(fontsize=8.5)
    ax = axs[1, 1]
    ax.bar(["first third", "middle third", "last third"], 100 * by_third.mean(0), color=[C2, C3, C1])
    ax.set_title("E) Frames beyond the bridge reach, by phase", fontweight="bold")
    ax.set_ylabel("% of frames (mean over demonstrations)")
    ax.grid(True, axis="y")
    ax = axs[1, 2]
    bins = np.logspace(np.log10(max(1.0, pct([r["sharp_fastest"] for r in rows], 1))),
                       np.log10(pct([r["sharp_final"] for r in rows], 100) + 1), 40)
    ax.hist([r["sharp_final"] for r in rows], bins=bins, histtype="step", lw=2, color=C1, label="Final frames (v8)")
    ax.hist([r["sharp_fastest"] for r in rows], bins=bins, histtype="step", lw=2, color=C2, label="Fastest frame (v8)")
    if ref:
        ax.axvline(ref["sharp_final_median"], color=GRAY, ls="--", lw=1.0, label=f"{ref['dataset']} final (median)")
        ax.axvline(ref["sharp_fastest_median"], color=GRAY, ls=":", lw=1.0, label=f"{ref['dataset']} fastest (median)")
    ax.set_xscale("log")
    ax.set_title("F) Image sharpness (variance of the Laplacian)", fontweight="bold")
    ax.set_xlabel("Sharpness (log)")
    ax.set_ylabel("Demonstrations")
    ax.grid(True)
    ax.legend(fontsize=8.5)
    plt.tight_layout()
    save(fig, out / "fig7_protocol_checks")
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
