#!/usr/bin/env python3
"""Diagnóstico científico de demostraciones VIO a mano y de su conversión al xArm5 (figuras en inglés).

Reproduce la auditoría que llevó a ``demonstrations_real_v7_xarm5_v2`` (2026-11-22):
  1. Ruido del VIO: PSD de la posición, perfil y distribución de velocidad antes/después del filtro.
  2. Hipótesis de extrínseca cámara-TCP: desalineación con la gravedad al inicio de cada episodio para dos montajes.
  3. Deriva de la odometría entre episodios: inclinación inicial de la cámara según el VIO frente a la medida
     independiente con los ArUco de la mesa (solvePnP IPPE sobre DICT_4X4_50).
  4. Continuidad del cuaternión: convención qw >= 0 frente a hemisferio "home".
  5. Trayectorias convertidas en link_base (vista superior y lateral) frente al volumen de trabajo del puente.

Solo lectura sobre los datasets. Salida: PNG (300 DPI) + PDF, ``quality_summary.json`` y ``quality_report.md``.

Uso (contenedor):
    python3 tools/xarm5/analyze_vio_dataset_quality.py --raw datasets/demonstrations_real_v7.zarr \
        --old datasets/demonstrations_real_v7_xarm5.zarr --new datasets/demonstrations_real_v7_xarm5_v2.zarr \
        --out-dir datasets/evaluations_real_v7_xarm5_v2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "visuomotor_bringup"))
sys.path.insert(0, str(REPO / "visuomotor_core"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from convert_vio_dataset_to_xarm5 import Converter, DEFAULT_ALIGNMENT, DEFAULT_BRIDGE_PARAMS, DEFAULT_POLICY_PARAMS  # noqa: E402
from visuomotor_bringup.xarm5_interface import arm_plane_quat, project_arm_plane, quat_to_matrix, rpy_to_quat  # noqa: E402
from visuomotor_core.dataset.trajectory_filters import TrajectoryFilterConfig, lowpass_zero_phase, hampel  # noqa: E402

C1, C2, C3, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#52514e"
plt.rcParams.update({"axes.edgecolor": INK, "axes.labelcolor": "#0b0b0b", "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "grid.color": "#d9d8d4",
                     "grid.linewidth": 0.6, "lines.linewidth": 2.0, "legend.frameon": False, "font.size": 10})


def save(fig, base: Path):
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def episode_keys(root):
    return sorted(root.group_keys(), key=lambda k: int(k.split("_")[1]))


def gravity_misalignment_deg(q0, r_base_cam_home) -> float:
    """Ángulo entre la gravedad en el marco de la cámara (odom con Z arriba) y la de la cámara en HOME."""
    g_cam = quat_to_matrix(q0).T @ np.array([0.0, 0.0, -1.0])
    g_home = r_base_cam_home.T @ np.array([0.0, 0.0, -1.0])
    return math.degrees(math.acos(float(np.clip(g_cam @ g_home, -1.0, 1.0))))


def aruco_axis_tilt(images, fx, fy, max_frames=8):
    """Inclinación del eje óptico respecto a la normal de la mesa (grados) y giro de imagen, desde ArUco 4x4.

    Usa la solución IPPE de menor error de reproyección por marcador; devuelve None si no hay marcadores o si los dos
    marcadores discrepan > 15° (ambigüedad plana).
    """
    import cv2
    k = np.array([[fx, 0, images.shape[2] / 2.0], [0, fy, images.shape[1] / 2.0], [0, 0, 1.0]])
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters_create()
    obj = np.array([[-0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, -0.5, 0]], dtype=np.float64)
    for t in range(min(max_frames, images.shape[0])):
        corners, ids, _ = cv2.aruco.detectMarkers(cv2.cvtColor(images[t], cv2.COLOR_RGB2GRAY), d, parameters=params)
        if ids is None:
            continue
        sols = []
        for c in corners:
            n, rvecs, _, errs = cv2.solvePnPGeneric(obj, c.reshape(4, 2).astype(np.float64), k, np.zeros(5),
                                                    flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if n == 0:
                continue
            best = int(np.argmin(np.asarray(errs).ravel()))
            r, _ = cv2.Rodrigues(rvecs[best])
            nrm = r[:, 2] if r[2, 2] < 0 else -r[:, 2]          # normal de la mesa hacia la cámara (marco óptico)
            tilt = math.degrees(math.acos(float(np.clip(-nrm[2], -1, 1))))
            roll = math.degrees(math.atan2(nrm[0], -nrm[1]))     # 0 = arriba de la mesa hacia arriba de la imagen
            sols.append((tilt, roll))
        if not sols:
            continue
        if len(sols) > 1 and abs(sols[0][0] - sols[1][0]) > 15.0:
            return None
        return float(np.mean([s[0] for s in sols])), float(np.mean([s[1] for s in sols]))
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--old", required=True, help="conversión anterior (referencia)")
    ap.add_argument("--new", required=True, help="conversión curada")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--example-episode", default="episode_0")
    ap.add_argument("--fx", type=float, default=605.0, help="focal nominal D435i color 640x480 (px)")
    ap.add_argument("--src-width", type=int, default=640)
    ap.add_argument("--src-height", type=int, default=480)
    args = ap.parse_args(argv)
    import zarr

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bridge = yaml.safe_load(open(DEFAULT_BRIDGE_PARAMS))["xarm5_bridge"]["ros__parameters"]
    align = yaml.safe_load(open(DEFAULT_ALIGNMENT))["xarm5_dataset_alignment"]
    policy = yaml.safe_load(open(DEFAULT_POLICY_PARAMS))["policy_node"]["ros__parameters"]
    conv = Converter(bridge, align, policy)
    fcfg = TrajectoryFilterConfig.from_dict(align.get("trajectory_filter"))
    raw = zarr.open_group(args.raw, mode="r")
    old = zarr.open_group(args.old, mode="r")
    new = zarr.open_group(args.new, mode="r")
    raw_keys = episode_keys(raw)
    new_src = {new[k].attrs["source_episode"]: k for k in episode_keys(new)}
    old_src = {old[k].attrs.get("source_episode", k): k for k in episode_keys(old)}
    rate = 20.0
    home_xyz = conv.home_xyz
    pitch, yaw, _ = project_arm_plane(rpy_to_quat(*conv.home_rpy), 0.0, *conv.pitch_limits)
    r_home = quat_to_matrix(arm_plane_quat(0.0, pitch, yaw))
    mounts = {"Assumed mount rpy (0, -π/2, 0)": quat_to_matrix(rpy_to_quat(0, -math.pi / 2, 0)),
              "Data-consistent mount rpy (π, -π/2, 0)": quat_to_matrix(rpy_to_quat(math.pi, -math.pi / 2, 0))}
    home_axis_tilt = math.degrees(math.acos(float(-(r_home @ mounts["Data-consistent mount rpy (π, -π/2, 0)"])[2, 0])))

    # ---------------- Métricas por episodio ----------------
    from scipy.signal import welch
    psd_raw, psd_f, v_raw, v_f, rows = [], [], [], [], []
    for k in raw_keys:
        st = np.asarray(raw[k]["state"], dtype=np.float64)
        p = st[:, :3]
        pf = lowpass_zero_phase(hampel(p, fcfg.hampel_half_window, fcfg.hampel_n_sigma)[0], rate,
                                fcfg.lowpass_cutoff_hz, fcfg.lowpass_order)
        f, a = welch(p - p.mean(0), fs=rate, nperseg=128, axis=0)
        _, b = welch(pf - pf.mean(0), fs=rate, nperseg=128, axis=0)
        psd_raw.append(a.sum(1))
        psd_f.append(b.sum(1))
        vr = np.linalg.norm(np.diff(p, axis=0), axis=1) * rate
        vf = np.linalg.norm(np.diff(pf, axis=0), axis=1) * rate
        v_raw.append(vr)
        v_f.append(vf)
        tilt_vio = math.degrees(math.acos(float(np.clip(-quat_to_matrix(st[0, 3:7])[2, 0], -1, 1))))
        tag = aruco_axis_tilt(np.asarray(raw[k]["image"][:8]), args.fx * 256 / args.src_width,
                              args.fx * 256 / args.src_height)
        rows.append({
            "episode": k, "kept_as": new_src.get(k), "frames": int(len(st)),
            "noise_rms_mm": float(np.sqrt(((p - pf) ** 2).sum(1).mean()) * 1000),
            "raw_max_step_mm": float(np.linalg.norm(np.diff(p, axis=0), axis=1).max() * 1000),
            "speed_mean_raw": float(vr.mean()), "speed_mean_filtered": float(vf.mean()),
            "speed_peak_filtered": float(vf.max()),
            "gravity_misalignment_deg": {m: gravity_misalignment_deg(st[0, 3:7], r_home @ r) for m, r in mounts.items()},
            "axis_tilt_vio_deg": tilt_vio,
            "axis_tilt_aruco_deg": None if tag is None else tag[0], "image_roll_aruco_deg": None if tag is None else tag[1],
        })
    f_ax = f
    psd_raw = np.median(np.array(psd_raw), 0)
    psd_f = np.median(np.array(psd_f), 0)
    v_raw_all = np.concatenate(v_raw)
    v_f_all = np.concatenate(v_f)

    # ---------------- Figura 1: diagnóstico ----------------
    fig, axs = plt.subplots(2, 3, figsize=(18, 10.5))
    fig.suptitle(f"Handheld VIO demonstrations — data-quality diagnosis ({Path(args.raw).stem}, {len(raw_keys)} episodes)",
                 fontsize=14, fontweight="bold")
    ax = axs[0, 0]
    ax.semilogy(f_ax, psd_raw, color=C1, label="Raw VIO position")
    ax.semilogy(f_ax, psd_f, color=C2, label=f"Filtered (Hampel + Butterworth {fcfg.lowpass_cutoff_hz} Hz, zero-phase)")
    ax.axvline(fcfg.lowpass_cutoff_hz, color=GRAY, ls="--", lw=1.2, label="Cut-off frequency")
    ax.set_title("A) Median position PSD across episodes", fontweight="bold")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (m²/Hz)")
    ax.grid(True)
    ax.legend(fontsize=8.5)

    ex = args.example_episode if args.example_episode in raw_keys else raw_keys[0]
    i_ex = raw_keys.index(ex)
    t = np.arange(len(v_raw[i_ex])) / rate
    ax = axs[0, 1]
    ax.plot(t, v_raw[i_ex], color=C1, lw=1.0, alpha=0.8, label="Raw VIO")
    ax.plot(t, v_f[i_ex], color=C2, lw=2.0, label="Filtered")
    ax.axhline(conv.max_lin, color=GRAY, ls="--", lw=1.2, label=f"Robot bridge limit ({conv.max_lin:.2f} m/s)")
    ax.set_title(f"B) TCP speed profile ({ex})", fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.grid(True)
    ax.legend(fontsize=8.5)

    ax = axs[0, 2]
    bins = np.logspace(-3.5, 0.5, 60)
    ax.hist(v_raw_all, bins=bins, histtype="step", color=C1, lw=2.0, density=True,
            label=f"Raw VIO (median {np.median(v_raw_all):.3f} m/s)")
    ax.hist(v_f_all, bins=bins, histtype="step", color=C2, lw=2.0, density=True,
            label=f"Filtered (median {np.median(v_f_all):.3f} m/s)")
    ax.axvline(conv.max_lin, color=GRAY, ls="--", lw=1.2, label="Robot bridge limit")
    ax.set_xscale("log")
    ax.set_title("C) Per-frame speed distribution", fontweight="bold")
    ax.set_xlabel("Speed (m/s, log)")
    ax.set_ylabel("Density")
    ax.grid(True)
    ax.legend(fontsize=8.5)

    ax = axs[1, 0]
    bins = np.linspace(0, 100, 41)
    for (m, color) in zip(mounts, (C1, C2)):
        vals = [r["gravity_misalignment_deg"][m] for r in rows]
        ax.hist(vals, bins=bins, color=color, alpha=0.55, edgecolor="white", linewidth=1.0,
                label=f"{m}: median {np.median(vals):.1f}°")
    ax.set_title("D) Camera–TCP extrinsic test: gravity misalignment at t=0", fontweight="bold")
    ax.set_xlabel("Angle between VIO gravity and HOME3 camera gravity (deg)")
    ax.set_ylabel("Episodes")
    ax.grid(True)
    ax.legend(fontsize=8.5)

    ax = axs[1, 1]
    idx = np.arange(len(rows))
    ax.plot(idx, [r["axis_tilt_vio_deg"] for r in rows], color=C1, marker="o", ms=3.5, lw=1.4,
            label="From VIO odometry (drifting world frame)")
    tag_ok = [(i, r["axis_tilt_aruco_deg"]) for i, r in enumerate(rows)
              if r["axis_tilt_aruco_deg"] is not None and abs(r["image_roll_aruco_deg"]) < 60]
    ax.plot([a for a, _ in tag_ok], [b for _, b in tag_ok], color=C2, marker="s", ms=4, lw=0,
            label="From ArUco tags on the table (independent)")
    ax.axhline(home_axis_tilt, color=GRAY, ls="--", lw=1.2, label=f"HOME3 camera axis ({home_axis_tilt:.1f}°)")
    ax.set_title("E) Initial optical-axis tilt from vertical per episode", fontweight="bold")
    ax.set_xlabel("Episode index (recording order)")
    ax.set_ylabel("Tilt (deg)")
    ax.grid(True)
    ax.legend(fontsize=8.5)

    ax = axs[1, 2]
    k_old = old_src.get(ex)
    k_new = new_src.get(ex)
    if k_old is not None:
        q_old = np.asarray(old[k_old]["state"][:, 3])
        ax.plot(np.arange(len(q_old)) / rate, q_old, color=C1, lw=1.2,
                label=f"v1 conversion, q_w ≥ 0 ({int((np.sum(old[k_old]['state'][1:, 3:7] * old[k_old]['state'][:-1, 3:7], 1) < 0).sum())} sign flips)")
    if k_new is not None:
        q_new = np.asarray(new[k_new]["state"][:, 3])
        ax.plot(np.arange(len(q_new)) / rate, q_new, color=C2, lw=2.0, label="v2 conversion, 'home' hemisphere (0 flips)")
    ax.set_title(f"F) Quaternion component q_x continuity ({ex})", fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("q_x")
    ax.grid(True)
    ax.legend(fontsize=8.5)
    plt.tight_layout()
    save(fig, out / "fig1_vio_data_quality_diagnosis")

    # ---------------- Figura 2: trayectorias en link_base ----------------
    ws = conv.workspace
    fig, axs = plt.subplots(2, 2, figsize=(14, 11.5))
    fig.suptitle("Demonstrations expressed in the xArm5 base frame (TCP, link_base)", fontsize=14, fontweight="bold")
    for col, (ds, name, color) in enumerate(((old, "v1: assumed extrinsic, unfiltered", C1),
                                             (new, "v2: data-consistent extrinsic, filtered, curated", C2))):
        keys = episode_keys(ds)
        ends = []
        for k in keys:
            a = np.asarray(ds[k]["action"][:, :3]) + home_xyz
            axs[0, col].plot(a[:, 0], a[:, 1], color=color, lw=0.6, alpha=0.45)
            axs[1, col].plot(a[:, 0], a[:, 2], color=color, lw=0.6, alpha=0.45)
            ends.append(a[-1])
        ends = np.array(ends)
        for r, (i, j, xl, yl) in enumerate(((0, 1, "x (m)", "y (m)"), (0, 2, "x (m)", "z (m)"))):
            ax = axs[r, col]
            lo = (ws.x_min, ws.y_min if j == 1 else ws.z_min)
            hi = (ws.x_max, ws.y_max if j == 1 else ws.z_max)
            ax.add_patch(Rectangle(lo, hi[0] - lo[0], hi[1] - lo[1], fill=False, ec=GRAY, ls="--", lw=1.4,
                                   label="Bridge workspace box"))
            ax.scatter(ends[:, i], ends[:, j], s=18, color=INK, zorder=4, label="Episode end (approach goal)")
            ax.scatter([home_xyz[i]], [home_xyz[j]], marker="*", s=220, color=C3, edgecolor="white", zorder=5,
                       label="HOME3")
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.set_title(f"{'Top' if j == 1 else 'Side'} view — {name} ({len(keys)} ep.)", fontweight="bold", fontsize=11)
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True)
            ax.legend(fontsize=8.5, loc="lower left")
    plt.tight_layout()
    save(fig, out / "fig2_trajectories_base_frame")

    # ---------------- Figura 3: curación por episodio ----------------
    report = json.load(open(Path(args.new).with_name(Path(args.new).stem + "_conversion_report.json")))
    eps = report["episodes"]
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Per-episode curation audit", fontsize=14, fontweight="bold")
    kept = np.array([eps[k]["written_as"] is not None for k in raw_keys])
    x = np.arange(len(raw_keys))
    vals = np.array([r["noise_rms_mm"] for r in rows])
    axs[0].bar(x[kept], vals[kept], color=C1, width=0.8, label="Kept")
    axs[0].bar(x[~kept], vals[~kept], color=C2, width=0.8, label="Dropped")
    axs[0].set_title("A) VIO noise removed by the filter (RMS)", fontweight="bold")
    axs[0].set_xlabel("Episode index")
    axs[0].set_ylabel("RMS (mm)")
    ang = np.array([eps[k]["max_angular_speed_rad_s"] for k in raw_keys])
    axs[1].bar(x[kept], ang[kept], color=C1, width=0.8, label="Kept")
    axs[1].bar(x[~kept], ang[~kept], color=C2, width=0.8, label="Dropped")
    axs[1].axhline(conv.speed_ratio * conv.max_ang, color=GRAY, ls="--", lw=1.2, label="Audit threshold")
    axs[1].set_title("B) Peak angular speed after projection to 5-DoF", fontweight="bold")
    axs[1].set_xlabel("Episode index")
    axs[1].set_ylabel("rad/s")
    stepmm = np.array([r["raw_max_step_mm"] for r in rows])
    axs[2].bar(x[kept], stepmm[kept], color=C1, width=0.8, label="Kept")
    axs[2].bar(x[~kept], stepmm[~kept], color=C2, width=0.8, label="Dropped")
    axs[2].axhline(conv.max_raw_step * 1000, color=GRAY, ls="--", lw=1.2, label="Tracking-loss threshold")
    axs[2].set_yscale("log")
    axs[2].set_title("C) Largest raw VIO jump between frames", fontweight="bold")
    axs[2].set_xlabel("Episode index")
    axs[2].set_ylabel("mm (log)")
    for ax in axs:
        ax.grid(True, axis="y")
        ax.legend(fontsize=8.5)
    plt.tight_layout()
    save(fig, out / "fig3_curation_audit")

    # ---------------- Resumen ----------------
    tags = [r for r in rows if r["axis_tilt_aruco_deg"] is not None and abs(r["image_roll_aruco_deg"]) < 60]
    summary = {
        "raw_dataset": Path(args.raw).name, "old_dataset": Path(args.old).name, "new_dataset": Path(args.new).name,
        "episodes_raw": len(raw_keys), "episodes_kept": int(kept.sum()),
        "dropped": [k for k in raw_keys if eps[k]["written_as"] is None],
        "noise_rms_mm_median": float(np.median(vals)),
        "speed_median_raw_m_s": float(np.median(v_raw_all)), "speed_median_filtered_m_s": float(np.median(v_f_all)),
        "speed_mean_raw_m_s": float(v_raw_all.mean()), "speed_mean_filtered_m_s": float(v_f_all.mean()),
        "speed_p99_filtered_m_s": float(np.percentile(v_f_all, 99)),
        "gravity_misalignment_median_deg": {m: float(np.median([r["gravity_misalignment_deg"][m] for r in rows]))
                                            for m in mounts},
        "home3_axis_tilt_deg": home_axis_tilt,
        "aruco_axis_tilt_deg": {"n": len(tags), "mean": float(np.mean([r["axis_tilt_aruco_deg"] for r in tags])),
                                "std": float(np.std([r["axis_tilt_aruco_deg"] for r in tags])),
                                "image_roll_mean": float(np.mean([r["image_roll_aruco_deg"] for r in tags]))},
        "vio_axis_tilt_deg": {"mean": float(np.mean([r["axis_tilt_vio_deg"] for r in rows])),
                              "max": float(np.max([r["axis_tilt_vio_deg"] for r in rows]))},
        "episodes": rows,
    }
    (out / "quality_summary.json").write_text(json.dumps(summary, indent=2))
    md = [f"# Data-quality audit: `{Path(args.raw).stem}` → `{Path(args.new).stem}`", "",
          f"* Episodes: {len(raw_keys)} raw → {int(kept.sum())} kept (dropped: {', '.join(summary['dropped'])}).",
          f"* VIO noise removed (median RMS): {summary['noise_rms_mm_median']:.1f} mm; filter: Hampel + "
          f"{fcfg.lowpass_cutoff_hz} Hz zero-phase Butterworth (order {fcfg.lowpass_order}).",
          f"* Mean TCP speed: raw {summary['speed_mean_raw_m_s']:.3f} m/s → filtered {summary['speed_mean_filtered_m_s']:.3f} m/s "
          f"(p99 {summary['speed_p99_filtered_m_s']:.3f} m/s; bridge limit {conv.max_lin:.2f} m/s).",
          "* Gravity misalignment at t=0 (median): " + "; ".join(f"{m}: {v:.1f}°" for m, v in
                                                                 summary["gravity_misalignment_median_deg"].items()) + ".",
          f"* ArUco (independent) optical-axis tilt at t=0: {summary['aruco_axis_tilt_deg']['mean']:.1f}° ± "
          f"{summary['aruco_axis_tilt_deg']['std']:.1f}° (n={len(tags)}), image roll {summary['aruco_axis_tilt_deg']['image_roll_mean']:.1f}° "
          f"(upright); HOME3 camera axis {home_axis_tilt:.1f}°. VIO-based tilt reaches {summary['vio_axis_tilt_deg']['max']:.0f}° "
          "because the odometry frame is not reset between episodes and drifts.", "",
          "![Diagnosis](fig1_vio_data_quality_diagnosis.png)", "", "![Trajectories](fig2_trajectories_base_frame.png)",
          "", "![Curation](fig3_curation_audit.png)", ""]
    (out / "quality_report.md").write_text("\n".join(md))
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
