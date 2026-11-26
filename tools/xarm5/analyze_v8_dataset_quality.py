#!/usr/bin/env python3
"""Auditoría de calidad del dataset v8 (demos a mano con la D435i, poses de DROID), análoga a la de v7
(``analyze_vio_dataset_quality.py``: figuras 1-3 y ``quality_report.md``). Figuras e informe en inglés (paper).

En v8 las poses del dataset NO salen de la odometría en vivo (RTAB-Map, que solo guía la toma): salen de DROID-SLAM
offline sobre el bag crudo. Donde v7 comparaba "VIO crudo" con "filtrado", aquí se comparan tres fuentes sobre las
MISMAS demos: odometría en vivo (crudo), DROID (armado, antes del filtro) y DROID filtrado (el filtro del conversor
v3, con el que sale el dataset de entrenamiento). Entradas, todas por argumento (AGENTS.md §16):
  --raw          crudo del recolector (odometría en vivo, ``lost`` por frame, programa, descartes)
  --droid        armado con las poses de DROID (``build_dataset_from_bags.py``; mismo formato que el crudo)
  --train        dataset de entrenamiento (``convert_vio_dataset_to_xarm5.py``) y su informe de conversión
  --reference    (opcional) dataset del modelo que funcionó en el robot, para comparar cobertura (figura 2)
  --report       report.csv de ``validate_v8_episodes.py`` (veredictos, pérdidas y desacuerdo en vivo vs DROID)
  --start-poses  start_poses.json de ``estimate_start_poses.py`` (pose real de inicio)
Figuras (PNG 300 DPI + PDF):
  1. Ruido, velocidad e inicio: PSD mediana y velocidades por fuente; perfil de un episodio; desacuerdo en vivo vs
     DROID; inicio respecto de HOME3 por día; inclinación inicial del eje óptico (ArUco independiente vs pose de
     inicio) por episodio.
  2. Trayectorias del TCP en link_base (vista superior y lateral): referencia frente a v8, con la caja del puente.
  3. Auditoría por demo del crudo: estado (entrenamiento / a revisar / excluida por DROID / descartada), pérdidas de
     la odometría en vivo, salto máximo de DROID y velocidad angular tras la proyección a 5 GDL.
  5. Cobertura: demos por celda × orientación de las hojas (en el dataset y factibles para el robot).
Solo lectura sobre los datasets. Salida: figuras, ``quality_summary.json`` y ``quality_report.md``.

    python3 tools/xarm5/analyze_v8_dataset_quality.py --raw datasets/demonstrations_real_v8.zarr \\
        --droid <armado con DROID>.zarr --train datasets/demonstrations_real_v8_xarm5.zarr \\
        --reference datasets/demonstrations_real_v7_xarm5_v2.zarr --out-dir datasets/evaluations_real_v8_xarm5
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_vio_dataset_quality import (C1, C2, C3, GRAY, INK, Rectangle, aruco_axis_tilt,  # noqa: E402
                                         episode_keys, fill_masked, frame_speeds, plt, save)
from convert_vio_dataset_to_xarm5 import (Converter, DEFAULT_ALIGNMENT, DEFAULT_BRIDGE_PARAMS,  # noqa: E402
                                          DEFAULT_POLICY_PARAMS)
from matplotlib.transforms import blended_transform_factory  # noqa: E402
from visuomotor_bringup.xarm5_interface import ReachLimit, arm_azimuth, arm_plane_params  # noqa: E402
from visuomotor_core.dataset.trajectory_filters import TrajectoryFilterConfig, hampel, lowpass_zero_phase  # noqa: E402

STATUS = {"train": ("In training set", C1), "revisar": ("In training set (REVISAR)", C3),
          "repetir": ("Excluded (DROID non-physical jump)", C2), "discarded": ("Discarded (operator / validation)", GRAY)}


def filtered(p, rate, fcfg):
    return lowpass_zero_phase(hampel(p, fcfg.hampel_half_window, fcfg.hampel_n_sigma)[0], rate,
                              fcfg.lowpass_cutoff_hz, fcfg.lowpass_order)


def pct(v, q):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=float)
    return float(np.percentile(v, q)) if len(v) else float("nan")


def main(argv=None) -> int:
    import zarr
    from scipy.signal import welch
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--droid", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--reference", default="")
    ap.add_argument("--report", default="", help="defecto: experiments/v8_validation/<raw>/report.csv")
    ap.add_argument("--start-poses", default="", help="defecto: experiments/v8_validation/<raw>/start_poses.json")
    ap.add_argument("--reference-summary", default="",
                    help="quality_summary.json de la auditoría de la referencia (para el informe)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--example-episode", default="", help="episodio del crudo para el perfil (defecto: el mediano)")
    ap.add_argument("--fx", type=float, default=605.0, help="focal nominal D435i color 640x480 (px)")
    ap.add_argument("--src-width", type=int, default=640)
    ap.add_argument("--src-height", type=int, default=480)
    ap.add_argument("--max-reach-violation-fraction", type=float, default=0.05,
                    help="demo 'dentro del alcance' si a lo más esta fracción de frames pasa el alcance del puente")
    ap.add_argument("--straight-elbow-m", type=float, default=0.6406, help="hombro-muñeca con el codo recto")
    a = ap.parse_args(argv)

    raw_name = Path(a.raw).stem
    vdir = Path("experiments/v8_validation") / raw_name
    report_csv = Path(a.report) if a.report else vdir / "report.csv"
    start_json = Path(a.start_poses) if a.start_poses else vdir / "start_poses.json"
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    align = yaml.safe_load(open(DEFAULT_ALIGNMENT))["xarm5_dataset_alignment"]
    fcfg = TrajectoryFilterConfig.from_dict(align.get("trajectory_filter"))
    bridge = yaml.safe_load(open(DEFAULT_BRIDGE_PARAMS))["xarm5_bridge"]["ros__parameters"]
    policy = yaml.safe_load(open(DEFAULT_POLICY_PARAMS))["policy_node"]["ros__parameters"]
    conv = Converter(bridge, align, policy)
    raw = zarr.open_group(a.raw, mode="r")
    droid = zarr.open_group(a.droid, mode="r")
    train = zarr.open_group(a.train, mode="r")
    conv_report = json.load(open(Path(a.train).with_name(Path(a.train).stem + "_conversion_report.json")))["episodes"]
    with report_csv.open() as f:
        verdicts = {r["episode"]: r for r in csv.DictReader(f)}
    starts = json.load(open(start_json))
    starts = starts.get("episodes", starts)

    reach = ReachLimit(bridge["workspace_shoulder_xyz_m"], bridge["workspace_wrist_offset_m"],
                       bridge["workspace_max_reach_m"], bridge["workspace_min_reach_m"])
    pitch_lim = (float(bridge["arm_plane_pitch_min"]), float(bridge["arm_plane_pitch_max"]))
    home_train = np.asarray(dict(train.attrs)["xarm5_conversion"]["home_xyz_m"], dtype=float)

    def reach_fractions(tk):
        st = np.asarray(train[tk]["state"][:], dtype=float)
        p, q = st[:, :3] + home_train, st[:, 3:7]
        d = np.array([reach.reach(p[i], min(max(arm_plane_params(q[i], arm_azimuth(p[i]))[0], pitch_lim[0]),
                                                pitch_lim[1])) for i in range(len(p))])
        return float((d > reach.max_reach).mean()), float((d > a.straight_elbow_m).mean())

    raw_keys = episode_keys(raw)
    in_train = {}
    for tk in episode_keys(train):
        at = dict(train[tk].attrs)
        in_train[at.get("raw_episode") or at.get("source_episode")] = tk

    # ---------------- Por demo del crudo ----------------
    rows, psd_live, psd_droid, psd_filt, v_live, v_droid, v_filt = [], [], [], [], [], [], []
    f_ax = None
    for k in raw_keys:
        at = dict(raw[k].attrs)
        v = verdicts.get(k, {})
        status = ("discarded" if at.get("discarded") else "train" if k in in_train and v.get("verdict") != "REVISAR"
                  else "revisar" if k in in_train else "repetir" if v.get("verdict") == "REPETIR" else "discarded")
        sch = at.get("schedule") or {}
        cr = conv_report.get(k, {})
        row = {"episode": k, "status": status, "day": str(Path(str(at.get("raw_bag", ""))).name)[3:11],
               "cell": sch.get("cell"), "yaw_deg": sch.get("yaw_deg"), "verdict": v.get("verdict"),
               "lost_live_pct": float(v["lost_live_pct"]) if v.get("lost_live_pct") else None,
               "droid_max_step_mm": float(v["max_step_mm"]) if v.get("max_step_mm") else None,
               "live_vs_droid_p90_mm": float(v["live_vs_droid_p90_mm"]) if v.get("live_vs_droid_p90_mm") else None,
               "max_angular_speed_rad_s": cr.get("max_angular_speed_rad_s"),
               "reach_flagged": bool(cr.get("flags")), "discard_reason": at.get("discard_reason")}
        sp = starts.get(k) or {}
        row["start_offset_mm"] = sp.get("offset_from_home_mm")
        row["start_offset_deg"] = sp.get("offset_from_home_deg")
        if sp.get("T_base_camera_link") is not None:          # inclinación del eje óptico de camera_link (x) al inicio
            R = np.asarray(sp["T_base_camera_link"], dtype=float)[:3, :3]
            row["axis_tilt_start_pose_deg"] = math.degrees(math.acos(float(np.clip(-R[2, 0], -1, 1))))
        tag = aruco_axis_tilt(np.asarray(raw[k]["image"][:8]), a.fx * 256 / a.src_width, a.fx * 256 / a.src_height)
        row["axis_tilt_aruco_deg"] = None if tag is None else tag[0]
        row["image_roll_aruco_deg"] = None if tag is None else tag[1]
        if k in in_train:
            row["beyond_bridge_reach_frac"], row["beyond_straight_elbow_frac"] = reach_fractions(in_train[k])
            row["within_reach"] = row["beyond_bridge_reach_frac"] <= a.max_reach_violation_fraction
        if k in in_train and k in droid:
            rate = float(at.get("sampling_rate_hz", 20.0))
            lost = np.asarray(raw[k]["lost"][:], dtype=bool) if "lost" in raw[k] else np.zeros(len(raw[k]["state"]), bool)
            live = fill_masked(np.asarray(raw[k]["state"][:, :3], dtype=np.float64), lost)
            dp = np.asarray(droid[k]["state"][:, :3], dtype=np.float64)
            dpf = filtered(dp, rate, fcfg)
            row["noise_rms_live_mm"] = float(np.sqrt(((live - filtered(live, rate, fcfg)) ** 2).sum(1).mean()) * 1000)
            row["noise_rms_droid_mm"] = float(np.sqrt(((dp - dpf) ** 2).sum(1).mean()) * 1000)
            ok = ~(lost | np.r_[False, lost[:-1]] | np.r_[lost[1:], False])   # velocidad en vivo solo sin pérdidas
            sl, sd, sf = frame_speeds(live, rate)[ok], frame_speeds(dp, rate), frame_speeds(dpf, rate)
            v_live.append(sl)
            v_droid.append(sd)
            v_filt.append(sf)
            row.update(speed_mean_droid_filtered=float(sf.mean()), speed_peak_droid_filtered=float(sf.max()),
                       duration_s=float(len(dp) / rate))
            if len(dp) >= 128:                                 # PSD con la misma ventana en todas las demos
                f_ax, p1 = welch(live - live.mean(0), fs=rate, nperseg=128, axis=0)
                _, p2 = welch(dp - dp.mean(0), fs=rate, nperseg=128, axis=0)
                _, p3 = welch(dpf - dpf.mean(0), fs=rate, nperseg=128, axis=0)
                psd_live.append(p1.sum(1))
                psd_droid.append(p2.sum(1))
                psd_filt.append(p3.sum(1))
        rows.append(row)
    kept = [r for r in rows if r["status"] in ("train", "revisar")]
    v_live_all, v_droid_all, v_filt_all = (np.concatenate(x) for x in (v_live, v_droid, v_filt))
    psd_live, psd_droid, psd_filt = (np.median(np.array(x), 0) for x in (psd_live, psd_droid, psd_filt))
    days = sorted({r["day"] for r in rows if r["day"]})

    # ---------------- Figura 1 ----------------
    fig, axs = plt.subplots(2, 3, figsize=(18, 10.5))
    fig.suptitle(f"Handheld demonstrations with DROID-SLAM poses — data-quality diagnosis ({raw_name}, "
                 f"{len(kept)} demonstrations in the training set)", fontsize=14, fontweight="bold")
    ax = axs[0, 0]
    ax.semilogy(f_ax, psd_live, color=GRAY, label="Live odometry (RTAB-Map, guidance only)")
    ax.semilogy(f_ax, psd_droid, color=C2, label="DROID-SLAM RGB-D (offline)")
    ax.semilogy(f_ax, psd_filt, color=C1, label=f"DROID + filter (Hampel + {fcfg.lowpass_cutoff_hz} Hz) = training")
    ax.axvline(fcfg.lowpass_cutoff_hz, color=INK, ls="--", lw=1.0, label="Cut-off frequency")
    ax.set_title("A) Median position PSD across demonstrations", fontweight="bold")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (m²/Hz)")
    ax.grid(True)
    ax.legend(fontsize=8.5)

    lens = sorted((r["duration_s"], r["episode"]) for r in kept if "duration_s" in r)
    ex = a.example_episode if a.example_episode in in_train else lens[len(lens) // 2][1]
    rate = float(raw[ex].attrs.get("sampling_rate_hz", 20.0))
    lost = np.asarray(raw[ex]["lost"][:], dtype=bool)
    live = fill_masked(np.asarray(raw[ex]["state"][:, :3], dtype=np.float64), lost)
    dp = np.asarray(droid[ex]["state"][:, :3], dtype=np.float64)
    t = np.arange(len(dp)) / rate
    ax = axs[0, 1]
    sl = frame_speeds(live, rate)
    sl[lost] = np.nan
    ax.plot(t, sl, color=GRAY, lw=1.0, label="Live odometry (gaps = tracking lost)")
    ax.plot(t, frame_speeds(dp, rate), color=C2, lw=1.0, alpha=0.9, label="DROID")
    ax.plot(t, frame_speeds(filtered(dp, rate, fcfg), rate), color=C1, lw=2.0, label="DROID + filter (training)")
    ax.axhline(conv.max_lin, color=INK, ls="--", lw=1.0, label=f"Robot bridge limit ({conv.max_lin:.2f} m/s)")
    ax.set_title(f"B) Camera speed profile ({ex})", fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.grid(True)
    ax.legend(fontsize=8.5)

    ax = axs[0, 2]
    bins = np.logspace(-3.5, 0.5, 60)
    for vals, color, name in ((v_live_all, GRAY, "Live odometry"), (v_droid_all, C2, "DROID"),
                              (v_filt_all, C1, "DROID + filter (training)")):
        ax.hist(vals, bins=bins, histtype="step", color=color, lw=2.0, density=True,
                label=f"{name} (median {np.median(vals):.3f} m/s)")
    ax.axvline(conv.max_lin, color=INK, ls="--", lw=1.0, label="Robot bridge limit")
    ax.set_xscale("log")
    ax.set_title("C) Per-frame speed distribution", fontweight="bold")
    ax.set_xlabel("Speed (m/s, log)")
    ax.set_ylabel("Density")
    ax.grid(True)
    ax.legend(fontsize=8.5, loc="upper right")

    ax = axs[1, 0]
    dis = [r["live_vs_droid_p90_mm"] for r in kept if r["live_vs_droid_p90_mm"] is not None]
    ax.hist(dis, bins=np.linspace(0, max(100, pct(dis, 99)), 41), color=C1, edgecolor="white", linewidth=1.0,
            label=f"p90 per demonstration: median {np.median(dis):.1f} mm")
    ax.set_title("D) Live odometry vs DROID (rigid alignment, tracked frames)", fontweight="bold")
    ax.set_xlabel("Disagreement p90 (mm)")
    ax.set_ylabel("Demonstrations")
    ax.grid(True)
    ax.legend(fontsize=8.5)

    ax = axs[1, 1]
    for day, color in zip(days, (C2, C1, C3)):
        vals = [r["start_offset_mm"] for r in kept if r["day"] == day and r["start_offset_mm"] is not None]
        if vals:
            ax.hist(vals, bins=np.linspace(0, 160, 33), color=color, alpha=0.55, edgecolor="white", linewidth=1.0,
                    label=f"{day[:4]}-{day[4:6]}-{day[6:]}: median {np.median(vals):.0f} mm (n={len(vals)})")
    ax.set_title("E) Real start pose (ArUco + IMU) vs HOME3", fontweight="bold")
    ax.set_xlabel("TCP offset from HOME3 at the start (mm)")
    ax.set_ylabel("Demonstrations")
    ax.grid(True)
    ax.legend(fontsize=8.5)

    ax = axs[1, 2]
    tag_ok = [(i, r["axis_tilt_aruco_deg"]) for i, r in enumerate(kept)
              if r["axis_tilt_aruco_deg"] is not None and abs(r["image_roll_aruco_deg"]) < 60]
    ax.plot([i for i, r in enumerate(kept) if r.get("axis_tilt_start_pose_deg") is not None],
            [r["axis_tilt_start_pose_deg"] for r in kept if r.get("axis_tilt_start_pose_deg") is not None],
            color=C1, marker="o", ms=3, lw=0, label="Real start pose (ArUco PnP + IMU gravity)")
    ax.plot([i for i, _ in tag_ok], [b for _, b in tag_ok], color=C2, marker="s", ms=3, lw=0,
            label="ArUco tags on 256 px images (independent)")
    ax.set_title("F) Initial optical-axis tilt per demonstration (no drift: odometry reset per demo)", fontweight="bold",
                 fontsize=10)
    ax.set_xlabel("Demonstration index (recording order)")
    ax.set_ylabel("Tilt from vertical (deg)")
    ax.grid(True)
    ax.legend(fontsize=8.5)
    plt.tight_layout()
    save(fig, out / "fig1_data_quality_diagnosis")

    # ---------------- Figura 2: trayectorias en link_base ----------------
    ws = conv.workspace
    sets = []
    if a.reference:
        sets.append((zarr.open_group(a.reference, mode="r"), Path(a.reference).stem, GRAY))
    sets.append((train, Path(a.train).stem, C1))
    fig, axs = plt.subplots(2, len(sets), figsize=(7 * len(sets), 11.5), squeeze=False)
    fig.suptitle("Demonstrations expressed in the xArm5 base frame (TCP, link_base)", fontsize=14, fontweight="bold")
    for col, (ds, name, color) in enumerate(sets):
        home = np.asarray(dict(ds.attrs).get("xarm5_conversion", {}).get("home_xyz_m", conv.home_xyz))
        keys, ends = episode_keys(ds), []
        for k in keys:
            p = np.asarray(ds[k]["action"][:, :3]) + home
            axs[0, col].plot(p[:, 0], p[:, 1], color=color, lw=0.5, alpha=0.35)
            axs[1, col].plot(p[:, 0], p[:, 2], color=color, lw=0.5, alpha=0.35)
            ends.append(p[-1])
        ends = np.array(ends)
        for r, (i, j, xl, yl) in enumerate(((0, 1, "x (m)", "y (m)"), (0, 2, "x (m)", "z (m)"))):
            ax = axs[r, col]
            lo = (ws.x_min, ws.y_min if j == 1 else ws.z_min)
            hi = (ws.x_max, ws.y_max if j == 1 else ws.z_max)
            ax.add_patch(Rectangle(lo, hi[0] - lo[0], hi[1] - lo[1], fill=False, ec=INK, ls="--", lw=1.2,
                                   label="Bridge workspace box (current)"))
            ax.scatter(ends[:, i], ends[:, j], s=10, color=INK, zorder=4, label="Demonstration end (approach goal)")
            ax.scatter([home[i]], [home[j]], marker="*", s=220, color=C3, edgecolor="white", zorder=5, label="HOME3")
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.set_title(f"{'Top' if j == 1 else 'Side'} view — {name} ({len(keys)} demonstrations)",
                         fontweight="bold", fontsize=11)
            ax.grid(True)
    for r in range(2):                                       # mismos ejes en cada fila: la comparación es directa
        xs = [lim for ax in axs[r] for lim in ax.get_xlim()]
        ys = [lim for ax in axs[r] for lim in ax.get_ylim()]
        for ax in axs[r]:
            ax.set_xlim(min(xs), max(xs))
            ax.set_ylim(min(ys), max(ys))
            ax.set_aspect("equal", adjustable="box")
    h, lab = axs[0, -1].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=9)
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out / "fig2_trajectories_base_frame")

    # ---------------- Figura 3: auditoría por demo ----------------
    fig, axs = plt.subplots(3, 1, figsize=(18, 11), sharex=True)
    fig.suptitle("Per-demonstration curation audit (recording order)", fontsize=14, fontweight="bold")
    x = np.arange(len(rows))
    for st, (label, color) in STATUS.items():
        sel = [i for i, r in enumerate(rows) if r["status"] == st]
        for ax, key in zip(axs, ("lost_live_pct", "droid_max_step_mm", "max_angular_speed_rad_s")):
            vals = [rows[i][key] if rows[i][key] is not None else np.nan for i in sel]
            ax.bar(x[sel], vals, color=color, width=0.85, label=f"{label} ({len(sel)})")
    axs[0].set_title("A) Live odometry tracking loss (the dataset poses come from DROID)", fontweight="bold")
    axs[0].set_ylabel("% of frames")
    axs[1].set_title("B) Largest DROID step between frames (non-physical jump → excluded)", fontweight="bold")
    axs[1].set_ylabel("mm (log)")
    axs[1].set_yscale("log")
    axs[1].axhline(20, color=INK, ls="--", lw=1.0, label="Step reviewed against the accelerometer (20 mm)")
    axs[2].set_title("C) Peak angular speed after projection to 5-DoF (conversion report)", fontweight="bold")
    axs[2].set_ylabel("rad/s")
    axs[2].set_xlabel("Raw demonstration index")
    disc = [i for i, r in enumerate(rows) if r["status"] == "discarded"]
    for ax in axs:
        ax.grid(True, axis="y")
        ax.scatter(x[disc], np.full(len(disc), 0.03), marker="v", s=28, color=GRAY, zorder=5,
                   transform=blended_transform_factory(ax.transData, ax.transAxes), clip_on=False)
    h, lab = axs[1].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, 0.965))
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, out / "fig3_curation_audit")

    # ---------------- Figura 5: cobertura celda × orientación ----------------
    cells = sorted({int(r["cell"]) for r in kept if r["cell"] is not None})
    yaws = sorted({float(r["yaw_deg"]) for r in kept if r["yaw_deg"] is not None})
    mats = {}
    feas_name = f"within the robot's reach (≤ {100 * a.max_reach_violation_fraction:.0f} % of frames beyond it)"
    for name, sel in (("in the training set", kept), (feas_name, [r for r in kept if r.get("within_reach")])):
        m = np.zeros((len(cells), len(yaws)), dtype=int)
        for r in sel:
            if r["cell"] is not None and r["yaw_deg"] is not None:
                m[cells.index(int(r["cell"])), yaws.index(float(r["yaw_deg"]))] += 1
        mats[name] = m
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.suptitle("Coverage: demonstrations per table cell × blade orientation", fontsize=14, fontweight="bold")
    vmax = max(int(m.max()) for m in mats.values())
    for ax, (name, m) in zip(axs, mats.items()):
        im = ax.imshow(m, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, str(m[i, j]), ha="center", va="center",
                        color="white" if m[i, j] > 0.6 * vmax else "#0b0b0b", fontsize=9)
        ax.set_xticks(range(len(yaws)), [f"{y:+.0f}°" for y in yaws])
        ax.set_yticks(range(len(cells)), [f"cell {c}" for c in cells])
        ax.set_xlabel("Blade orientation (+90 = right, 180 = toward the base)")
        ax.set_title(f"{name}: {int(m.sum())}", fontweight="bold")
        ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=axs, shrink=0.8, label="Demonstrations")
    save(fig, out / "fig5_coverage_cells_orientations")

    # ---------------- Resumen ----------------
    count = Counter(r["status"] for r in rows)
    tags = [r for r in kept if r["axis_tilt_aruco_deg"] is not None and abs(r["image_roll_aruco_deg"]) < 60]
    summary = {
        "raw_dataset": Path(a.raw).name, "droid_dataset": Path(a.droid).name, "train_dataset": Path(a.train).name,
        "reference_dataset": Path(a.reference).name if a.reference else None,
        "demonstrations_raw": len(rows), "status_counts": dict(count),
        "demonstrations_in_training": len(kept), "frames_in_training": int(sum(len(train[k]["state"]) for k in in_train.values())),
        "per_day_in_training": dict(Counter(r["day"] for r in kept)),
        "reach_flagged_in_training": int(sum(r["reach_flagged"] for r in kept)),
        "within_reach_in_training": int(sum(bool(r.get("within_reach")) for r in kept)),
        "with_frames_beyond_straight_elbow": int(sum(r.get("beyond_straight_elbow_frac", 0) > 0 for r in kept)),
        "frames_beyond_bridge_reach_frac_median": pct([r.get("beyond_bridge_reach_frac") for r in kept], 50),
        "excluded_repetir": [r["episode"] for r in rows if r["status"] == "repetir"],
        "noise_rms_mm_median": {"live": pct([r.get("noise_rms_live_mm") for r in kept], 50),
                                "droid": pct([r.get("noise_rms_droid_mm") for r in kept], 50)},
        "speed_median_m_s": {"live": float(np.median(v_live_all)), "droid": float(np.median(v_droid_all)),
                             "droid_filtered": float(np.median(v_filt_all))},
        "speed_p99_droid_filtered_m_s": float(np.percentile(v_filt_all, 99)),
        "live_tracking_loss_pct_median": pct([r["lost_live_pct"] for r in kept], 50),
        "live_vs_droid_p90_mm_median": pct(dis, 50),
        "droid_max_step_mm_p50_p95": [pct([r["droid_max_step_mm"] for r in kept], 50),
                                       pct([r["droid_max_step_mm"] for r in kept], 95)],
        "start_offset_mm_median_by_day": {d: pct([r["start_offset_mm"] for r in kept if r["day"] == d], 50) for d in days},
        "aruco_axis_tilt_deg": {"n": len(tags), "mean": float(np.mean([r["axis_tilt_aruco_deg"] for r in tags])),
                                "std": float(np.std([r["axis_tilt_aruco_deg"] for r in tags]))},
        "duration_s_median": pct([r.get("duration_s") for r in kept], 50),
        "coverage": {name: m.tolist() for name, m in mats.items()}, "coverage_cells": cells, "coverage_yaws_deg": yaws,
        "episodes": rows,
    }
    (out / "quality_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    ref = json.load(open(a.reference_summary)) if a.reference_summary and Path(a.reference_summary).is_file() else {}
    feas = mats[feas_name]
    md = [f"# Data-quality audit: `{raw_name}` → `{Path(a.train).stem}`", "",
          f"* Demonstrations: {len(rows)} recorded → {len(kept)} in the training set "
          f"({summary['frames_in_training']} frames; per day: "
          + ", ".join(f"{d[:4]}-{d[4:6]}-{d[6:]} {n}" for d, n in sorted(summary["per_day_in_training"].items())) + "). "
          f"Excluded: {count.get('discarded', 0)} discarded (operator or validation), {count.get('repetir', 0)} with a "
          f"non-physical DROID jump ({', '.join(summary['excluded_repetir']) or 'none'}); "
          f"{count.get('revisar', 0)} kept as REVISAR (mostly the final hold still moving).",
          f"* Poses: DROID-SLAM RGB-D offline on the raw bags (the live RTAB-Map odometry only guides the operator). "
          f"Live odometry lost tracking in {summary['live_tracking_loss_pct_median']:.1f} % of the frames (median per "
          f"demonstration); live vs DROID disagreement p90: median {summary['live_vs_droid_p90_mm_median']:.1f} mm.",
          f"* Noise removed by the trajectory filter (median RMS): live odometry "
          f"{summary['noise_rms_mm_median']['live']:.1f} mm, DROID {summary['noise_rms_mm_median']['droid']:.1f} mm"
          + (f" (v7, live VIO: {ref['noise_rms_mm_median']:.1f} mm)" if "noise_rms_mm_median" in ref else "") + ".",
          f"* Median camera speed: live {summary['speed_median_m_s']['live']:.3f} m/s, DROID "
          f"{summary['speed_median_m_s']['droid']:.3f} m/s, DROID + filter {summary['speed_median_m_s']['droid_filtered']:.3f} m/s "
          f"(p99 {summary['speed_p99_droid_filtered_m_s']:.3f} m/s; robot bridge limit {conv.max_lin:.2f} m/s).",
          "* Real start pose vs HOME3 (median TCP offset): " + "; ".join(
              f"{d[:4]}-{d[4:6]}-{d[6:]} {v:.0f} mm" for d, v in summary["start_offset_mm_median_by_day"].items())
          + ". Every demonstration is anchored with its measured start pose (ArUco PnP + IMU).",
          f"* Initial optical-axis tilt from the ArUco tags: {summary['aruco_axis_tilt_deg']['mean']:.1f}° ± "
          f"{summary['aruco_axis_tilt_deg']['std']:.1f}° (n={len(tags)})"
          + (f"; v7: {ref['aruco_axis_tilt_deg']['mean']:.1f}° ± {ref['aruco_axis_tilt_deg']['std']:.1f}°"
             if "aruco_axis_tilt_deg" in ref else "") + ". The odometry is reset per demonstration: no drift.",
          f"* Coverage: {len(cells)} cells × {len(yaws)} blade orientations; per combination "
          f"{int(mats['in the training set'].min())}–{int(mats['in the training set'].max())} demonstrations in the "
          f"training set, {int(feas.min())}–{int(feas.max())} within the robot's reach.",
          f"* Reach: {summary['within_reach_in_training']} of {len(kept)} demonstrations stay within the bridge reach "
          f"(≤ {100 * a.max_reach_violation_fraction:.0f} % of frames beyond it); {summary['with_frames_beyond_straight_elbow']} "
          f"have frames beyond the straight elbow (physically unreachable), mostly the start of the approach (the hand "
          f"moves forward high before descending). The converter flags only {summary['reach_flagged_in_training']} "
          f"(> 50 % of the frames outside); all are kept in the training set.", "",
          "![Diagnosis](fig1_data_quality_diagnosis.png)", "", "![Trajectories](fig2_trajectories_base_frame.png)", "",
          "![Curation](fig3_curation_audit.png)", "", "![Coverage](fig5_coverage_cells_orientations.png)", ""]
    (out / "quality_report.md").write_text("\n".join(md))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("episodes", "coverage")}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
