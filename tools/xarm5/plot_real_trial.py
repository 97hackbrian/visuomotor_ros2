#!/usr/bin/env python3
"""Gráfica de control de un intento en el xArm5 real (figura en inglés, PNG 200 DPI + PDF).

Paneles:
- A, B y C: posición x, y, z del TCP en el tiempo, comando de la política frente a medida;
- D: velocidad del TCP frente al límite del puente;
- E: señal de término (done) frente al umbral, con el instante de la parada automática;
- F: trayectoria vista desde arriba (x-y), con los finales de las 76 demos como contexto.

Fuentes, de mejor a peor:
1. ``timeseries.npz`` del vigilante: comando de la política, objetivo filtrado, comando del puente, TCP medido y done
   a resolución completa.
2. Si no está (intentos 1 a 8 del 2026-11-23): ``tcp.npy`` (TCP medido) y ``policy.log``, de donde salen el comando
   y el done de la línea [IA] que la política escribe a 1 Hz.

Uso: python3 tools/xarm5/plot_real_trial.py experiments/xarm5_real_trials/2026-11-23/trial_008 [...]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C1, C2, C3, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#52514e"
plt.rcParams.update({"axes.edgecolor": INK, "axes.labelcolor": "#0b0b0b", "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "grid.color": "#d9d8d4",
                     "grid.linewidth": 0.6, "lines.linewidth": 2.0, "legend.frameon": False, "font.size": 10})
LOG_T = re.compile(r"\[(\d+\.\d+)\]")
LOG_IA = re.compile(r"X=(-?[\d.]+) Y=(-?[\d.]+) Z=(-?[\d.]+).*?Done\): (-?[\d.]+)")


def from_log(trial: Path):
    """Comando de la política y done a 1 Hz desde policy.log, con el tiempo relativo al fin del RESET."""
    t0, t_stop, rows = None, None, []
    for line in (trial / "policy.log").read_text(errors="ignore").splitlines():
        m = LOG_T.search(line)
        if not m:
            continue
        t = float(m.group(1))
        if "RESET finalizado" in line:
            t0 = t
        elif "Auto-Stop Neuronal Activado" in line:
            t_stop = t
        elif "[IA] Accion" in line:
            g = LOG_IA.search(line)
            if g:
                rows.append([t] + [float(v) for v in g.groups()])
    if t0 is None:
        return None
    a = np.array(rows)
    out = {"cmd": a[:, :4] - np.array([t0, 0, 0, 0]) if len(a) else np.zeros((0, 4)),
           "done": np.c_[a[:, 0] - t0, a[:, 4]] if len(a) else np.zeros((0, 2)),
           "t_stop": None if t_stop is None else t_stop - t0}
    out["cmd"] = out["cmd"][out["cmd"][:, 0] >= 0] if len(out["cmd"]) else out["cmd"]
    return out


def plot(trial: Path, session_ref: dict | None) -> Path:
    mon = json.loads((trial / "monitor.json").read_text())
    metrics = {}
    rows_file = trial.parent / "trials.jsonl"
    if rows_file.exists():
        for line in rows_file.read_text().splitlines():
            r = json.loads(line)
            if r["trial"] == trial.name:
                metrics = r
    ts = trial / "timeseries.npz"
    if ts.exists():
        d = np.load(ts)
        tcp = d["tcp"]
        cmd = d["target_raw"] if "target_raw" in d else np.zeros((0, 4))
        smooth = d["target"] if "target" in d else np.zeros((0, 4))
        bridge = d["preview"] if "preview" in d else np.zeros((0, 4))
        done = d["done"] if "done" in d else np.zeros((0, 2))
        t_stop = mon.get("autostop_s")
        source = "full-rate monitor recording"
    else:
        tcp = np.load(trial / "tcp.npy")
        lg = from_log(trial) or {"cmd": np.zeros((0, 4)), "done": np.zeros((0, 2)), "t_stop": None}
        cmd, done = lg["cmd"], lg["done"]
        smooth = bridge = np.zeros((0, 4))
        t_stop = mon.get("autostop_s") if mon.get("autostop_s") is not None else lg["t_stop"]
        source = "measured TCP (full rate) + policy command/done from the 1 Hz log line"
    fig, axs = plt.subplots(2, 3, figsize=(18, 9.5))
    ok_over = metrics.get("success_over_tool")
    ok_demo = metrics.get("success_auto")
    if metrics:
        verdict = (f"{'stopped over the tool' if ok_over else 'NOT over the tool'}, "
                   f"{'final view like the demos' if ok_demo else 'final view differs from the demos'}")
    else:
        verdict = "not evaluated yet (run real_trial_metrics.py)"
    title = f"xArm5 real trial {trial.name.split('_')[1]} — {metrics.get('label', '')} — {verdict}"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    for k, (ax, lab) in enumerate(zip(axs[0], ("x", "y", "z"))):
        if len(bridge):
            ax.plot(bridge[:, 0], bridge[:, 1 + k], color=GRAY, lw=1.2, label="Bridge servo command")
        if len(smooth):
            ax.plot(smooth[:, 0], smooth[:, 1 + k], color=C3, lw=1.4, label="Smoothed target (1.5 Hz)")
        if len(cmd):
            style = dict(lw=1.2) if ts.exists() else dict(lw=0, marker="o", ms=4)
            ax.plot(cmd[:, 0], cmd[:, 1 + k], color=C2, label="Policy command", **style)
        ax.plot(tcp[:, 0], tcp[:, 1 + k], color=C1, lw=2.0, label="Measured TCP")
        if t_stop is not None:
            ax.axvline(t_stop, color=INK, ls="--", lw=1.0, label="Neural auto-stop")
        ax.set_title(f"{'ABC'[k]}) TCP {lab} (link_base)", fontweight="bold")
        ax.set_xlabel("Time since policy start (s)")
        ax.set_ylabel(f"{lab} (m)")
        ax.grid(True)
        if k == 0:
            ax.legend(fontsize=8.5, loc="best")
    ax = axs[1, 0]
    if len(tcp) > 3:
        dt = np.maximum(np.diff(tcp[:, 0]), 1e-3)
        v = np.linalg.norm(np.diff(tcp[:, 1:4], axis=0), axis=1) / dt
        k = max(1, int(round(0.25 / np.median(dt))))                  # media móvil de 0,25 s
        vs = np.convolve(v, np.ones(k) / k, mode="same")
        ax.plot(tcp[1:, 0], vs * 100, color=C1, lw=1.6, label="Measured TCP speed (0.25 s mean)")
    ax.axhline(5.0, color=GRAY, ls="--", lw=1.2, label="Bridge limit (5 cm/s)")
    ax.set_title("D) TCP speed", fontweight="bold")
    ax.set_xlabel("Time since policy start (s)")
    ax.set_ylabel("Speed (cm/s)")
    ax.grid(True)
    ax.legend(fontsize=8.5)
    ax = axs[1, 1]
    if len(done):
        ax.plot(done[:, 0], done[:, 1], color=C2, lw=1.4 if ts.exists() else 0, marker=None if ts.exists() else "o",
                ms=4, label="Predicted done (9th dim)")
    ax.axhline(0.60, color=GRAY, ls="--", lw=1.2, label="Auto-stop threshold (0.60 × 5 steps)")
    if t_stop is not None:
        ax.axvline(t_stop, color=INK, ls="--", lw=1.0, label=f"Neural auto-stop ({t_stop:.1f} s)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("E) Task-completion signal", fontweight="bold")
    ax.set_xlabel("Time since policy start (s)")
    ax.set_ylabel("done")
    ax.grid(True)
    ax.legend(fontsize=8.5, loc="upper left")
    ax = axs[1, 2]
    if session_ref and "end_xyz" in session_ref:
        e = np.array(session_ref["end_xyz"])
        ax.scatter(e[:, 1], e[:, 0], s=12, color=GRAY, alpha=0.6, label="Demo end points (76)")
    ax.plot(tcp[:, 2], tcp[:, 1], color=C1, lw=2.0, label="Measured TCP path")
    ax.scatter([tcp[0, 2]], [tcp[0, 1]], marker="*", s=200, color=C3, edgecolor="white", zorder=5, label="HOME3")
    ax.scatter([tcp[-1, 2]], [tcp[-1, 1]], marker="o", s=70, color=C2, edgecolor="white", zorder=5, label="Final pose")
    ax.axvline(0.30, color=GRAY, ls=":", lw=1.0)
    ax.axvline(-0.30, color=GRAY, ls=":", lw=1.0, label="Bridge box |y| = 0.30 m")
    ax.invert_xaxis()                                          # +y a la izquierda, como se ve desde detrás del robot
    ax.set_title("F) Top view (robot base at the bottom)", fontweight="bold")
    ax.set_xlabel("y (m, +left)")
    ax.set_ylabel("x (m, forward)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)
    ax.legend(fontsize=8, loc="lower left")
    fig.text(0.01, 0.005, f"Data: {source}. Stop reason: {mon.get('reason')}. Path {mon.get('path_mm')} mm, "
             f"final TCP {mon.get('tcp_end')} m.", fontsize=8.5, color=INK)
    plt.tight_layout(rect=(0, 0.02, 1, 0.96))
    out = trial / "control"
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out.with_suffix(".png")


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    for a in args:
        trial = Path(a)
        ref_file = trial.parent / "demo_reference.json"
        ref = json.loads(ref_file.read_text()) if ref_file.exists() else None
        print(plot(trial, ref))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
