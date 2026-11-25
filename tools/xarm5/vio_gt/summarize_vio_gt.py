#!/usr/bin/env python3
"""Resumen de las corridas de odometría contra el xArm5: distribución entre corridas (nunca el mejor caso).

Lee ``runs/*/vio_vs_robot_<label>.json`` de una sesión y escribe ``summary_<label>.md`` y ``.json`` con la mediana,
el p90 y el máximo entre corridas de cada métrica, y la tabla por corrida. Con varias etiquetas (candidatos
reprocesados) agrega una tabla comparativa.

Uso: python3 tools/xarm5/vio_gt/summarize_vio_gt.py --session experiments/vio_gt/2026-11-24 [--labels live,baseline,...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KEYS = [("ATE anclada mediana (mm)", lambda r: r["ate_anchored_mm"]["median"]),
        ("ATE anclada p90 (mm)", lambda r: r["ate_anchored_mm"]["p90"]),
        ("ATE anclada máx (mm)", lambda r: r["ate_anchored_mm"]["max"]),
        ("error final (mm)", lambda r: r["final_error_anchored_mm"]),
        ("ATE Sim(3) mediana (mm)", lambda r: r["ate_sim3_mm"]["median"]),
        ("error de escala (%)", lambda r: r["scale_error_pct"]),
        ("RPE 0,5 s mediana (mm)", lambda r: r["rpe"]["0.5s"]["trans_mm"]["median"]),
        ("RPE 1 s p90 (mm)", lambda r: r["rpe"]["1s"]["trans_mm"]["p90"]),
        ("rotación anclada mediana (°)", lambda r: r["rot_anchored_deg"]["median"]),
        ("pérdidas (%)", lambda r: r["lost_pct"]),
        ("útil <= 10 mm (% de frames)", lambda r: r.get("useful_pct", {}).get("10mm", float("nan"))),
        ("útil <= 20 mm (% de frames)", lambda r: r.get("useful_pct", {}).get("20mm", float("nan")))]


def load(session: Path, label: str):
    rows = []
    for f in sorted(session.glob(f"runs/*/vio_vs_robot_{label}.json")):
        d = json.loads(f.read_text())
        rows.append((f.parent.name, d["plan"], d["odometry_vs_robot"]))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--labels", default="live")
    ap.add_argument("--kind", default="approach", help="prefijo del plan a resumir ('' = todos)")
    a = ap.parse_args(argv)
    session = Path(a.session)
    out, md = {}, []
    for label in a.labels.split(","):
        rows = [r for r in load(session, label) if r[1].startswith(a.kind)]
        if not rows:
            continue
        stats = {}
        for name, fn in KEYS:
            v = np.array([fn(r[2]) for r in rows], float)
            stats[name] = {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "max": float(v.max()),
                           "values": v.round(2).tolist()}
        out[label] = {"n_runs": len(rows), "runs": [r[0] for r in rows], "stats": stats}
        md += [f"## {label} ({len(rows)} corridas '{a.kind or 'todas'}')", "",
               "| métrica | mediana | p90 | máx |", "|---|---|---|---|"]
        md += [f"| {k} | {s['median']:.2f} | {s['p90']:.2f} | {s['max']:.2f} |" for k, s in stats.items()]
        md += ["", "| corrida | recorrido (m) | ATE med | ATE máx | final | escala % | RPE 0,5 s | pérdidas % |",
               "|---|---|---|---|---|---|---|---|"]
        for run, _, r in rows:
            md.append(f"| {run} | {r['path_length_m']:.2f} | {r['ate_anchored_mm']['median']:.1f} | "
                      f"{r['ate_anchored_mm']['max']:.1f} | {r['final_error_anchored_mm']:.1f} | "
                      f"{r['scale_error_pct']:+.2f} | {r['rpe']['0.5s']['trans_mm']['median']:.2f} | "
                      f"{r['lost_pct']:.1f} |")
        md.append("")
    if len(out) > 1:
        md += ["## Comparación (mediana entre corridas)", "",
               "| candidato | ATE med | ATE p90 | ATE máx | escala % | RPE 0,5 s | pérdidas % | útil ≤ 10 mm % | "
               "útil ≤ 20 mm % (peor corrida) |", "|---|---|---|---|---|---|---|---|---|"]
        for label, d in out.items():
            s = d["stats"]
            u20 = s["útil <= 20 mm (% de frames)"]
            md.append(f"| {label} | {s['ATE anclada mediana (mm)']['median']:.1f} | "
                      f"{s['ATE anclada p90 (mm)']['median']:.1f} | {s['ATE anclada máx (mm)']['median']:.1f} | "
                      f"{s['error de escala (%)']['median']:+.2f} | {s['RPE 0,5 s mediana (mm)']['median']:.2f} | "
                      f"{s['pérdidas (%)']['median']:.2f} | {s['útil <= 10 mm (% de frames)']['median']:.0f} | "
                      f"{u20['median']:.0f} ({min(u20['values']):.0f}) |")
    tag = "_".join(a.labels.split(","))[:60]
    (session / f"summary_{a.kind or 'all'}_{tag}.json").write_text(json.dumps(out, indent=2))
    (session / f"summary_{a.kind or 'all'}_{tag}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
