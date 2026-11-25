#!/usr/bin/env python3
"""Control de una tanda de demos v8: datos del episodio, bag crudo completo y foto del frame final de cada una.

Por episodio: entrada del programa (celda, orientación), frames, tramo done, frames perdidos en vivo, quién decidió el
final (odometría o giróscopo), velocidad máxima, y en el bag crudo los mensajes de color, profundidad e IMU (deben
cubrir el episodio, con color = profundidad) y que empiece en ARMING (anclaje de inicio). Las descartadas ([U]) se
marcan. Además, una imagen con el último frame de cada episodio (centrado y con las hojas hacia arriba).

Solo lectura. Uso (contenedor con ROS, por ``ros2 bag info``):
    python3 tools/hardware/audit_v8_episodes.py --first 58 --last 67 --png experiments/v8_audits/2026-11-25/ep58_67.png
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TOPICS = {"color": "/camera/camera/color/image_raw/compressed",
          "depth": "/camera/camera/aligned_depth_to_color/image_raw/compressedDepth", "imu": "/camera/camera/imu"}


def bag_counts(bag: str) -> dict:
    if not bag or not os.path.isdir(bag):
        return {k: 0 for k in TOPICS}
    info = subprocess.run(["ros2", "bag", "info", bag], capture_output=True, text=True).stdout
    out = {}
    for k, t in TOPICS.items():
        m = re.search(r"Topic: " + re.escape(t) + r" \| Type: \S+ \| Count: (\d+)", info)
        out[k] = int(m.group(1)) if m else 0
    return out


def main(argv=None) -> int:
    import cv2
    import zarr
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="datasets/demonstrations_real_v8.zarr")
    ap.add_argument("--first", type=int, required=True)
    ap.add_argument("--last", type=int, required=True)
    ap.add_argument("--png", default="", help="imagen con el frame final de cada episodio")
    ap.add_argument("--rate", type=float, default=20.0)
    a = ap.parse_args(argv)
    g = zarr.open_group(str(REPO / a.dataset), mode="r")
    keys = [f"episode_{i}" for i in range(a.first, a.last + 1) if f"episode_{i}" in g]
    tiles, bad = [], []
    print("ep   #prog celda  yaw  frames done perd final      vmax   color depth   imu  bag")
    for k in keys:
        e = g[k]
        at = dict(e.attrs)
        s = at.get("schedule") or {}
        n = len(e["timestamp"])
        lost = int(np.asarray(e["lost"]).sum())
        done = int((np.asarray(e["action"])[:, 8] > 0.5).sum())
        c = bag_counts(at.get("raw_bag", ""))
        ok = (c["color"] > 0 and abs(c["color"] - c["depth"]) <= 2 and c["imu"] > 150 * n / a.rate
              and c["color"] >= 0.95 * 30 * n / a.rate and at.get("raw_bag_starts_at") == "arming")
        if not ok and not at.get("discarded"):
            bad.append(k)
        print(f"{k[8:]:>3} {s.get('index', '-'):>6} {s.get('cell', '-'):>5} {s.get('yaw_deg', 0):+5.0f} {n:>7} {done:>4} "
              f"{lost:>4} {at.get('done_source', '?'):9s} {at.get('max_speed_m_s', 0) * 100:5.1f}  {c['color']:>6} "
              f"{c['depth']:>5} {c['imu']:>5}  {'OK' if ok else 'REVISAR'}{'  DESCARTADA' if at.get('discarded') else ''}")
        if a.png:
            im = cv2.cvtColor(np.asarray(e["image"][-1]), cv2.COLOR_RGB2BGR).copy()
            cv2.putText(im, f"{k[8:]} c{s.get('cell')} {s.get('yaw_deg', 0):+.0f}{' X' if at.get('discarded') else ''}",
                        (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            tiles.append(cv2.resize(im, (160, 160)))
    print("bags a revisar:", bad or "ninguno")
    if a.png and tiles:
        while len(tiles) % 5:
            tiles.append(np.zeros_like(tiles[0]))
        out = REPO / a.png
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), np.vstack([np.hstack(tiles[r:r + 5]) for r in range(0, len(tiles), 5)]))
        print(f"-> {out}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
