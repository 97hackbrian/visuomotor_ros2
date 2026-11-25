#!/usr/bin/env python3
"""¿Está bien la calibración de fábrica de la cámara de color? Calibración con el tablero desde una corrida static.

Toma las imágenes de las paradas de una corrida ``static`` de ``run_vio_gt.sh`` (robot quieto, tablero a la vista
desde distintas poses), detecta las esquinas del ChArUco MEDIDO (``measured: true``) y:
1. Reproyección con las intrínsecas de fábrica (``camera_info`` del bag) y PnP por imagen.
2. ``cv2.calibrateCamera`` partiendo de las de fábrica: fx, fy, cx, cy, k1 y k2 libres. Reporta el cambio relativo de
   la focal. Con un tablero bien medido, un cambio de focal de ~X % se ve como un error de escala de ~X % en las
   distancias del PnP (y en la calibración cámara-brida con escala libre).
3. Repetibilidad: la calibración con la mitad de las paradas (pares e impares) da la dispersión de la estimación.

No modifica nada: escribe ``color_intrinsics_check.json`` en la carpeta de la corrida.

Uso (contenedor, con ROS para leer el bag):
    python3 tools/xarm5/vio_gt/check_color_intrinsics.py --run experiments/vio_gt/<fecha>/runs/<static>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
for p in ("tools/hardware", "visuomotor_bringup"):
    sys.path.insert(0, str(REPO / p))
from vio_bag_utils import decode_image, iter_messages, stamp_of  # noqa: E402

COLOR = "/camera/camera/color/image_raw/compressed"
INFO = "/camera/camera/color/camera_info"


def calibrate(obj, img, size, K0, D0, fix_dist=False, focal_only=False):
    import cv2
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    if focal_only:   # un solo factor de focal: centro, proporción fx/fy y distorsión de fábrica fijos
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2
    if fix_dist:
        flags |= cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2
    D = np.zeros(5) if D0 is None or len(D0) < 5 else np.asarray(D0[:5], float).copy()
    rms, K, D, _, _ = cv2.calibrateCamera(obj, img, size, K0.copy(), D, flags=flags)
    return float(rms), K, D.ravel()


def main(argv=None) -> int:
    import cv2
    from visuomotor_bringup.charuco import CharucoConfig, chessboard_corners, detect_corners, make_board
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--board-config", default=str(REPO / "visuomotor_bringup/config/realsense/charuco_board.yaml"))
    ap.add_argument("--min-corners", type=int, default=24)
    ap.add_argument("--per-stop", type=int, default=6, help="imágenes por parada (repartidas en la ventana)")
    a = ap.parse_args(argv)
    run = Path(a.run)
    cfg = CharucoConfig.load(a.board_config)
    if not cfg.measured:
        raise SystemExit("el tablero debe estar medido (measured: true)")
    board = make_board(cfg)
    P3 = chessboard_corners(board)
    plan = json.loads((run / "plan.json").read_text())
    offset = json.loads((run / "clock.json").read_text())["offset_pc_minus_jetson_s"]
    t0 = float(np.load(run / "robot.npz")["t0"]) + offset
    # Ventanas de quietud (margen amplio: el reloj ssh pudo tener ~0,4 s de sesgo en esta sesión)
    wins = [(t0 + s + 2.0, t0 + e - 0.6, n) for s, e, n in plan["segments"] if n.startswith("static")]
    bag = REPO / (run / "bag_path.txt").read_text().strip()
    K0 = D0 = None
    size = None
    frames = {w[2]: [] for w in wins}
    for topic, m, _ in iter_messages(bag, [INFO, COLOR]):
        if topic == INFO:
            if K0 is None:
                K0, D0, size = np.array(m.k, float).reshape(3, 3), np.array(m.d, float), (m.width, m.height)
            continue
        ts = stamp_of(m)
        w = next((w for w in wins if w[0] <= ts <= w[1]), None)
        if w is not None:
            frames[w[2]].append(m)
    obj, img, names = [], [], []
    for name, msgs in frames.items():
        for m in [msgs[int(i)] for i in np.linspace(0, len(msgs) - 1, min(a.per_stop, len(msgs))).round()] if msgs else []:
            g = cv2.cvtColor(decode_image(m), cv2.COLOR_RGB2GRAY)
            px, ids = detect_corners(g, board, cfg, K0, D0)
            if ids is None or len(ids) < a.min_corners:
                continue
            obj.append(P3[np.asarray(ids).ravel()].astype(np.float32))
            img.append(np.asarray(px, np.float32).reshape(-1, 1, 2))
            names.append(name)
    if len(obj) < 10:
        raise SystemExit(f"solo {len(obj)} imágenes con >= {a.min_corners} esquinas")
    # 1) Fábrica: reproyección con PnP por imagen (intrínsecas fijas)
    errs = []
    for o, i in zip(obj, img):
        ok, rv, tv = cv2.solvePnP(o, i, K0, D0)
        pr, _ = cv2.projectPoints(o, rv, tv, K0, D0)
        errs.append(np.sqrt(np.mean(np.sum((pr - i) ** 2, axis=2))))
    rms_factory = float(np.sqrt(np.mean(np.square(errs))))
    # 2) Calibración completa y con distorsión fija
    rms, K, D = calibrate(obj, img, size, K0, D0)
    rms_f, Kf, _ = calibrate(obj, img, size, K0, D0, fix_dist=True)
    # Un solo factor de focal (lo único que estas vistas, todas desde un mismo lado, determinan bien)
    rms_s, Ks, _ = calibrate(obj, img, size, K0, D0, focal_only=True)
    # 3) Repetibilidad por mitades de paradas (con el modelo de un solo factor)
    stops = sorted(set(names))
    halves = []
    for h in (stops[0::2], stops[1::2]):
        idx = [k for k, n in enumerate(names) if n in h]
        _, Kh, _ = calibrate([obj[k] for k in idx], [img[k] for k in idx], size, K0, D0, focal_only=True)
        halves.append(Kh)
    pct = lambda Kx, k: 100.0 * (Kx[k] / K0[k] - 1.0)  # noqa: E731
    out = {"images": len(obj), "stops": len(stops), "image_size": list(size),
           "factory": {"fx": K0[0, 0], "fy": K0[1, 1], "cx": K0[0, 2], "cy": K0[1, 2], "d": D0.tolist(),
                       "reprojection_rms_px": rms_factory},
           "calibrated": {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2], "d": D.tolist(),
                          "reprojection_rms_px": rms, "fx_change_pct": pct(K, (0, 0)), "fy_change_pct": pct(K, (1, 1))},
           "calibrated_fixed_distortion": {"fx": Kf[0, 0], "fy": Kf[1, 1], "reprojection_rms_px": rms_f,
                                           "fx_change_pct": pct(Kf, (0, 0)), "fy_change_pct": pct(Kf, (1, 1))},
           "focal_only": {"fx": Ks[0, 0], "fy": Ks[1, 1], "reprojection_rms_px": rms_s,
                          "focal_change_pct": pct(Ks, (0, 0))},
           "halves_fx_change_pct": [pct(Kh, (0, 0)) for Kh in halves],
           "halves_fy_change_pct": [pct(Kh, (1, 1)) for Kh in halves]}
    (run / "color_intrinsics_check.json").write_text(json.dumps(out, indent=2))
    print(f"{len(obj)} imágenes de {len(stops)} paradas (>= {a.min_corners} esquinas)")
    print(f"fábrica:    fx {K0[0, 0]:.1f} fy {K0[1, 1]:.1f} cx {K0[0, 2]:.1f} cy {K0[1, 2]:.1f} | reproyección "
          f"{rms_factory:.3f} px")
    print(f"calibrada:  fx {K[0, 0]:.1f} ({pct(K, (0, 0)):+.2f} %) fy {K[1, 1]:.1f} ({pct(K, (1, 1)):+.2f} %) "
          f"cx {K[0, 2]:.1f} cy {K[1, 2]:.1f} k1 {D[0]:+.4f} k2 {D[1]:+.4f} | reproyección {rms:.3f} px")
    print(f"sin distorsión libre: fx {pct(Kf, (0, 0)):+.2f} % fy {pct(Kf, (1, 1)):+.2f} % | reproyección {rms_f:.3f} px")
    print(f"solo la focal (centro, fx/fy y distorsión de fábrica): f {pct(Ks, (0, 0)):+.2f} % | reproyección "
          f"{rms_s:.3f} px")
    print("  mitades de paradas: " + ", ".join(f"{v:+.2f} %" for v in out["halves_fx_change_pct"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
