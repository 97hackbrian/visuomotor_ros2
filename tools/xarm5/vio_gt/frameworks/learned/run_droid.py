#!/usr/bin/env python3
"""DROID-SLAM (Teed y Deng, NeurIPS 2021) en modo RGB-D sobre un export TUM: escala métrica desde la profundidad.

Preprocesado como en ``demo.py`` de DROID-SLAM: la imagen se reescala a ~384x512 píxeles de área y se recorta a
múltiplos de 8; las intrínsecas se escalan igual y la profundidad (m) se reescala con vecino más cercano. Al final,
``terminate`` rellena las poses de todos los frames (no solo de los keyframes) tras el ajuste global. Salida: TUM
(pose de la cámara en el mundo, marco óptico), una pose por frame.

Uso (imagen visuomotor_learned_vo:v1, con --gpus all): python3 run_droid.py <carpeta_export> <salida.tum> [--buffer N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, "/opt/DROID-SLAM/droid_slam")
from droid import Droid  # noqa: E402


def stream(data, frames, K, scale_depth):
    for i, (t, rgb, _, dep) in enumerate(frames):
        img = cv2.imread(str(data / rgb))
        h0, w0 = img.shape[:2]
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
        img = cv2.resize(img, (w1, h1))[: h1 - h1 % 8, : w1 - w1 % 8]
        d = cv2.imread(str(data / dep), cv2.IMREAD_UNCHANGED).astype(np.float32) / scale_depth
        d = cv2.resize(d, (w1, h1), interpolation=cv2.INTER_NEAREST)[: h1 - h1 % 8, : w1 - w1 % 8]
        intr = torch.tensor([K[0, 0] * w1 / w0, K[1, 1] * h1 / h0, K[0, 2] * w1 / w0, K[1, 2] * h1 / h0])
        yield i, torch.from_numpy(img).permute(2, 0, 1)[None], torch.from_numpy(d), intr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--buffer", type=int, default=512)
    ap.add_argument("--filter-thresh", type=float, default=2.4, help="flujo mínimo (px) para aceptar un frame")
    ap.add_argument("--warmup", type=int, default=8, help="keyframes antes de inicializar")
    a = ap.parse_args()
    data = Path(a.data)
    cal = json.loads((data / "calib.json").read_text())
    K = np.array(cal["K"]).reshape(3, 3)
    frames = [line.split() for line in (data / "associations.txt").read_text().splitlines() if line.strip()]
    # Valores por defecto de demo.py de DROID-SLAM
    args = argparse.Namespace(weights="/opt/DROID-SLAM/droid.pth", buffer=a.buffer, image_size=[384, 512],
                              disable_vis=True, stereo=False, upsample=False, beta=0.3, filter_thresh=a.filter_thresh,
                              warmup=a.warmup,
                              keyframe_thresh=4.0, frontend_thresh=16.0, frontend_window=25, frontend_radius=2,
                              frontend_nms=1, backend_thresh=22.0, backend_radius=2, backend_nms=3,
                              asynchronous=False, frontend_device="cuda", backend_device="cuda")
    droid, t0 = None, time.time()
    for i, img, d, intr in stream(data, frames, K, cal["depth_scale"]):
        if droid is None:
            args.image_size = [img.shape[2], img.shape[3]]
            droid = Droid(args)
        droid.track(i, img, d, intrinsics=intr)
        if i % 500 == 0:
            print(f"frame {i}/{len(frames)} {time.time() - t0:.0f} s, GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB",
                  flush=True)
    # El relleno de poses (trajectory_filler) espera (t, imagen, intrínsecas), sin profundidad
    traj = droid.terminate((i, img, intr) for i, img, _, intr in stream(data, frames, K, cal["depth_scale"]))
    traj = traj.cpu().numpy() if hasattr(traj, "cpu") else np.asarray(traj)
    with open(a.out, "w") as f:
        for fr, p in zip(frames, traj):
            f.write(f"{float(fr[0]):.6f} " + " ".join(f"{v:.8f}" for v in p[:7]) + "\n")
    print(f"{len(traj)} poses de {len(frames)} frames, {time.time() - t0:.0f} s, "
          f"GPU máx {torch.cuda.max_memory_allocated() / 2**30:.2f} GB -> {a.out}")


if __name__ == "__main__":
    main()
