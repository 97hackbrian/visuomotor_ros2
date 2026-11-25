#!/usr/bin/env python3
"""DPVO (Teed, Lipson y Deng, NeurIPS 2023): odometría visual monocular aprendida sobre un export TUM.

Monocular: la trayectoria NO tiene escala métrica. Evaluarla con ``evaluate_vio_gt.py --scale-free`` (ajusta la escala
Sim(3) y la reporta). Entrada como en ``demo.py`` de DPVO: imagen BGR uint8 completa (640x480, múltiplo de 16) e
intrínsecas [fx, fy, cx, cy]. Salida: TUM (pose de la cámara en el mundo, marco óptico), una pose por frame.

Uso (imagen visuomotor_learned_vo:v1, con --gpus all): python3 run_dpvo.py <carpeta_export> <salida.tum>
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, "/opt/DPVO")
from dpvo.config import cfg  # noqa: E402
from dpvo.dpvo import DPVO  # noqa: E402


def main():
    data, out = Path(sys.argv[1]), sys.argv[2]
    # Paso entre frames (por defecto 3 = 10 Hz). Con los 30 Hz completos, la optimización final de terminate() se
    # queda sin memoria en 8 GB de GPU en secuencias de > 2000 frames: con el movimiento lento casi no se descartan
    # frames del grafo. Los frames saltados no tienen pose: evaluar SIN --frame-times.
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    cfg.merge_from_file("/opt/DPVO/config/default.yaml")
    cfg.BUFFER_SIZE = 1024
    cal = json.loads((data / "calib.json").read_text())
    K = np.array(cal["K"]).reshape(3, 3)
    frames = [line.split() for line in (data / "rgb.txt").read_text().splitlines() if line.strip()][::stride]
    intr = torch.tensor([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=torch.float32).cuda()
    slam, t0 = None, time.time()
    for i, (_, rel) in enumerate(frames):
        img = cv2.imread(str(data / rel))
        h, w = img.shape[0] - img.shape[0] % 16, img.shape[1] - img.shape[1] % 16
        img = torch.from_numpy(img[:h, :w]).permute(2, 0, 1).cuda()
        if slam is None:
            slam = DPVO(cfg, "/opt/DPVO/dpvo.pth", ht=h, wd=w, viz=False)
        with torch.no_grad():
            slam(i, img, intr)
        if i % 500 == 0:
            print(f"frame {i}/{len(frames)} {time.time() - t0:.0f} s, GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB",
                  flush=True)
    res = slam.terminate()
    poses, tst = (res if isinstance(res, tuple) else (res, np.arange(len(frames))))[:2]
    poses, tst = np.asarray(poses), np.asarray(tst).astype(int)
    times = np.array([float(f[0]) for f in frames])
    with open(out, "w") as f:
        for p, k in zip(poses, tst):
            f.write(f"{times[k]:.6f} " + " ".join(f"{v:.8f}" for v in p[:7]) + "\n")
    print(f"{len(poses)} poses de {len(frames)} frames, {time.time() - t0:.0f} s, "
          f"GPU máx {torch.cuda.max_memory_allocated() / 2**30:.2f} GB -> {out}")


if __name__ == "__main__":
    main()
