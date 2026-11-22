#!/usr/bin/env python3
"""Imagen de referencia de la vista de la cámara en HOME3, sacada de las demostraciones (solo lectura).

Antes de armar el brazo hay que comprobar que la cámara montada en la brida ve en HOME3 lo mismo que la cámara de
mano al empezar cada demostración: misma orientación (arriba de la imagen = lejos de la base), la mesa y los objetos en
la misma zona. Si no, la política recibe imágenes fuera de distribución y falla de forma sistemática.

Salida: PNG (inglés) con el primer frame de N episodios repartidos por el dataset y su mediana por píxel. Se guarda por
defecto junto a los pesos (deploy/jetson/model/, que no se versiona): copiarlo a la Jetson con model.safetensors y
compararlo con la cámara en vivo (rqt_image_view sobre camera_topic) con el brazo en HOME3.

Uso (contenedor):
    python3 tools/xarm5/export_home3_reference.py --dataset datasets/demonstrations_real_v7_xarm5_v2.zarr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import zarr

REPO = Path(__file__).resolve().parents[2]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#52514e"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=12, help="episodios a mostrar (repartidos por el dataset)")
    ap.add_argument("--out", default=str(REPO / "deploy" / "jetson" / "model" / "home3_reference_view.png"))
    args = ap.parse_args(argv)

    g = zarr.open_group(args.dataset, mode="r")
    names = sorted(g.group_keys(), key=lambda n: int(n.split("_")[-1]) if n.split("_")[-1].isdigit() else n)
    idx = np.unique(np.linspace(0, len(names) - 1, min(args.n, len(names))).round().astype(int))
    sel = [names[i] for i in idx]
    first = np.stack([np.asarray(g[n]["image"][0]) for n in sel])                   # [N, H, W, 3] uint8
    median_all = np.median(np.stack([np.asarray(g[n]["image"][0]) for n in names]), axis=0).astype(np.uint8)

    cols = 4
    rows = int(np.ceil(len(sel) / cols))
    fig = plt.figure(figsize=(4.2 * cols + 6.5, 4.3 * rows))
    gs = fig.add_gridspec(rows, cols + 2, width_ratios=[1] * cols + [0.15, 1.6])
    fig.suptitle(f"HOME3 reference view — first frame of the demonstrations ({Path(args.dataset).name})",
                 fontsize=15, fontweight="bold")
    for k, (name, img) in enumerate(zip(sel, first)):
        ax = fig.add_subplot(gs[k // cols, k % cols])
        ax.imshow(img)
        ax.set_title(name.replace("episode_", "episode "), fontsize=10, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
    ax = fig.add_subplot(gs[:, cols + 1])
    ax.imshow(median_all)
    ax.set_title(f"Per-pixel median of all {len(names)} first frames", fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Top of the image = away from the robot base.\n"
                  "At HOME3 the live camera must show the table\n"
                  "and objects with this orientation and framing,\n"
                  "and no gripper fingers (absent in the demos).", fontsize=10, color=INK)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Referencia HOME3 ({len(sel)} episodios + mediana de {len(names)}) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
