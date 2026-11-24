#!/usr/bin/env python3
"""Referencia de la vista de HOME3 para la guía de inicio del recolector (``vo_start_guide.py``).

Toma las imágenes que la cámara montada en el xArm5 grabó en HOME3 antes de cada intento real (``pre.png``). El brazo
repite HOME3 con < 0,1 mm, así que esas imágenes definen la vista con la que arranca la política. Guarda la mediana
de las esquinas de los ArUco de la mesa en un YAML.

No necesita ROS ni el robot. Uso (desde la raíz del repo, en el contenedor o en el host con OpenCV):
    python3 tools/hardware/make_home3_view_reference.py
    python3 tools/hardware/make_home3_view_reference.py --images "experiments/xarm5_real_trials/2026-11-23/trial_*/pre.png"
"""

from __future__ import annotations

import argparse
import datetime
import glob
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "visuomotor_data_collection"))

from visuomotor_data_collection.vo_start_guide import build_reference  # noqa: E402

DEFAULT_OUT = REPO / "visuomotor_bringup" / "config" / "realsense" / "home3_view_reference.yaml"
HEADER = """# Vista de HOME3 (cámara montada en el xArm5): esquinas de los ArUco de la mesa, en píxeles de image_size.
# La usa la guía de inicio del recolector (vo_collector_node, start_guide_*) para empezar cada demostración con la
# misma vista con la que arranca la política en el robot.
# Generado por tools/hardware/make_home3_view_reference.py: no editar a mano; regenerar si cambian la cámara, su
# montaje, HOME3 o la posición de los marcadores en la mesa.
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default="experiments/xarm5_real_trials/*/trial_*/pre.png",
                    help="glob de imágenes tomadas en HOME3 (relativo a la raíz del repo)")
    ap.add_argument("--dictionary", default="DICT_4X4_50")
    ap.add_argument("--min-fraction", type=float, default=0.8,
                    help="fracción mínima de imágenes en las que debe verse un marcador para usarlo")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args(argv)

    import cv2
    import yaml
    files = sorted(glob.glob(str(REPO / a.images))) or sorted(glob.glob(a.images))
    if not files:
        print(f"ERROR: ninguna imagen coincide con {a.images}")
        return 1
    images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in files]
    rel = [str(Path(f).resolve().relative_to(REPO)) if Path(f).resolve().is_relative_to(REPO) else f for f in files]
    ref = build_reference(images, a.dictionary, a.min_fraction, source=f"{len(files)} imágenes: {a.images}")
    ref.extra["created"] = datetime.date.today().isoformat()
    ref.extra["files"] = rel
    out = Path(a.out)
    out.write_text(HEADER + yaml.safe_dump(ref.to_dict(), sort_keys=False, allow_unicode=True, width=120))
    print(f"Referencia: {len(ref.markers)} marcadores {sorted(ref.markers)} en {ref.image_size[0]}x{ref.image_size[1]}, "
          f"{len(files)} imágenes; repetibilidad p95 {ref.repeatability_px:.2f} px")
    for i, c in sorted(ref.markers.items()):
        side = sum(((c[k] - c[(k + 1) % 4]) ** 2).sum() ** 0.5 for k in range(4)) / 4
        print(f"  ArUco {i}: centro ({c[:, 0].mean():.1f}, {c[:, 1].mean():.1f}) px, lado {side:.1f} px, "
              f"visto en {ref.extra['n_images_per_marker'][i]}/{len(files)}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
