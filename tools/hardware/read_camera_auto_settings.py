#!/usr/bin/env python3
"""Lee la exposición, la ganancia y el balance de blancos que elige la D435i en automático, para fijarlos a mano.

Fase 1 del plan: con la luz fija de la celda, la cámara debe usar exposición y balance MANUALES. Así la imagen no
cambia entre demostraciones, ni entre las demos y la inferencia en el robot. Además, una exposición corta reduce el
desenfoque en movimiento, que degrada la odometría.

Procedimiento:
1. Con la luz definitiva encendida y la cámara en la vista de HOME3, correr este script unos 10 s con la cámara en
   automático. Imprime:
   - la mediana de cada campo numérico de ``/camera/camera/color/metadata``;
   - los parámetros ``rgb_camera.*`` que expone el driver, con sus nombres y valores actuales.
2. Escribir en ``d435i_camera_params.yaml`` los valores sugeridos, en las MISMAS unidades que reporta la cámara:
   ``rgb_camera.enable_auto_exposure: false``, ``rgb_camera.exposure``, ``rgb_camera.gain``,
   ``rgb_camera.enable_auto_white_balance: false``, ``rgb_camera.white_balance`` y
   ``rgb_camera.auto_exposure_priority: false`` (30 fps aunque falte luz).
3. Si la imagen se ve oscura con una exposición más corta, subir la ganancia, no la exposición.
4. La Jetson usa el mismo YAML (``d435i_camera.launch.py``): llevar el cambio también allí.

Uso (contenedor de la cámara con el driver corriendo):
    python3 tools/hardware/read_camera_auto_settings.py --seconds 10
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--metadata-topic", default="/camera/camera/color/metadata")
    ap.add_argument("--camera-node", default="/camera/camera")
    a = ap.parse_args(argv)

    import rclpy
    from rcl_interfaces.srv import GetParameters, ListParameters
    from realsense2_camera_msgs.msg import Metadata

    rclpy.init()
    node = rclpy.create_node("read_camera_auto_settings")
    rows = []
    node.create_subscription(Metadata, a.metadata_topic, lambda m: rows.append(json.loads(m.json_data)), 10)
    t_end = time.time() + a.seconds
    while time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not rows:
        print(f"Sin mensajes en {a.metadata_topic}: ¿driver corriendo y metadatos activados?")
    else:
        print(f"{len(rows)} frames de metadatos en {a.seconds:.0f} s. Mediana [mín, máx] de cada campo numérico:")
        keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
        for k in keys:
            v = np.array([r[k] for r in rows if isinstance(r.get(k), (int, float))], dtype=np.float64)
            if len(v) and not k.endswith(("timestamp", "counter", "number", "arrival")):
                print(f"  {k:36s} {np.median(v):12.2f}  [{v.min():.2f}, {v.max():.2f}]")
    # Nombres reales de los parámetros del driver (validan las claves del YAML)
    lc = node.create_client(ListParameters, f"{a.camera_node}/list_parameters")
    gc = node.create_client(GetParameters, f"{a.camera_node}/get_parameters")
    if lc.wait_for_service(timeout_sec=3.0) and gc.wait_for_service(timeout_sec=3.0):
        req = ListParameters.Request()
        req.prefixes = ["rgb_camera"]
        fut = lc.call_async(req)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=5.0)
        names = sorted(n for n in (fut.result().result.names if fut.result() else []) if any(
            w in n for w in ("exposure", "gain", "white_balance", "priority")))
        g = GetParameters.Request()
        g.names = names
        fut2 = gc.call_async(g)
        rclpy.spin_until_future_complete(node, fut2, timeout_sec=5.0)
        vals = fut2.result().values if fut2.result() else []
        print(f"\nParámetros de {a.camera_node} (nombre: valor actual):")
        for n, v in zip(names, vals):
            val = {1: v.bool_value, 2: v.integer_value, 3: v.double_value, 4: v.string_value}.get(v.type, "?")
            print(f"  {n}: {val}")
    else:
        print(f"\nSin servicio de parámetros en {a.camera_node}.")
    print("\nCopiar a d435i_camera_params.yaml (mismas unidades que arriba), por ejemplo:\n"
          "    rgb_camera.enable_auto_exposure: false\n    rgb_camera.exposure: <actual_exposure, o más corta>\n"
          "    rgb_camera.gain: <gain_level>\n    rgb_camera.enable_auto_white_balance: false\n"
          "    rgb_camera.white_balance: <white_balance>\n    rgb_camera.auto_exposure_priority: false")
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
