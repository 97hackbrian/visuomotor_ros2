"""Etapa C, vigilante: espera a que la política termine el RESET, registra el TCP y desarma el puente si la brida baja
de --z-min, si el puente o el driver reportan un error, al pasar --max-s segundos de inferencia o 2 s después de la
parada automática. Al final SIEMPRE desarma (sin frenos: el brazo queda sostenido en modo posición) y guarda un frame."""
import argparse, json, time
import numpy as np, rclpy, cv2
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from rcl_interfaces.msg import Log
from std_srvs.srv import Trigger
import tf2_ros
ap = argparse.ArgumentParser()
ap.add_argument("--log", default="/tmp/vtrials/policy.log")
ap.add_argument("--max-s", type=float, default=45.0)
ap.add_argument("--z-min", type=float, default=0.15)
a = ap.parse_args()
rclpy.init(); n = Node("stageC_monitor")
buf = tf2_ros.Buffer(); tf2_ros.TransformListener(buf, n)
done, warns, imgs = [], [], {}
n.create_subscription(Float32, "/policy/metrics/done_flag", lambda m: done.append((time.time(), float(m.data))), 50)
def log_cb(m):
    if m.level >= 30 and m.name in ("xarm5_bridge", "ufactory_driver", "policy_node", "pose_smoother"):
        warns.append((time.time(), int(m.level), m.name, m.msg[:160]))
n.create_subscription(Log, "/rosout", log_cb, 100)
n.create_subscription(Image, "/camera/camera/color/image_raw", lambda m: imgs.__setitem__("last", m), 2)
cli = n.create_client(Trigger, "/xarm5_bridge/disarm")
def logtxt():
    try:
        return open(a.log, errors="ignore").read()
    except OSError:
        return ""
t_wait = time.time()
while "RESET finalizado" not in logtxt():
    rclpy.spin_once(n, timeout_sec=0.1)
    if time.time() - t_wait > 120:
        break
t0 = time.time(); print(f"[vigilante] inferencia iniciada", flush=True)
tcp, reason, t_auto = [], None, None
last_print = 0
while True:
    rclpy.spin_once(n, timeout_sec=0.02)
    now = time.time()
    try:
        tr = buf.lookup_transform("link_base", "xarm_tcp", rclpy.time.Time())
        p = (tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z)
        if not tcp or tcp[-1][1:] != p:
            tcp.append((now - t0,) + p)
    except Exception:
        pass
    if tcp and tcp[-1][3] < a.z_min:
        reason = f"VIGILANTE: brida a z={tcp[-1][3]:.3f} m < {a.z_min} m"; break
    if any(w[1] >= 40 and w[2] in ("xarm5_bridge", "ufactory_driver") and w[0] >= t0 for w in warns):
        reason = "ERROR del puente o del driver: " + [w for w in warns if w[1] >= 40][-1][3]; break
    if t_auto is None and int(now * 2) != int((now - 0.02) * 2) and "Auto-Stop Neuronal Activado" in logtxt():
        t_auto = now - t0; print(f"[vigilante] parada automática a los {t_auto:.1f} s", flush=True)
    if t_auto is not None and now - t0 > t_auto + 2.0:
        reason = "parada automática de la política"; break
    if now - t0 > a.max_s:
        reason = f"tiempo máximo ({a.max_s:.0f} s)"; break
    if now - last_print > 3 and tcp:
        last_print = now
        print(f"[vigilante] t={now - t0:5.1f}s TCP=({tcp[-1][1]:.3f}, {tcp[-1][2]:.3f}, {tcp[-1][3]:.3f})", flush=True)
# Desarmar SIEMPRE (no frena ni deshabilita motores: el puente vuelve a modo posición y el brazo queda sostenido)
cli.wait_for_service(timeout_sec=3.0)
f = cli.call_async(Trigger.Request()); rclpy.spin_until_future_complete(n, f, timeout_sec=5.0)
disarm_msg = f.result().message if f.result() else "SIN RESPUESTA"
print(f"[vigilante] fin: {reason} | disarm: {disarm_msg}", flush=True)
t_end = time.time()
while "last" not in imgs and time.time() - t_end < 3:
    rclpy.spin_once(n, timeout_sec=0.1)
if "last" in imgs:
    m = imgs["last"]
    im = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, -1)
    cv2.imwrite("/tmp/vtrials/final.png", cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
T = np.array(tcp)
res = {"reason": reason, "disarm": disarm_msg, "inference_s": round(time.time() - t0, 1), "autostop_s": t_auto}
if len(T) > 1:
    v = np.linalg.norm(np.diff(T[:, 1:], axis=0), axis=1) / np.maximum(np.diff(T[:, 0]), 1e-3)
    res.update({"tcp_start": np.round(T[0, 1:], 4).tolist(), "tcp_end": np.round(T[-1, 1:], 4).tolist(),
                "displacement_mm": np.round((T[-1, 1:] - T[0, 1:]) * 1000, 1).tolist(),
                "path_mm": round(float(np.linalg.norm(np.diff(T[:, 1:], axis=0), axis=1).sum() * 1000), 1),
                "z_min": round(float(T[:, 3].min()), 4), "speed_p95_mm_s": round(float(np.percentile(v, 95) * 1000), 1)})
D = np.array(done)
if len(D):
    res["done_max"] = round(float(D[:, 1].max()), 3)
seen = {}
for w in warns:
    seen[(w[2], w[3][:90])] = seen.get((w[2], w[3][:90]), 0) + 1
res["warnings"] = [f"{k[0]} x{c}: {k[1]}" for k, c in list(seen.items())[:8]]
np.save("/tmp/vtrials/tcp.npy", T)
print(json.dumps(res, ensure_ascii=False))
open("/tmp/vtrials/monitor.json", "w").write(json.dumps(res, ensure_ascii=False))
n.destroy_node(); rclpy.shutdown()
