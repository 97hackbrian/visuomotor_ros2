"""Ejecuta un plan de plan_vio_gt.py en el xArm5 real y registra el TCP medido (referencia para la odometría).

Corre DENTRO del contenedor de la Jetson (puente ya lanzado por el usuario, brazo en HOME3, motores sin frenos).
Publica las poses del plan a 20 Hz en /target_frame_raw (pasan por pose_smoother y el puente, con su caja, su alcance
y sus límites de velocidad). Registra la TF link_base -> xarm_tcp con el reloj de la Jetson.

Seguridad:
* no arma si el puente está armado o con error, ni si el TCP no está en la primera pose del plan (15 mm, 3°);
* vigilante: TCP bajo --z-min, error del puente o del driver, puente desarmado por su cuenta, TCP a más de
  --max-track-m de la pose pedida, o tiempo excedido -> deja de publicar y desarma;
* al final SIEMPRE desarma (sin frenos: el brazo queda sostenido en modo posición). Ctrl+C también desarma.
Salida: /tmp/vio_gt/<plan>.npz (tcp: t x y z qx qy qz qw; cmd: t índice) y <plan>.json (resumen).
"""
import argparse
import json
import math
import signal
import time

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from rcl_interfaces.msg import Log
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

ap = argparse.ArgumentParser()
ap.add_argument("plan")
ap.add_argument("--out-dir", default="/tmp/vio_gt")
ap.add_argument("--z-min", type=float, default=0.15)
ap.add_argument("--max-track-m", type=float, default=0.06)
ap.add_argument("--start-tol-m", type=float, default=0.015)
ap.add_argument("--start-tol-deg", type=float, default=3.0)
ap.add_argument("--no-arm", action="store_true", help="solo comprobaciones previas; no arma ni publica")
a = ap.parse_args()
plan = json.load(open(a.plan))
P = np.asarray(plan["poses"], float)
rate = float(plan["rate_hz"])

rclpy.init()
n = Node("vio_gt_motion")
buf = tf2_ros.Buffer()
tf2_ros.TransformListener(buf, n)
pub = n.create_publisher(PoseStamped, "/target_frame_raw", 10)
status, errors = {}, []
n.create_subscription(String, "/xarm5_bridge/status", lambda m: status.__setitem__("s", m.data), 10)


def log_cb(m):
    if m.level >= 40 and m.name in ("xarm5_bridge", "ufactory_driver", "pose_smoother"):
        errors.append((time.time(), m.name, m.msg[:160]))


n.create_subscription(Log, "/rosout", log_cb, 100)
arm_cli = n.create_client(Trigger, "/xarm5_bridge/arm")
disarm_cli = n.create_client(Trigger, "/xarm5_bridge/disarm")
tcp, cmd = [], []
stop = {"reason": None}
signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("reason", "Ctrl+C"))
signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("reason", "SIGTERM"))


def read_tcp():
    try:
        tr = buf.lookup_transform("link_base", "xarm_tcp", rclpy.time.Time())
    except Exception:
        return None
    t, r = tr.transform.translation, tr.transform.rotation
    return np.array([t.x, t.y, t.z, r.x, r.y, r.z, r.w])


def call(cli, timeout=8.0):
    if not cli.wait_for_service(timeout_sec=3.0):
        return "SIN SERVICIO"
    f = cli.call_async(Trigger.Request())
    rclpy.spin_until_future_complete(n, f, timeout_sec=timeout)
    return f.result().message if f.result() else "SIN RESPUESTA"


def qang(q0, q1):
    return math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(q0, q1))))))


t_wait = time.time()
while time.time() - t_wait < 3.0:
    rclpy.spin_once(n, timeout_sec=0.05)
cur = read_tcp()
s = status.get("s", "")
pre = []
if cur is None:
    pre.append("sin TF link_base->xarm_tcp")
if "armed=False" not in s or "err=0" not in s:
    pre.append(f"estado del puente no apto: {s[:90]}")
if "dry_run=False" not in s:
    pre.append("el puente está en dry_run")
if cur is not None:
    d, ang = np.linalg.norm(cur[:3] - P[0, :3]), qang(cur[3:], P[0, 3:])
    if d > a.start_tol_m or ang > a.start_tol_deg:
        pre.append(f"TCP lejos de la pose inicial del plan ({d * 1000:.1f} mm, {ang:.1f}°): llevar a HOME3")
print(f"[vio_gt] plan {plan['name']}: {len(P)} poses, {plan['duration_s']} s, v_max {plan['v_max_m_s']} m/s, "
      f"z_min {plan['z_min_m']} m", flush=True)
if pre or a.no_arm:
    print("[vio_gt] chequeo previo: " + ("; ".join(pre) if pre else "OK (sin armar: --no-arm)"), flush=True)
    n.destroy_node(); rclpy.shutdown()
    raise SystemExit(1 if pre else 0)

msg = call(arm_cli)
print(f"[vio_gt] arm: {msg}", flush=True)
if "servo mode" not in msg:
    print("[vio_gt] ABORTO: el puente no se armó en modo servo", flush=True)
    call(disarm_cli)
    raise SystemExit(2)

t0 = time.time()
i, last_print = 0, 0.0
while stop["reason"] is None:
    now = time.time()
    i = int((now - t0) * rate)
    if i >= len(P):
        stop["reason"] = "plan completo"
        break
    p = P[i]
    m = PoseStamped()
    m.header.stamp = n.get_clock().now().to_msg()
    m.header.frame_id = "link_base"
    m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, p[:3])
    m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = map(float, p[3:])
    pub.publish(m)
    cmd.append((now, i))
    t_next = t0 + (i + 1) / rate
    while time.time() < t_next:
        rclpy.spin_once(n, timeout_sec=max(0.0, min(0.01, t_next - time.time())))
        c = read_tcp()
        # Registro a tasa fija (también quieto: el controlador repite la misma pose)
        if c is not None and (not tcp or time.time() - tcp[-1][0] >= 0.01):
            tcp.append(np.r_[time.time(), c])
    if tcp:
        c = tcp[-1][1:]
        if c[2] < a.z_min:
            stop["reason"] = f"VIGILANTE: TCP z={c[2]:.3f} < {a.z_min}"
        track = float(np.linalg.norm(c[:3] - P[i, :3]))
        if track > a.max_track_m and now - t0 > 1.0:
            stop["reason"] = f"VIGILANTE: TCP a {track * 1000:.0f} mm de la pose pedida"
    if errors:
        stop["reason"] = f"ERROR {errors[-1][1]}: {errors[-1][2]}"
    if "armed=True" not in status.get("s", "armed=True") and now - t0 > 2.0:
        stop["reason"] = f"el puente se desarmó: {status.get('s', '')[:90]}"
    if now - last_print > 5.0 and tcp:
        last_print = now
        c = tcp[-1]
        print(f"[vio_gt] t={now - t0:6.1f}/{plan['duration_s']} s  TCP=({c[1]:.3f}, {c[2]:.3f}, {c[3]:.3f})",
              flush=True)

dis = call(disarm_cli)
print(f"[vio_gt] fin: {stop['reason']} | disarm: {dis}", flush=True)
T = np.array(tcp) if tcp else np.zeros((0, 8))
C = np.array(cmd, float) if cmd else np.zeros((0, 2))
import os  # noqa: E402
os.makedirs(a.out_dir, exist_ok=True)
np.savez(f"{a.out_dir}/{plan['name']}.npz", tcp=T, cmd=C, t0=t0)
res = {"plan": plan["name"], "reason": stop["reason"], "disarm": dis, "t0_jetson": t0, "n_tcp": int(len(T)),
       "completed": stop["reason"] == "plan completo", "last_index": int(i), "errors": errors[-5:]}
if len(T):
    k = np.searchsorted(C[:, 0], T[:, 0]).clip(0, len(C) - 1)
    trk = np.linalg.norm(T[:, 1:4] - P[C[k, 1].astype(int), :3], axis=1)
    res.update(tcp_end=np.round(T[-1, 1:4], 4).tolist(), z_min=round(float(T[:, 3].min()), 4),
               track_p95_mm=round(float(np.percentile(trk, 95) * 1000), 1))
json.dump(res, open(f"{a.out_dir}/{plan['name']}.json", "w"))
print(json.dumps(res, ensure_ascii=False), flush=True)
n.destroy_node()
rclpy.shutdown()
