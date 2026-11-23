"""Etapa C, chequeo previo (solo lectura): puente en modo real y desarmado, sin errores, frenos liberados, TCP en
HOME3, cámara publicando y ninguna política corriendo. Sale con código 0 solo si todo pasa."""
import sys, time
import numpy as np, rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
import tf2_ros
from xarm_msgs.msg import RobotMsg
HOME3 = np.array([0.15005, -0.00004, 0.75001])
rclpy.init(); n = Node("stageC_preflight")
buf = tf2_ros.Buffer(); tf2_ros.TransformListener(buf, n)
d = {"img": 0}
n.create_subscription(String, "/xarm5_bridge/status", lambda m: d.__setitem__("status", m.data), 5)
n.create_subscription(RobotMsg, "/xarm/robot_states", lambda m: d.__setitem__("rs", m), 5)
n.create_subscription(Image, "/camera/camera/color/image_raw", lambda m: d.__setitem__("img", d["img"] + 1), 5)
t0 = time.time()
while time.time() - t0 < 4.0:
    rclpy.spin_once(n, timeout_sec=0.05)
checks = []
st = d.get("status", "")
checks.append(("puente en modo REAL (dry_run=False)", "dry_run=False" in st))
checks.append(("puente desarmado", "armed=False" in st))
checks.append(("sin error del controlador (err=0)", " err=0" in st))
rs = d.get("rs")
checks.append(("frenos liberados y motores habilitados (31/31)", rs is not None and int(rs.mt_brake) == 31 and int(rs.mt_able) == 31))
try:
    t = buf.lookup_transform("link_base", "xarm_tcp", rclpy.time.Time())
    p = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
    dist = float(np.linalg.norm(p - HOME3) * 1000)
    checks.append((f"TCP en HOME3 ({dist:.1f} mm)", dist < 2.0))
except Exception as e:
    checks.append((f"TCP en HOME3 (TF no disponible: {e})", False))
checks.append((f"cámara publicando ({d['img']} imágenes en 4 s)", d["img"] >= 60))
pubs = n.count_publishers("/target_frame_raw")
checks.append((f"ninguna política corriendo ({pubs} publicadores de /target_frame_raw)", pubs == 0))
for name, ok in checks:
    print(("OK   " if ok else "FALLA") + "  " + name)
print("estado:", st[:150])
n.destroy_node(); rclpy.shutdown()
sys.exit(0 if all(ok for _, ok in checks) else 1)
