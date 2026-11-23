"""Lectura (sin mover nada): frame de la cámara, TF link_base->xarm_tcp, juntas y estado del xArm en HOME3."""
import json, math, time
import numpy as np, cv2, rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import tf2_ros
try:
    from xarm_msgs.msg import RobotMsg
except Exception:
    RobotMsg = None
rclpy.init()
n = Node("home3_reader")
buf = tf2_ros.Buffer(); tf2_ros.TransformListener(buf, n)
data = {}
def img_cb(m):
    data["img"] = m
n.create_subscription(Image, "/camera/camera/color/image_raw", img_cb, 2)
if RobotMsg is not None:
    n.create_subscription(RobotMsg, "/xarm/robot_states", lambda m: data.__setitem__("rs", m), 5)
t0 = time.time()
while time.time() - t0 < 6.0:
    rclpy.spin_once(n, timeout_sec=0.1)
out = {}
m = data.get("img")
if m is not None:
    img = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, -1)
    cv2.imwrite("/tmp/vtrials/pre.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    out["image"] = f"{m.width}x{m.height} {m.encoding}"
try:
    t = buf.lookup_transform("link_base", "xarm_tcp", rclpy.time.Time())
    p, q = t.transform.translation, t.transform.rotation
    out["tcp_xyz_m"] = [round(p.x, 5), round(p.y, 5), round(p.z, 5)]
    out["tcp_quat_xyzw"] = [round(v, 5) for v in (q.x, q.y, q.z, q.w)]
except Exception as e:
    out["tf_error"] = str(e)[:120]
rs = data.get("rs")
if rs is not None:
    out["joints_deg"] = [round(math.degrees(float(a)), 2) for a in list(rs.angle)[:5]]
    out["controller_pose"] = [round(float(v), 4) for v in list(rs.pose)[:6]]
    out["state_mode_err_warn"] = [int(rs.state), int(rs.mode), int(rs.err), int(rs.warn)]
    out["mt_brake_able"] = [int(rs.mt_brake), int(rs.mt_able)]
print(json.dumps(out))
n.destroy_node(); rclpy.shutdown()
