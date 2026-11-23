"""Registra durante N s: /xarm5_bridge/command_preview, /target_frame_raw, /policy/metrics/done_flag y el TCP medido."""
import json, sys, time
import numpy as np, rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
import tf2_ros
dur = float(sys.argv[1]) if len(sys.argv) > 1 else 25.0
rclpy.init(); n = Node("stageB_recorder")
buf = tf2_ros.Buffer(); tf2_ros.TransformListener(buf, n)
prev, raw, done, tcp = [], [], [], []
P = lambda m: [m.pose.position.x, m.pose.position.y, m.pose.position.z]  # noqa: E731
n.create_subscription(PoseStamped, "/xarm5_bridge/command_preview", lambda m: prev.append([time.time()] + P(m)), 50)
n.create_subscription(PoseStamped, "/target_frame_raw", lambda m: raw.append([time.time()] + P(m)), 50)
n.create_subscription(Float32, "/policy/metrics/done_flag", lambda m: done.append([time.time(), m.data]), 50)
t0 = time.time()
while time.time() - t0 < dur:
    rclpy.spin_once(n, timeout_sec=0.02)
    try:
        t = buf.lookup_transform("link_base", "xarm_tcp", rclpy.time.Time())
        tcp.append([time.time(), t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
    except Exception:
        pass
res = {}
for name, arr in (("preview", prev), ("target_raw", raw)):
    a = np.array(arr)
    if len(a) > 2:
        res[name] = {"n": len(a), "rate_hz": round((len(a) - 1) / (a[-1, 0] - a[0, 0]), 2),
                     "first_xyz": np.round(a[0, 1:], 4).tolist(), "last_xyz": np.round(a[-1, 1:], 4).tolist(),
                     "mean_xyz": np.round(a[:, 1:].mean(0), 4).tolist()}
d = np.array(done)
if len(d):
    res["done_flag"] = {"n": len(d), "max": round(float(d[:, 1].max()), 4), "mean": round(float(d[:, 1].mean()), 4)}
tc = np.array(tcp)
if len(tc):
    res["tcp_measured_m"] = {"first": np.round(tc[0, 1:], 5).tolist(), "last": np.round(tc[-1, 1:], 5).tolist(),
                             "max_motion_mm": round(float(np.linalg.norm(tc[:, 1:] - tc[0, 1:], axis=1).max() * 1000), 2)}
print(json.dumps(res))
open("/tmp/vtrials/dry_run_record.json", "w").write(json.dumps(res, indent=2))
n.destroy_node(); rclpy.shutdown()
