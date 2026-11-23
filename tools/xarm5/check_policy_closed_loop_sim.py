#!/usr/bin/env python3
"""Prueba de integración ROS 2 de la cadena de despliegue con el xArm5 SIMULADO (sin hardware, sin mover nada).

Cadena probada (la misma que en la Jetson):
    cámara (falsa: frames de un episodio real) -> policy_node (jetson_policy_params.yaml + checkpoint)
      -> target_frame_raw -> pose_smoother (xarm5_pose_smoother_params.yaml) -> target_frame
      -> xarm5_bridge (dry_run + simulate_robot_state: el "robot" sigue el comando con los límites de velocidad,
         volumen y orientación reales) -> TF link_base->xarm_tcp -> policy_node (estado observado)

El robot simulado arranca en HOME3. Las imágenes no reaccionan al movimiento (se reproduce el video de la
demostración en tiempo real), así que no mide éxito de tarea: verifica que la cadena funciona, que la dirección del
movimiento coincide con la demostración, que la orientación es válida y que el auto-stop se dispara.

Aislamiento: ROS_DOMAIN_ID propio + ROS_LOCALHOST_ONLY=1 (nunca interfiere con un brazo real en el dominio 42).

Uso (contenedor, con install/ compilado y xarm_msgs disponible, p. ej. `source /tmp/xarm_ws/install/setup.bash`):
    python3 tools/xarm5/check_policy_closed_loop_sim.py --checkpoint <dir> \
        --dataset datasets/demonstrations_real_v7_xarm5_v2.zarr --episode episode_7 --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--episode", default="episode_7")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--policy-params", default=str(REPO / "deploy/jetson/config/jetson_policy_params.yaml"))
    ap.add_argument("--bridge-params", default=str(REPO / "visuomotor_bringup/config/xarm5/xarm5_real_params.yaml"))
    ap.add_argument("--domain-id", type=int, default=77)
    ap.add_argument("--reset-s", type=float, default=4.0)
    ap.add_argument("--extra-s", type=float, default=4.0, help="segundos tras el final del video")
    args = ap.parse_args(argv)
    import zarr

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pol = yaml.safe_load(open(args.policy_params))["policy_node"]["ros__parameters"]
    home_xyz = [pol["home_x"], pol["home_y"], pol["home_z"]]
    home_rpy = [pol["home_roll"], pol["home_pitch"], pol["home_yaw"]]

    # Parámetros del puente: los reales, con el robot simulado arrancando en HOME3
    bcfg = yaml.safe_load(open(args.bridge_params))
    bcfg["xarm5_bridge"]["ros__parameters"].update({"sim_initial_xyz": home_xyz, "sim_initial_rpy": home_rpy,
                                                    "dry_run": True, "simulate_robot_state": True})
    bridge_yaml = out / "xarm5_sim_params.yaml"
    bridge_yaml.write_text(yaml.safe_dump(bcfg))

    env = dict(os.environ, ROS_DOMAIN_ID=str(args.domain_id), ROS_LOCALHOST_ONLY="1",
               ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST", PYTHONUNBUFFERED="1")
    os.environ.update({k: env[k] for k in ("ROS_DOMAIN_ID", "ROS_LOCALHOST_ONLY", "ROS_AUTOMATIC_DISCOVERY_RANGE")})
    procs = []
    t_video_start = float("inf")
    logs = {n: open(out / f"{n}.log", "w") for n in ("bringup", "policy")}
    procs.append(subprocess.Popen(
        ["ros2", "launch", "visuomotor_bringup", "xarm5_real.launch.py", f"params_file:={bridge_yaml}",
         "simulate_robot_state:=true", "dry_run:=true", "start_driver:=false", "start_rsp:=false",
         "start_teleop:=false"], env=env, stdout=logs["bringup"], stderr=subprocess.STDOUT, start_new_session=True))

    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image
        from geometry_msgs.msg import PoseStamped
        from std_srvs.srv import Trigger
        import tf2_ros

        rclpy.init()
        node = Node("closed_loop_check")
        img_pub = node.create_publisher(Image, pol["camera_topic"], 10)
        tf_buf = tf2_ros.Buffer()
        tf2_ros.TransformListener(tf_buf, node)
        raw_cmds = []
        # Los comandos se escuchan en un nodo y un hilo propios que giran sin pausa. El bucle principal procesa un
        # solo callback cada ~45 ms (spin_once + sleep), compartido con el TF a ~100 Hz: medía su propio ritmo
        # (16 Hz en el PC, 7 Hz en la Jetson) y perdía mensajes, no el ritmo real de la política.
        import threading
        from rclpy.executors import SingleThreadedExecutor
        cmd_node = Node("closed_loop_cmd_monitor")
        cmd_node.create_subscription(PoseStamped, "target_frame_raw", lambda m: raw_cmds.append(
            (time.time(), m.pose.position.x, m.pose.position.y, m.pose.position.z, m.pose.orientation.x,
             m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w)), 100)
        cmd_exec = SingleThreadedExecutor()
        cmd_exec.add_node(cmd_node)
        threading.Thread(target=cmd_exec.spin, daemon=True).start()

        root = zarr.open_group(args.dataset, mode="r")
        images = np.asarray(root[args.episode]["image"])
        demo = np.asarray(root[args.episode]["action"])

        def publish(frame):
            msg = Image()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.header.frame_id = "camera_color_optical_frame"
            msg.height, msg.width = frame.shape[:2]
            msg.encoding = "rgb8"
            msg.step = msg.width * 3
            msg.data = frame.tobytes()
            img_pub.publish(msg)

        def spin_for(sec, frame=None):
            t_end = time.time() + sec
            while time.time() < t_end:
                if frame is not None:
                    publish(frame)
                rclpy.spin_once(node, timeout_sec=0.02)

        # 1) Esperar al puente y armarlo (dry-run: no se envía nada a ningún robot)
        cli = node.create_client(Trigger, "/xarm5_bridge/arm")
        t0 = time.time()
        while not cli.wait_for_service(timeout_sec=0.5):
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.time() - t0 > 60:
                raise SystemExit("El puente no apareció en 60 s (ver bringup.log)")
        fut = cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(node, fut, timeout_sec=10)
        arm_msg = fut.result().message if fut.result() else "sin respuesta"
        print(f"[check] arm: {arm_msg}", flush=True)

        # 2) Lanzar la política (RESET corto) mientras la cámara muestra el primer frame
        procs.append(subprocess.Popen(
            ["ros2", "run", "visuomotor_ros2", "policy_node", "--ros-args", "--params-file", args.policy_params,
             "-p", f"checkpoint_path:={args.checkpoint}", "-p", f"reset_duration_s:={args.reset_s}"],
            env=env, stdout=logs["policy"], stderr=subprocess.STDOUT, start_new_session=True))
        # la RESET cuenta desde la construcción del nodo (carga del modelo incluida): esperar a que publique
        t0 = time.time()
        while not raw_cmds and time.time() - t0 < 120:
            spin_for(0.2, images[0])
        spin_for(args.reset_s + 1.0, images[0])

        # 3) Reproducir el video de la demostración a 20 Hz y registrar la TCP del robot simulado
        tcp = []
        t_start = time.time()
        t_video_start = t_start
        n = len(images)
        while True:
            k = int((time.time() - t_start) * 20.0)
            frame = images[min(k, n - 1)]
            publish(frame)
            rclpy.spin_once(node, timeout_sec=0.01)
            try:
                tr = tf_buf.lookup_transform("link_base", "xarm_tcp", rclpy.time.Time())
                p, q = tr.transform.translation, tr.transform.rotation
                tcp.append((time.time() - t_start, p.x, p.y, p.z, q.x, q.y, q.z, q.w))
            except Exception:
                pass
            time.sleep(0.03)
            if k >= n + args.extra_s * 20:
                break
    finally:
        for p in procs:
            try:
                os.killpg(p.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        time.sleep(3)
        for p in procs:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        try:
            cmd_exec.shutdown(timeout_sec=1.0)
            cmd_node.destroy_node()
        except Exception:
            pass
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass

    tcp = np.array(tcp)
    policy_log = (out / "policy.log").read_text(errors="ignore")
    stopped = "Auto-Stop Neuronal Activado" in policy_log
    # Las trazas provocadas por el SIGINT de apagado de esta prueba no son errores de la política
    blocks = policy_log.split("Traceback (most recent call last):")
    errors = [b.strip().splitlines()[-1] for b in blocks[1:]
              if not any(x in b for x in ("KeyboardInterrupt", "ExternalShutdownException", "RCLError"))][:10]
    errors += [l for l in blocks[0].splitlines() if "[ERROR]" in l][:10]
    bridge_log = (out / "bringup.log").read_text(errors="ignore")
    rejected = sum(1 for l in bridge_log.splitlines() if "rechaz" in l.lower() or "reject" in l.lower())
    disp = (tcp[-1, 1:4] - np.array(home_xyz)) if len(tcp) else np.zeros(3)
    demo_disp = demo[-1, :3]
    cos = float(disp @ demo_disp / max(1e-9, np.linalg.norm(disp) * np.linalg.norm(demo_disp)))
    qn = np.linalg.norm(tcp[:, 4:8], axis=1) if len(tcp) else np.array([1.0])
    # Ritmo real del lazo de la política mientras corre el video (hasta su último comando = auto-stop): con
    # inferencia síncrona cada n_action_steps ticks, una inferencia más lenta que el período salta ticks.
    cmd_t = np.array([c[0] for c in raw_cmds if c[0] >= t_video_start])
    gaps = np.diff(cmd_t) * 1000.0 if len(cmd_t) > 2 else np.zeros(0)
    timing = {"policy_command_rate_hz": float((len(cmd_t) - 1) / (cmd_t[-1] - cmd_t[0])) if len(cmd_t) > 2 else 0.0,
              "command_gap_ms_median": float(np.median(gaps)) if len(gaps) else None,
              "command_gap_ms_p95": float(np.percentile(gaps, 95)) if len(gaps) else None,
              "command_gap_ms_max": float(gaps.max()) if len(gaps) else None}
    res = {"episode": args.episode, "checkpoint": args.checkpoint, "arm": arm_msg,
           "tcp_samples": int(len(tcp)), "policy_commands": len(raw_cmds),
           "robot_displacement_m": disp.tolist(), "demo_displacement_m": demo_disp.tolist(),
           "displacement_cosine_similarity": cos, "robot_path_m": float(np.linalg.norm(np.diff(tcp[:, 1:4], axis=0), axis=1).sum()) if len(tcp) > 1 else 0.0,
           "neural_autostop": stopped, "bridge_rejections": rejected, "policy_errors": errors,
           "tcp_quat_norm_min": float(qn.min()), **timing}
    (out / "closed_loop_check.json").write_text(json.dumps(res, indent=2))
    np.save(out / "tcp_trace.npy", tcp)
    print(json.dumps(res, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C1, C2, INK = "#2a78d6", "#eb6834", "#52514e"
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"ROS 2 closed-loop integration check (simulated xArm5, {args.episode})", fontweight="bold")
    hx = np.array(home_xyz)
    for ax, (i, j, lab) in zip(axs, ((0, 1, "y"), (0, 2, "z"))):
        ax.plot(demo[:, i] + hx[i], demo[:, j] + hx[j], color=C1, lw=2, label="Demonstration (replayed video)")
        if len(tcp):
            ax.plot(tcp[:, 1 + i], tcp[:, 1 + j], color=C2, lw=2, label="Simulated robot TCP")
        ax.scatter([hx[i]], [hx[j]], marker="*", s=180, color=INK, label="HOME3")
        ax.set_xlabel("x (m)")
        ax.set_ylabel(f"{lab} (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, color="#d9d8d4", lw=0.6)
        for side in ("top", "right"):                   # compatible con matplotlib < 3.4 (Jetson)
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    fig.savefig(out / "closed_loop_check.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "closed_loop_check.pdf", bbox_inches="tight")
    return 0 if (len(tcp) and not errors) else 1


if __name__ == "__main__":
    sys.exit(main())
