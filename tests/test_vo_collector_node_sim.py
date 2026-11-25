"""Integración del recolector v2 con una cámara y una odometría simuladas (ROS 2, dominio aislado).

Una cámara falsa publica a 30 Hz la imagen, la odometría con el MISMO sello (como RTAB-Map), ``odom_info`` y el
IMU, y ofrece ``/reset_odom``. Se comprueba que:
* el recolector graba en una rejilla regular de 20 Hz con sellos;
* reinicia la odometría;
* etiqueta done solo en el tramo quieto;
* rechaza un episodio con poses nulas, salvo que se fuerce el guardado.
v3 (segunda prueba): la cámara falsa muestra los ArUco de la mesa como en HOME3 (o corridos). Se comprueba que la
guía de inicio no deja empezar con la vista corrida, que el inicio automático arranca con la vista alineada y quieta,
que el guardado automático guarda al quedarse quieta, y que el episodio guarda ``start_view`` y la entrada del programa.
Requiere ``source /opt/ros/jazzy/setup.bash && source install/setup.bash``; si no, se omite.
"""

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("ROS_DOMAIN_ID", "87")
os.environ["ROS_LOCALHOST_ONLY"] = "1"
# Igual que los scripts de recolección (run_vo_container.sh). Con FastDDS, el suscriptor de imágenes en Python
# perdió ráfagas de ~300 ms en esta simulación (el QA las marca, pero la prueba no sería determinista).
os.environ["RMW_IMPLEMENTATION"] = "rmw_cyclonedds_cpp"
try:
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rtabmap_msgs.msg import OdomInfo
    from sensor_msgs.msg import Image, Imu
    from std_srvs.srv import Empty
    sys.path.insert(0, str(ROOT / "visuomotor_data_collection"))
    from visuomotor_data_collection.vo_collector_node import RECORDING, VOCollectorNode  # noqa: E402
    HAVE_ROS = True
except Exception:  # noqa: BLE001
    HAVE_ROS = False


if HAVE_ROS:
    class FakeCamera(Node):
        """Cámara + odometría falsas. Corre en OTRO proceso (spawn) con un bucle de 30 Hz por plazos: un timer de Python en
        el mismo proceso que el recolector tenía huecos de hasta ~290 ms, y una D435i real es regular."""

        def __init__(self, flags):
            super().__init__("fake_d435i")
            self.f = flags
            self.pub_img = self.create_publisher(Image, "/camera/camera/color/image_raw", 10)
            self.pub_odom = self.create_publisher(Odometry, "/rtabmap/odom", 50)
            self.pub_info = self.create_publisher(OdomInfo, "/odom_info", 50)
            self.pub_imu = self.create_publisher(Imu, "/camera/camera/imu", 50)
            self.create_service(Empty, "/reset_odom", self._reset)
            self.views = None
            if flags["img_mode"].value:                       # vista de HOME3 (1) o corrida 40 px (2), 640x480
                from test_vo_start_guide import REF_YAML, _render, _transform
                from visuomotor_data_collection.vo_start_guide import ViewReference
                ref = ViewReference.load(REF_YAML)
                self.views = {1: _render(ref.markers), 2: _render(_transform(ref.markers, t=(40.0, 0.0)))}

        def _reset(self, _req, res):
            self.f["x"].value = 0.0
            self.f["resets"].value += 1
            return res

        def publish_frame(self):
            f = self.f
            stamp = self.get_clock().now().to_msg()
            img = Image()
            img.header.stamp, img.header.frame_id = stamp, "camera_color_optical_frame"
            if self.views is not None:
                im = self.views[f["img_mode"].value]
                img.height, img.width, img.encoding, img.step = im.shape[0], im.shape[1], "rgb8", im.shape[1] * 3
                img.data = im.tobytes()
            else:
                img.height, img.width, img.encoding, img.step = 48, 64, "rgb8", 64 * 3
                img.data = bytes(np.full((48, 64, 3), int(f["x"].value * 1000) % 256, np.uint8).tobytes())
            self.pub_img.publish(img)
            if f["moving"].value:
                f["x"].value += 0.05 / 30.0
            od = Odometry()
            od.header.stamp, od.header.frame_id = stamp, "odom"
            if f["null"].value:                               # odometría nula de RTAB-Map al perder el rastreo
                od.pose.covariance = [9999.0 if i % 7 == 0 else 0.0 for i in range(36)]
            else:
                od.pose.pose.position.x = 0.3 + f["x"].value
                od.pose.pose.orientation.y = 0.7071068
                od.pose.pose.orientation.w = 0.7071068
                od.pose.covariance = [1e-4 if i % 7 == 0 else 0.0 for i in range(36)]
            self.pub_odom.publish(od)
            info = OdomInfo()
            info.header.stamp, info.lost = stamp, bool(f["null"].value)
            info.inliers = 0 if f["null"].value else 250
            self.pub_info.publish(info)
            imu = Imu()
            imu.header.stamp, imu.header.frame_id = stamp, "camera_imu_optical_frame"
            imu.linear_acceleration.y = 9.81
            imu.angular_velocity.z = 0.4 if f["moving"].value else 0.0   # la mano gira al moverse
            self.pub_imu.publish(imu)

    def _camera_process(flags):
        rclpy.init()
        node = FakeCamera(flags)
        threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
        t_next = time.monotonic()
        while not flags["stop"].value:
            node.publish_frame()
            t_next += 1.0 / 30.0
            time.sleep(max(0.0, t_next - time.monotonic()))
        node.destroy_node()
        rclpy.try_shutdown()


@unittest.skipUnless(HAVE_ROS, "ROS 2 / rtabmap_msgs no disponibles (source install/setup.bash)")
class CollectorSimTest(unittest.TestCase):
    def test_record_reset_done_and_qa(self):
        import zarr
        import multiprocessing as mp
        ctx = mp.get_context("spawn")   # fork tras usar DDS en el mismo proceso puede colgar al hijo
        flags = {k: ctx.Value("b", 0) for k in ("moving", "null", "stop", "img_mode")}
        flags["x"], flags["resets"] = ctx.Value("d", 0.0), ctx.Value("i", 0)
        proc = ctx.Process(target=_camera_process, args=(flags,), daemon=True)
        proc.start()
        rclpy.init()
        tmp = tempfile.mkdtemp()
        try:
            col = _make_collector(tmp, ["-p", "start_guide_enabled:=false"])
            ex = MultiThreadedExecutor(num_threads=4)
            ex.add_node(col)
            th = threading.Thread(target=ex.spin, daemon=True)
            th.start()
            self.assertTrue(_wait(lambda: col.poses.latest() is not None and len(col.images) > 5, 8.0))

            # Episodio 1: correcto
            self.assertTrue(col._start())
            self.assertTrue(_wait(lambda: col.state == RECORDING, 4.0))
            self.assertEqual(flags["resets"].value, 1)
            flags["moving"].value = 1
            time.sleep(2.0)
            flags["moving"].value = 0
            col._hold()
            time.sleep(2.6)
            ok, msg = col._save()
            self.assertTrue(ok, msg)

            # Episodio 2: rastreo perdido a mitad -> QA lo rechaza; forzado queda marcado
            self.assertTrue(col._start())
            self.assertTrue(_wait(lambda: col.state == RECORDING, 4.0))
            flags["moving"].value = 1
            time.sleep(1.0)
            flags["null"].value = 1
            time.sleep(0.2)
            flags["null"].value = 0
            time.sleep(0.8)
            flags["moving"].value = 0
            col._hold()
            time.sleep(2.6)
            ok, msg = col._save()
            self.assertFalse(ok)
            self.assertIn("rastreo perdido", msg)
            ok, msg = col._save(force=True)
            self.assertTrue(ok, msg)
            ex.shutdown()

            root = zarr.open_group(str(Path(tmp) / "sim_demos.zarr"), mode="r")
            ep0, ep1 = root["episode_0"], root["episode_1"]
            t = np.asarray(ep0["timestamp"])
            np.testing.assert_allclose(np.diff(t), 0.05, atol=1e-6)
            self.assertLessEqual(np.abs(np.asarray(ep0["image_stamp"]) - t).max(), 0.02)
            self.assertFalse(np.asarray(ep0["lost"]).any())
            st, ac = np.asarray(ep0["state"]), np.asarray(ep0["action"])
            np.testing.assert_allclose(st[0, :3], 0.0, atol=1e-9)
            self.assertGreater(st[-1, 0], 0.08)                            # ~0.10 m recorridos
            done = ac[:, 8] > 0.5
            self.assertGreaterEqual(done.sum(), 40)
            v = np.linalg.norm(np.diff(st[:, :3], axis=0), axis=1) * 20
            # done solo en frames quietos (el primer frame en HOLD puede incluir el último paso: latencia de 1 frame)
            self.assertLess(v[done[1:]][2:].max(), 0.005)
            a0 = dict(ep0.attrs)
            self.assertEqual(a0["collector_version"], 3)
            self.assertTrue(a0["odom_reset"])
            self.assertEqual(a0["qa_problems"], [])
            self.assertAlmostEqual(a0["start_gravity_mps2"][1], 9.81, places=3)
            a1 = dict(ep1.attrs)
            self.assertTrue(a1["qa_forced"])
            self.assertGreater(a1["quality"]["lost_frames"], 0)
            self.assertTrue(np.asarray(ep1["lost"]).any())
            q1 = np.asarray(ep1["state"])[:, 3:7]
            self.assertTrue(np.all(np.abs(q1 @ np.array([0, 0.7071068, 0, 0.7071068])) > 0.99))  # nunca identidad
        finally:
            flags["stop"].value = 1
            proc.join(timeout=5)
            if rclpy.ok():
                rclpy.shutdown()


@unittest.skipUnless(HAVE_ROS, "ROS 2 / rtabmap_msgs no disponibles (source install/setup.bash)")
class CollectorV3GuideTest(unittest.TestCase):
    def test_guide_auto_start_auto_save_schedule(self):
        import zarr
        import multiprocessing as mp
        sys.path.insert(0, str(ROOT / "tests"))
        from visuomotor_data_collection.vo_schedule import make_schedule, save_csv
        ctx = mp.get_context("spawn")   # fork tras usar DDS en el mismo proceso puede colgar al hijo
        flags = {k: ctx.Value("b", 0) for k in ("moving", "null", "stop", "img_mode")}
        flags["x"], flags["resets"] = ctx.Value("d", 0.0), ctx.Value("i", 0)
        flags["img_mode"].value = 2                                   # vista corrida
        proc = ctx.Process(target=_camera_process, args=(flags,), daemon=True)
        proc.start()
        rclpy.init()
        tmp = tempfile.mkdtemp()
        sched = Path(tmp) / "schedule.csv"
        save_csv(make_schedule([1, 5], [0.0, 90.0], reps=1, seed=3), sched)
        try:
            col = _make_collector(tmp, ["-p", "auto_start:=true", "-p", "auto_save:=true",
                                        "-p", "auto_start_hold_s:=0.6", "-p", "auto_start_cooldown_s:=3.0",
                                        "-p", "min_hold_duration_s:=1.0", "-p", "auto_save_margin_s:=0.4",
                                        "-p", "auto_save_min_travel_m:=0.08", "-p", "terminal_bell:=false",
                                        "-p", f"schedule_file:={sched}",
                                        # v3.1: bag crudo con pre-roll desde ARMING (cámara quieta)
                                        "-p", "raw_bag_enabled:=true", "-p", f"raw_bag_dir:={Path(tmp) / 'raw'}",
                                        "-p", "raw_bag_topics:=['/camera/camera/imu','/rtabmap/odom']"])
            first = col.schedule[0]
            ex = MultiThreadedExecutor(num_threads=4)
            ex.add_node(col)
            threading.Thread(target=ex.spin, daemon=True).start()
            self.assertTrue(_wait(lambda: col.guide_res is not None and col.poses.latest() is not None, 8.0))
            # Vista corrida: la guía no deja empezar y da la pista correcta
            self.assertTrue(_wait(lambda: col.guide_res[1]["error_px"] is not None, 4.0))
            res = col.guide_res[1]
            self.assertFalse(res["aligned"])
            self.assertTrue(any("DERECHA" in h for h in res["hints"]), res["hints"])
            self.assertFalse(col._start())
            self.assertEqual(col.state, "IDLE")
            # Vista de HOME3 y quieta: empieza sola (tras el reposo de 3 s desde el arranque del nodo)
            flags["img_mode"].value = 1
            self.assertTrue(_wait(lambda: col.state == RECORDING, 10.0))
            flags["img_mode"].value = 2                               # que no vuelva a empezar sola tras guardar
            flags["moving"].value = 1
            time.sleep(2.2)                                           # ~0,11 m
            flags["moving"].value = 0
            # Quieta al final: guarda sola tras min_hold + margen (1,4 s)
            self.assertTrue(_wait(lambda: col.storage.get_num_episodes() == 1 and col.state == "IDLE", 6.0))
            ex.shutdown()
            ep = zarr.open_group(str(Path(tmp) / "sim_demos.zarr"), mode="r")["episode_0"]
            a = dict(ep.attrs)
            self.assertEqual(a["collector_version"], 3)
            self.assertTrue(a["start_view"]["aligned"])
            self.assertEqual(a["start_view"]["source"], "auto")
            self.assertFalse(a["start_view"]["override"])
            self.assertLess(a["start_view"]["error_px"], 1.5)
            self.assertEqual(a["save_trigger"], "auto")
            self.assertTrue(a["hold_detected_auto"])
            self.assertEqual(a["qa_problems"], [])
            self.assertEqual(a["schedule"]["index"], first.index)
            self.assertEqual(a["object_cell"], first.cell)
            self.assertGreater(a["travel_m"], 0.08)
            self.assertGreater(a["max_speed_m_s"], 0.03)
            self.assertIn(first.index, col.sched_done)
            done = np.asarray(ep["action"])[:, 8] > 0.5
            self.assertGreaterEqual(done.sum(), 20)                  # >= 1,0 s a 20 Hz
            # Pre-roll: el bag empieza en ARMING, antes del primer frame del episodio (ventana quieta para el anclaje)
            self.assertEqual(a["raw_bag_starts_at"], "arming")
            sys.path.insert(0, str(ROOT / "tools/hardware"))
            from vio_bag_utils import iter_messages, stamp_of
            stamps = [stamp_of(m) for _, m, _ in iter_messages(a["raw_bag"], ["/camera/camera/imu"])]
            t0 = float(np.asarray(ep["timestamp"])[0])
            self.assertGreater(t0 - min(stamps), 0.15, "el bag no tiene la ventana quieta de ARMING")
        finally:
            flags["stop"].value = 1
            proc.join(timeout=5)
            if rclpy.ok():
                rclpy.shutdown()


    def test_lost_at_end_saves_by_imu_without_blocking(self):
        """v8 a mano: la odometría en vivo se pierde al bajar cerca de la mesa (pose congelada). No debe guardar
        mientras la mano se mueve; guarda cuando el giróscopo dice quieto, con el done desde ahí, y la pérdida queda
        como aviso (qa_tracking_blocking: false), porque las poses salen después de DROID desde el bag crudo."""
        import zarr
        import multiprocessing as mp
        sys.path.insert(0, str(ROOT / "tests"))
        ctx = mp.get_context("spawn")
        flags = {k: ctx.Value("b", 0) for k in ("moving", "null", "stop", "img_mode")}
        flags["x"], flags["resets"] = ctx.Value("d", 0.0), ctx.Value("i", 0)
        flags["img_mode"].value = 2
        proc = ctx.Process(target=_camera_process, args=(flags,), daemon=True)
        proc.start()
        rclpy.init()
        tmp = tempfile.mkdtemp()
        try:
            col = _make_collector(tmp, ["-p", "auto_start:=true", "-p", "auto_save:=true",
                                        "-p", "auto_start_hold_s:=0.6", "-p", "auto_start_cooldown_s:=3.0",
                                        "-p", "min_hold_duration_s:=1.0", "-p", "auto_save_margin_s:=0.4",
                                        "-p", "auto_save_min_travel_m:=0.05", "-p", "terminal_bell:=false",
                                        "-p", "qa_tracking_blocking:=false"])
            ex = MultiThreadedExecutor(num_threads=4)
            ex.add_node(col)
            threading.Thread(target=ex.spin, daemon=True).start()
            self.assertTrue(_wait(lambda: col.guide_res is not None and col.poses.latest() is not None, 8.0))
            flags["img_mode"].value = 1
            self.assertTrue(_wait(lambda: col.state == RECORDING, 10.0))
            flags["img_mode"].value = 2
            flags["moving"].value = 1
            time.sleep(1.4)                                          # ~7 cm con pose válida
            flags["null"].value = 1                                  # se pierde la odometría... y la mano sigue
            time.sleep(1.6)
            self.assertEqual(col.storage.get_num_episodes(), 0, "guardó con la pose congelada mientras se movía")
            t_stop = time.time()
            flags["moving"].value = 0                                # quieta, todavía perdida
            self.assertTrue(_wait(lambda: col.storage.get_num_episodes() == 1 and col.state == "IDLE", 6.0))
            ex.shutdown()
            ep = zarr.open_group(str(Path(tmp) / "sim_demos.zarr"), mode="r")["episode_0"]
            a = dict(ep.attrs)
            self.assertEqual(a["done_source"], "imu")
            self.assertEqual(a["qa_problems"], [])
            self.assertTrue(any("rastreo perdido" in w for w in a["qa_warnings"]), a["qa_warnings"])
            ts = np.asarray(ep["timestamp"])
            done = np.asarray(ep["action"])[:, 8] > 0.5
            self.assertGreaterEqual(done.sum(), 20)
            self.assertGreater(ts[done][0], t_stop - 0.35, "el done empieza antes de que la mano se detuviera")
        finally:
            flags["stop"].value = 1
            proc.join(timeout=5)
            if rclpy.ok():
                rclpy.shutdown()


def _make_collector(tmp, extra=()):
    """Recolector con parámetros de prueba (dataset temporal) sin tocar el YAML."""
    ctx_args = ["--ros-args", "-p", f"dataset_path:={Path(tmp) / 'sim_demos.zarr'}", "-p", f"repo_root:={ROOT}",
                "-p", "image_size:=32", *extra]
    # Los parámetros se pasan como argumentos de línea de comandos del nodo (sin tocar el YAML)
    orig = Node.__init__

    def patched(self, name, **kw):
        kw.setdefault("cli_args", ctx_args)
        orig(self, name, **kw)
    Node.__init__ = patched
    try:
        return VOCollectorNode()
    finally:
        Node.__init__ = orig


def _wait(cond, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.05)
    return False


if __name__ == "__main__":
    unittest.main()
