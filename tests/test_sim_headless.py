#!/usr/bin/env python3
"""
test_sim_headless.py — Suite de pruebas unitarias para infraestructura Headless
=============================================================================
Pruebas exhaustivas de:
1. Resolución determinista de rutas de stages USD (incluyendo mapeo Host <-> Docker).
2. Lógica de reajuste en memoria de RenderProducts y OmniGraph nodes por nombre y node:type.
3. Protocolo TCP cliente-servidor (Micro-Daemon y Cliente).
4. Terminación limpia de grupos de procesos y desmonte de recursos en stop_sim_process.
5. Validación sintáctica y argumentos CLI del orquestador run_headless_system.sh.
"""

import os
import sys
import json
import time
import signal
import socket
import unittest
import threading
import subprocess
from unittest.mock import MagicMock

# Añadir ruta de tools a sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
TOOLS_DIR = os.path.join(REPO_DIR, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from sim.run_headless_stage import resolve_stage_path, resize_render_products_in_memory
from sim.sim_client import send_command
import sim.isaac_sim_daemon as daemon_mod


class TestHeadlessStageRunner(unittest.TestCase):
    def test_resolve_existing_stage(self):
        # new_world_3.usda existe en robo_imitate
        stage = "new_world_3.usda"
        resolved = resolve_stage_path(stage)
        self.assertTrue(os.path.isabs(resolved))
        self.assertTrue(os.path.isfile(resolved))
        self.assertTrue(resolved.endswith("new_world_3.usda"))

    def test_resolve_object_picking_stage(self):
        stage = "object_picking.usda"
        resolved = resolve_stage_path(stage)
        self.assertTrue(os.path.isabs(resolved))
        self.assertTrue(os.path.isfile(resolved))
        self.assertTrue(resolved.endswith("object_picking.usda"))

    def test_resolve_fallback_nonexistent(self):
        stage = "non_existent_stage_xyz123.usda"
        resolved = resolve_stage_path(stage)
        self.assertEqual(resolved, stage)

    def test_resolve_cross_container_paths(self):
        # 1. Ruta absoluta de Host
        host_path = "/home/hackbrian/Documents/gits/robo_imitate/xarm_bringup/isaac/new_world_3.usda"
        resolved_host = resolve_stage_path(host_path)
        self.assertTrue(os.path.isfile(resolved_host))

        # 2. Ruta simulada de contenedor (debe mapear al archivo físico en Host)
        cont_path = "/workspace/robo_imitate/xarm_bringup/isaac/new_world_3.usda"
        resolved_cont = resolve_stage_path(cont_path)
        self.assertTrue(os.path.isfile(resolved_cont))

    def test_resize_render_products_mock_stage(self):
        # Mock de un stage USD con 2 prims: uno RenderProduct y uno OmniGraphNode por nombre
        mock_stage = MagicMock()

        # Prim 1: RenderProduct
        prim1 = MagicMock()
        prim1.IsA.return_value = False
        prim1.GetTypeName.return_value = "RenderProduct"
        prim1.HasAttribute.return_value = False
        res_attr = MagicMock()
        res_attr.IsValid.return_value = True
        res_attr.Get.return_value = (1280, 720)
        prim1.GetAttribute.return_value = res_attr
        prim1.GetPath.return_value = "/Render/OmniverseKit/HydraTextures/Replicator_01"

        # Prim 2: OmniGraphNode camera
        prim2 = MagicMock()
        prim2.IsA.return_value = False
        prim2.GetTypeName.return_value = "OmniGraphNode"
        prim2.GetName.return_value = "ros2_camera_helper_01"

        def prim2_has_attr(name):
            return name in ("inputs:width", "inputs:height")

        prim2.HasAttribute.side_effect = prim2_has_attr

        w_attr = MagicMock()
        w_attr.IsValid.return_value = True
        w_attr.Get.return_value = 1280
        h_attr = MagicMock()
        h_attr.IsValid.return_value = True
        h_attr.Get.return_value = 720

        def prim2_get_attr(name):
            if name == "inputs:width":
                return w_attr
            elif name == "inputs:height":
                return h_attr
            return None

        prim2.GetAttribute.side_effect = prim2_get_attr
        prim2.GetPath.return_value = "/World/ros2_graph/ros2_camera_helper_01"

        mock_stage.Traverse.return_value = [prim1, prim2]

        count = resize_render_products_in_memory(mock_stage, target_w=256, target_h=256)
        self.assertEqual(count, 2)
        mock_stage.Save.assert_not_called()

    def test_resize_render_products_by_node_type(self):
        # Prim con nombre genérico pero con node:type="isaacsim.core.nodes.IsaacCreateRenderProduct"
        mock_stage = MagicMock()

        prim = MagicMock()
        prim.IsA.return_value = False
        prim.GetTypeName.return_value = "OmniGraphNode"
        prim.GetName.return_value = "custom_pipeline_node_42"

        def prim_has_attr(name):
            return name in ("inputs:width", "inputs:height", "node:type")

        prim.HasAttribute.side_effect = prim_has_attr

        w_attr = MagicMock()
        w_attr.IsValid.return_value = True
        w_attr.Get.return_value = 1920
        h_attr = MagicMock()
        h_attr.IsValid.return_value = True
        h_attr.Get.return_value = 1080

        node_type_attr = MagicMock()
        node_type_attr.IsValid.return_value = True
        node_type_attr.Get.return_value = "isaacsim.core.nodes.IsaacCreateRenderProduct"

        def prim_get_attr(name):
            if name == "inputs:width":
                return w_attr
            elif name == "inputs:height":
                return h_attr
            elif name == "node:type":
                return node_type_attr
            return None

        prim.GetAttribute.side_effect = prim_get_attr
        prim.GetPath.return_value = "/World/ros2_graph/custom_pipeline_node_42"

        mock_stage.Traverse.return_value = [prim]

        count = resize_render_products_in_memory(mock_stage, target_w=256, target_h=256)
        self.assertEqual(count, 1)
        w_attr.Set.assert_called_with(256)
        h_attr.Set.assert_called_with(256)


class TestDaemonTCPProtocol(unittest.TestCase):
    TEST_PORT = 19999
    server_socket = None
    server_thread = None
    running = False

    @classmethod
    def setUpClass(cls):
        # Iniciar servidor TCP simulando daemon_mod en puerto 19999
        cls.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        cls.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        cls.server_socket.bind(("127.0.0.1", cls.TEST_PORT))
        cls.server_socket.listen(5)
        cls.running = True

        def run_server():
            while cls.running:
                try:
                    cls.server_socket.settimeout(0.5)
                    conn, addr = cls.server_socket.accept()
                    daemon_mod.handle_client(conn, addr)
                except socket.timeout:
                    continue
                except Exception:
                    break

        cls.server_thread = threading.Thread(target=run_server)
        cls.server_thread.daemon = True
        cls.server_thread.start()
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        cls.running = False
        if cls.server_socket:
            cls.server_socket.close()

    def test_ping(self):
        resp = send_command("ping", host="127.0.0.1", port=self.TEST_PORT)
        self.assertEqual(resp.get("status"), "pong")
        self.assertIn("time", resp)

    def test_status_idle(self):
        resp = send_command("status", host="127.0.0.1", port=self.TEST_PORT)
        self.assertEqual(resp.get("status"), "idle")
        self.assertIsNone(resp.get("pid"))

    def test_stop_when_not_running(self):
        resp = send_command("stop", host="127.0.0.1", port=self.TEST_PORT)
        self.assertEqual(resp.get("status"), "not_running")

    def test_connection_refused(self):
        # Puerto aleatorio cerrado
        resp = send_command("ping", host="127.0.0.1", port=19998)
        self.assertIn("error", resp)
        self.assertIn("No se pudo conectar", resp["error"])

    def test_stop_sim_process_with_active_proc(self):
        # Iniciar un subproceso real en su propio grupo de procesos
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            preexec_fn=os.setsid,
        )
        self.assertIsNone(proc.poll())

        daemon_mod.current_proc = proc
        daemon_mod.current_stage = "test_stage.usda"

        stopped = daemon_mod.stop_sim_process()
        self.assertTrue(stopped)
        self.assertIsNone(daemon_mod.current_proc)
        self.assertIsNone(daemon_mod.current_stage)
        # Verificar que el subproceso efectivamente murió
        self.assertIsNotNone(proc.poll())


class TestSystemScriptIntegrity(unittest.TestCase):
    def test_script_syntax_and_permissions(self):
        script_path = os.path.join(TOOLS_DIR, "sim", "run_headless_system.sh")
        self.assertTrue(os.path.isfile(script_path))
        self.assertTrue(os.access(script_path, os.X_OK))

        # Validación sintáctica con bash -n
        res = subprocess.run(["bash", "-n", script_path], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Error de sintaxis en bash: {res.stderr}")

    def test_script_help_execution(self):
        script_path = os.path.join(TOOLS_DIR, "sim", "run_headless_system.sh")
        res = subprocess.run([script_path, "--help"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("Lanzador Maestro Headless", res.stdout)
        self.assertIn("sim-only", res.stdout)
        self.assertIn("collect", res.stdout)
        self.assertIn("rl", res.stdout)


if __name__ == "__main__":
    unittest.main()
