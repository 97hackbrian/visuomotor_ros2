#!/usr/bin/env python3
"""
sim_client.py — Cliente de Control Remoto Inter-Contenedor para Isaac Sim
========================================================================
Herramienta CLI para comunicarse con 'isaac_sim_daemon.py' desde adentro del
contenedor 'visuomotor_ros2' o desde el Host a través de 127.0.0.1:9999.

Permite arrancar, detener y monitorear Isaac Sim Headless sin requerir acceso
a /var/run/docker.sock.

Uso:
  python3 sim_client.py start [--stage new_world_3.usda] [--wait]
  python3 sim_client.py stop
  python3 sim_client.py status
  python3 sim_client.py ping
  python3 sim_client.py restart [--stage new_world_3.usda]
"""

import sys
import json
import time
import socket
import argparse

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9999
TIMEOUT_SEC = 5.0

# Colores terminales
GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
BOLD = "\033[1m"
NC = "\033[0m"


def send_command(cmd, stage=None, host=DEFAULT_HOST, port=DEFAULT_PORT):
    """Envía un comando JSON al micro-daemon y retorna la respuesta deserializada."""
    payload = {"cmd": cmd}
    if stage:
        payload["stage"] = stage

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(TIMEOUT_SEC)
    try:
        s.connect((host, port))
        msg = json.dumps(payload).encode("utf-8")
        s.sendall(msg)
        response_bytes = s.recv(4096)
        if not response_bytes:
            return {"error": "Respuesta vacía del daemon"}
        return json.loads(response_bytes.decode("utf-8"))
    except ConnectionRefusedError:
        return {
            "error": (
                f"No se pudo conectar a {host}:{port}. "
                "Asegúrate de que 'isaac_sim_daemon.py' esté corriendo en isaac-sim-6.0-wayland."
            )
        }
    except socket.timeout:
        return {"error": f"Tiempo de espera agotado ({TIMEOUT_SEC}s) conectando a {host}:{port}"}
    except Exception as exc:
        return {"error": f"Error de socket: {exc}"}
    finally:
        s.close()


def main():
    parser = argparse.ArgumentParser(
        description="Cliente de control para Isaac Sim Micro-Daemon (127.0.0.1:9999)"
    )
    parser.add_argument(
        "action",
        choices=["start", "stop", "status", "ping", "restart"],
        help="Acción a solicitar al daemon",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="new_world_3.usda",
        help="Nombre o ruta del stage a simular (defecto: new_world_3.usda)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help=f"Dirección IP del daemon (defecto: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Puerto TCP del daemon (defecto: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Espera activa tras 'start' hasta confirmar estado 'running'",
    )

    args = parser.parse_args()

    action = args.action

    if action == "restart":
        print(f"{CYAN}Solicitando detención de simulación previa...{NC}")
        res_stop = send_command("stop", host=args.host, port=args.port)
        time.sleep(1)
        action = "start"

    if action == "start":
        print(f"{CYAN}Solicitando arranque de stage: {BOLD}{args.stage}{NC}...")
        res = send_command("start", stage=args.stage, host=args.host, port=args.port)
        if "error" in res:
            print(f"{RED}[ERROR] {res['error']}{NC}")
            sys.exit(1)

        status = res.get("status", "")
        pid = res.get("pid", "desconocido")
        stage = res.get("stage", args.stage)

        if status == "already_running":
            print(f"{YELLOW}[AVISO] La simulación ya estaba corriendo (PID: {pid}, Stage: {stage}){NC}")
        elif status == "started":
            print(f"{GREEN}[OK] Simulación Headless iniciada con éxito (PID: {pid}, Stage: {stage}){NC}")

        if args.wait:
            print(f"{CYAN}Esperando confirmación activa del proceso...{NC}")
            for _ in range(15):
                time.sleep(1)
                st = send_command("status", host=args.host, port=args.port)
                if st.get("status") == "running":
                    print(f"{GREEN}✔ Simulación confirmada en ejecución activa.{NC}")
                    break
            else:
                print(f"{YELLOW}Aviso: No se confirmó el estado en los primeros 15 segundos.{NC}")

    elif action == "stop":
        print(f"{CYAN}Solicitando detención de la simulación...{NC}")
        res = send_command("stop", host=args.host, port=args.port)
        if "error" in res:
            print(f"{RED}[ERROR] {res['error']}{NC}")
            sys.exit(1)
        status = res.get("status", "")
        if status == "stopped":
            print(f"{GREEN}[OK] Simulación detenida y recursos GPU liberados.{NC}")
        else:
            print(f"{YELLOW}[INFO] No había ninguna simulación en ejecución.{NC}")

    elif action == "status":
        res = send_command("status", host=args.host, port=args.port)
        if "error" in res:
            print(f"{RED}[ERROR] {res['error']}{NC}")
            sys.exit(1)
        status = res.get("status", "desconocido")
        pid = res.get("pid")
        stage = res.get("stage")

        print("--------------------------------------------------")
        print(f"Estado de Simulación: {BOLD}{GREEN if status == 'running' else YELLOW}{status.upper()}{NC}")
        if pid:
            print(f"PID del Proceso:      {pid}")
        if stage:
            print(f"Stage Activo:         {stage}")
        print("--------------------------------------------------")

    elif action == "ping":
        res = send_command("ping", host=args.host, port=args.port)
        if "error" in res:
            print(f"{RED}[ERROR] {res['error']}{NC}")
            sys.exit(1)
        print(f"{GREEN}[OK] Micro-daemon responde: {res}{NC}")


if __name__ == "__main__":
    main()
