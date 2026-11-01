#!/usr/bin/env python3
"""
isaac_sim_daemon.py — Micro-Daemon TCP Inter-Contenedor para Isaac Sim
======================================================================
Servicio en segundo plano que corre dentro de 'isaac-sim-6.0-wayland'
escuchando en 127.0.0.1:9999 (gracias a NetworkMode: host).

Permite que el contenedor 'visuomotor_ros2' envíe órdenes remotas:
- start (inicia simulación headless)
- stop (detiene la simulación y libera la GPU)
- status (consulta el estado actual y PID)
- ping (prueba de conectividad básica)
"""

import os
import sys
import json
import time
import socket
import signal
import subprocess
import threading

HOST = "127.0.0.1"
PORT = 9999
BUFFER_SIZE = 4096

current_proc = None
current_stage = None
current_log_file = None
proc_lock = threading.Lock()
daemon_running = True


def resolve_python_runner():
    """Determina el comando ejecutor de Python en Isaac Sim."""
    candidates = [
        "/isaac-sim/python.sh",
        os.path.join(os.getcwd(), "python.sh"),
        sys.executable,
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return sys.executable


def resolve_script_path():
    """Determina la ruta absoluta del script run_headless_stage.py."""
    candidates = [
        "/workspace/robo_imitate/xarm_bringup/isaac/run_headless_stage.py",
        "/workspace/robo_imitate/xarm_bringup/isaac/tools/sim/run_headless_stage.py",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_headless_stage.py"),
        "/home/hackbrian/Documents/gits/robo_imitate/xarm_bringup/isaac/run_headless_stage.py",
        "/home/hackbrian/Documents/gits/visuomotor_ros2/tools/sim/run_headless_stage.py",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return candidates[0]


def is_proc_alive(proc):
    """Comprueba si un proceso subprocess.Popen sigue en ejecución."""
    if proc is None:
        return False
    return proc.poll() is None


def stop_sim_process():
    """
    Detiene el proceso de simulación activo y a todo su grupo de procesos (incluyendo
    el ejecutable kit subyacente), liberando al 100% los recursos de GPU.
    """
    global current_proc, current_stage, current_log_file
    with proc_lock:
        stopped = False
        if current_proc is not None and is_proc_alive(current_proc):
            pid = current_proc.pid
            print(f"[Daemon] Enviando SIGINT a grupo de procesos de simulación PID {pid}...")
            try:
                os.killpg(pid, signal.SIGINT)
            except Exception:
                current_proc.send_signal(signal.SIGINT)

            try:
                current_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"[Daemon] Tiempo expirado. Enviando SIGKILL a grupo PID {pid}...")
                try:
                    os.killpg(pid, signal.SIGKILL)
                except Exception:
                    current_proc.kill()
                try:
                    current_proc.wait(timeout=2)
                except Exception:
                    pass

            stopped = True

        # Limpieza residual infalible para procesos huérfanos kit
        try:
            subprocess.run(["pkill", "-f", "kit"], capture_output=True)
        except Exception:
            pass

        if current_log_file is not None:
            try:
                current_log_file.close()
            except Exception:
                pass
            current_log_file = None

        current_proc = None
        current_stage = None
        return stopped


def handle_client(conn, addr):
    """Maneja las solicitudes entrantes de clientes TCP."""
    global current_proc, current_stage
    try:
        data = conn.recv(BUFFER_SIZE)
        if not data:
            return

        text = data.decode("utf-8").strip()
        try:
            req = json.loads(text)
        except json.JSONDecodeError:
            conn.sendall(json.dumps({"error": "JSON no válido"}).encode("utf-8"))
            return

        cmd = req.get("cmd", "")

        if cmd == "ping":
            resp = {"status": "pong", "time": time.time()}

        elif cmd == "status":
            with proc_lock:
                alive = is_proc_alive(current_proc)
                resp = {
                    "status": "running" if alive else "idle",
                    "pid": current_proc.pid if alive else None,
                    "stage": current_stage if alive else None,
                }

        elif cmd == "start":
            stage = req.get("stage", "new_world_3.usda")
            with proc_lock:
                if is_proc_alive(current_proc):
                    resp = {
                        "status": "already_running",
                        "pid": current_proc.pid,
                        "stage": current_stage,
                    }
                else:
                    runner = resolve_python_runner()
                    script = resolve_script_path()

                    env = os.environ.copy()
                    env["RMW_IMPLEMENTATION"] = "rmw_cyclonedds_cpp"
                    env["ROS_LOCALHOST_ONLY"] = "1"
                    env["ROS_DOMAIN_ID"] = "42"
                    env["OMNI_KIT_ACCEPT_EULA"] = "YES"

                    cmd_args = [runner, script, "--stage", stage]
                    print(f"[Daemon] Lanzando comando: {' '.join(cmd_args)}")

                    current_log_file = open("/tmp/isaac_headless.log", "w")
                    current_proc = subprocess.Popen(
                        cmd_args,
                        stdout=current_log_file,
                        stderr=subprocess.STDOUT,
                        env=env,
                        preexec_fn=os.setsid,
                    )
                    current_stage = stage
                    resp = {
                        "status": "started",
                        "pid": current_proc.pid,
                        "stage": stage,
                    }

        elif cmd == "stop":
            stopped = stop_sim_process()
            resp = {"status": "stopped" if stopped else "not_running"}

        else:
            resp = {"error": f"Comando desconocido: '{cmd}'"}

        conn.sendall(json.dumps(resp).encode("utf-8"))

    except Exception as exc:
        print(f"[Daemon] Error procesando solicitud: {exc}")
        try:
            conn.sendall(json.dumps({"error": str(exc)}).encode("utf-8"))
        except Exception:
            pass
    finally:
        conn.close()


def sig_daemon_handler(signum, frame):
    """Manejo de señales de cierre del daemon."""
    global daemon_running
    print(f"\n[Daemon] Señal {signum} recibida. Apagando micro-daemon...")
    daemon_running = False
    stop_sim_process()
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, sig_daemon_handler)
    signal.signal(signal.SIGTERM, sig_daemon_handler)

    print("==================================================================")
    print("      Isaac Sim TCP Micro-Daemon (Control Inter-Contenedor)       ")
    print("==================================================================")
    print(f"Escuchando en: {HOST}:{PORT}")
    print(f"PID del daemon: {os.getpid()}")
    print("==================================================================")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server.bind((HOST, PORT))
        server.listen(5)
    except Exception as e:
        print(f"[Daemon] [ERROR FATAL] No se pudo vincular socket {HOST}:{PORT}: {e}")
        sys.exit(1)

    while daemon_running:
        try:
            server.settimeout(1.0)
            conn, addr = server.accept()
            client_thread = threading.Thread(target=handle_client, args=(conn, addr))
            client_thread.daemon = True
            client_thread.start()
        except socket.timeout:
            continue
        except Exception as e:
            if daemon_running:
                print(f"[Daemon] Error en bucle accept(): {e}")
            break

    server.close()
    stop_sim_process()
    print("[Daemon] Servicio finalizado correctamente.")


if __name__ == "__main__":
    main()
