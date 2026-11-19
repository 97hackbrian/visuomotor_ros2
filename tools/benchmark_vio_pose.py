#!/usr/bin/env python3
"""
tools/benchmark_vio_pose.py
---------------------------
Herramienta de metrología y benchmarking interactivo para VIO (RTAB-Map + RealSense D435i/D415).
Protocolo de Cierre de Bucle 3D (Loop Closure Test) para cuantificar deriva métrica,
error angular, jitter estático en reposo y consistencia de inliers.

Uso:
    python3 tools/benchmark_vio_pose.py --label "conf 0 (Baseline)"
"""

import os
import sys
import time
import json
import argparse
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from rtabmap_msgs.msg import OdomInfo

# Códigos ANSI para formato de consola
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def quaternion_to_euler_deg(qx, qy, qz, qw):
    """Convierte cuaternión a ángulos de Euler (Roll, Pitch, Yaw) en grados."""
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1.0:
        pitch = np.copysign(np.pi / 2.0, sinp)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.degrees([roll, pitch, yaw])

def quaternion_geodesic_distance_deg(q1, q2):
    """Calcula la distancia geodésica entre dos cuaterniones en grados (error angular en SO(3))."""
    dot = np.abs(np.dot(q1, q2))
    dot = np.clip(dot, -1.0, 1.0)
    angle_rad = 2.0 * np.arccos(dot)
    return float(np.degrees(angle_rad))

class VIOBenchmarkNode(Node):
    def __init__(self):
        super().__init__('vio_benchmark_node')
        
        # Buffer de telemetría continua
        self.lock = threading.Lock()
        self.current_pose = None      # [x, y, z]
        self.current_quat = None      # [qx, qy, qz, qw]
        self.current_inliers = 0
        self.current_features = 0
        self.current_lost = False
        self.current_fps = 0.0
        self.last_odom_time = None
        
        # Registro por fases
        self.active_buffer_positions = []
        self.active_buffer_quats = []
        self.active_buffer_times = []
        self.active_buffer_inliers = []
        self.active_buffer_lost = []
        
        self.is_recording = False
        
        # Suscripciones
        self.create_subscription(Odometry, '/rtabmap/odom', self._odom_callback, 10)
        self.create_subscription(OdomInfo, '/odom_info', self._info_callback, 10)

    def _odom_callback(self, msg: Odometry):
        now = time.perf_counter()
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        
        with self.lock:
            if self.last_odom_time is not None:
                dt = now - self.last_odom_time
                if dt > 0:
                    self.current_fps = 0.9 * self.current_fps + 0.1 * (1.0 / dt) if self.current_fps > 0 else (1.0 / dt)
            self.last_odom_time = now
            
            p = np.array([pos.x, pos.y, pos.z])
            q = np.array([ori.x, ori.y, ori.z, ori.w])
            norm_q = np.linalg.norm(q)
            if norm_q > 1e-6:
                q = q / norm_q
                
            self.current_pose = p
            self.current_quat = q
            
            if self.is_recording:
                self.active_buffer_positions.append(p)
                self.active_buffer_quats.append(q)
                self.active_buffer_times.append(now)

    def _info_callback(self, msg: OdomInfo):
        with self.lock:
            self.current_inliers = msg.inliers
            self.current_features = msg.features
            self.current_lost = bool(msg.lost)
            
            if self.is_recording:
                self.active_buffer_inliers.append(msg.inliers)
                self.active_buffer_lost.append(msg.lost)

    def start_recording(self):
        with self.lock:
            self.active_buffer_positions = []
            self.active_buffer_quats = []
            self.active_buffer_times = []
            self.active_buffer_inliers = []
            self.active_buffer_lost = []
            self.is_recording = True

    def stop_recording(self):
        with self.lock:
            self.is_recording = False
            return (
                np.array(self.active_buffer_positions),
                np.array(self.active_buffer_quats),
                np.array(self.active_buffer_times),
                np.array(self.active_buffer_inliers),
                np.array(self.active_buffer_lost)
            )

    def get_live_status(self):
        with self.lock:
            if self.current_pose is None:
                return None
            return {
                'pos': self.current_pose.copy(),
                'quat': self.current_quat.copy(),
                'inliers': self.current_inliers,
                'features': self.current_features,
                'lost': self.current_lost,
                'fps': self.current_fps
            }

def run_evaluation(label: str, static_duration: float = 3.0, history_file: str = "experiments/vio_benchmark_history.json"):
    # Asegurar directorio de logs con permisos de usuario independiente
    user_log_dir = f"/tmp/ros_log_{os.getuid()}"
    os.makedirs(user_log_dir, exist_ok=True)
    os.environ["ROS_LOG_DIR"] = user_log_dir

    rclpy.init()
    node = VIOBenchmarkNode()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    
    # Hilo en background para spin de ROS 2
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    print(f"\n{Colors.HEADER}{Colors.BOLD}╔═══════════════════════════════════════════════════════════════════════════╗{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}║           BENCHMARK DE ODOMETRÍA VISUAL-INERCIAL (VIO RTAB-MAP)           ║{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}║                  PROTOCOLO DE RETORNO Y CIERRE DE BUCLE                   ║{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}╚═══════════════════════════════════════════════════════════════════════════╝{Colors.ENDC}")
    print(f"Configuración bajo prueba: {Colors.OKCYAN}{Colors.BOLD}{label}{Colors.ENDC}\n")

    # 1. Esperar flujo de odometría
    print(f"[{Colors.OKBLUE}INFO{Colors.ENDC}] Esperando flujo en /rtabmap/odom y /odom_info...", end="", flush=True)
    t0 = time.time()
    while node.get_live_status() is None:
        if time.time() - t0 > 15.0:
            print(f"\n{Colors.FAIL}[ERROR] Timeout: No se reciben mensajes en /rtabmap/odom.{Colors.ENDC}")
            print("Verifique que el launch de VIO esté corriendo en la otra terminal.")
            rclpy.shutdown()
            sys.exit(1)
        time.sleep(0.1)
    print(f" {Colors.OKGREEN}CONECTADO ({node.get_live_status()['fps']:.1f} Hz){Colors.ENDC}\n")

    # ──────────────────────────────────────────────────────────────────────────
    # FASE 1: REPOSO INICIAL
    # ──────────────────────────────────────────────────────────────────────────
    print(f"{Colors.BOLD}[PASO 1 / 3] POSICIÓN DE REPOSO INICIAL (ORIGEN){Colors.ENDC}")
    print("  1. Coloque la cámara RealSense firmemente apoyada en la marca física de inicio.")
    print("  2. Asegúrese de que no vibre ni se mueva.")
    input(f"  --> Presione {Colors.OKGREEN}[ENTER]{Colors.ENDC} cuando la cámara esté en reposo sobre la marca...")

    print(f"  {Colors.OKBLUE}Midiendo estabilidad estática durante {static_duration} segundos...{Colors.ENDC}")
    node.start_recording()
    time.sleep(static_duration)
    pos_s1, quat_s1, times_s1, inliers_s1, lost_s1 = node.stop_recording()

    if len(pos_s1) < 10:
        print(f"{Colors.FAIL}[ERROR] Muestras insuficientes en fase estática ({len(pos_s1)}).{Colors.ENDC}")
        rclpy.shutdown()
        sys.exit(1)

    p_start = np.mean(pos_s1, axis=0)
    q_start = np.mean(quat_s1, axis=0)
    q_start = q_start / np.linalg.norm(q_start)

    std_s1 = np.std(pos_s1, axis=0) * 1000.0  # en mm
    jitter_pos_s1 = np.linalg.norm(std_s1)
    euler_s1 = np.array([quaternion_to_euler_deg(q[0], q[1], q[2], q[3]) for q in quat_s1])
    std_euler_s1 = np.std(euler_s1, axis=0)
    jitter_rot_s1 = np.linalg.norm(std_euler_s1)

    print(f"  {Colors.OKGREEN}✓ Pose inicial fijada:{Colors.ENDC} X={p_start[0]:.4f}m, Y={p_start[1]:.4f}m, Z={p_start[2]:.4f}m")
    print(f"  {Colors.OKGREEN}✓ Jitter estático inicial:{Colors.ENDC} σ_pos={Colors.BOLD}{jitter_pos_s1:.4f} mm{Colors.ENDC} (X={std_s1[0]:.3f}, Y={std_s1[1]:.3f}, Z={std_s1[2]:.3f} mm)")
    print(f"  {Colors.OKGREEN}✓ Jitter angular inicial:{Colors.ENDC}  σ_rot={jitter_rot_s1:.4f}°\n")

    # ──────────────────────────────────────────────────────────────────────────
    # FASE 2: TRAYECTORIA DINÁMICA DE BUCLE CERRADO
    # ──────────────────────────────────────────────────────────────────────────
    print(f"{Colors.BOLD}[PASO 2 / 3] TRAYECTORIA DINÁMICA (APROXIMACIÓN Y RETORNO){Colors.ENDC}")
    print("  1. Tome la cámara con la mano.")
    print("  2. Desplácela hacia adelante (~30 a 50 cm), puede realizar una leve rotación/aproximación.")
    print("  3. Regrésela y apóyela con cuidado EXACTAMENTE en la misma marca física inicial.")
    print(f"  --> Cuando esté de vuelta en la marca física, presione {Colors.OKGREEN}[ENTER]{Colors.ENDC}.\n")

    node.start_recording()
    
    stop_event = threading.Event()
    
    def wait_user_input():
        input()
        stop_event.set()

    input_thread = threading.Thread(target=wait_user_input, daemon=True)
    input_thread.start()

    max_dist = 0.0
    while not stop_event.is_set():
        status = node.get_live_status()
        if status is not None:
            dist = np.linalg.norm(status['pos'] - p_start)
            if dist > max_dist:
                max_dist = dist
            lost_str = f"{Colors.FAIL}[TRACKING PERDIDO]{Colors.ENDC}" if status['lost'] else f"{Colors.OKGREEN}[TRACKING OK]{Colors.ENDC}"
            sys.stdout.write(
                f"\r  {Colors.BOLD}EN VIVO:{Colors.ENDC} Excursión: {dist*100.0:5.1f} cm | "
                f"Máx alcanzada: {Colors.OKCYAN}{max_dist*100.0:5.1f} cm{Colors.ENDC} | "
                f"Inliers: {status['inliers']:3d} | FPS: {status['fps']:4.1f} | {lost_str}   "
            )
            sys.stdout.flush()
        time.sleep(0.08)
    
    print("\n")
    pos_dyn, quat_dyn, times_dyn, inliers_dyn, lost_dyn = node.stop_recording()

    # ──────────────────────────────────────────────────────────────────────────
    # FASE 3: REPOSO FINAL Y VERIFICACIÓN DE CIERRE
    # ──────────────────────────────────────────────────────────────────────────
    print(f"{Colors.BOLD}[PASO 3 / 3] VERIFICACIÓN ESTÁTICA DE RETORNO{Colors.ENDC}")
    print(f"  {Colors.OKBLUE}Midiendo estabilidad en reposo final durante {static_duration} segundos...{Colors.ENDC}")
    
    node.start_recording()
    time.sleep(static_duration)
    pos_s2, quat_s2, times_s2, inliers_s2, lost_s2 = node.stop_recording()

    p_end = np.mean(pos_s2, axis=0)
    q_end = np.mean(quat_s2, axis=0)
    q_end = q_end / np.linalg.norm(q_end)

    std_s2 = np.std(pos_s2, axis=0) * 1000.0
    jitter_pos_s2 = np.linalg.norm(std_s2)

    # ──────────────────────────────────────────────────────────────────────────
    # CÁLCULO DE MÉTRICAS MATEMÁTICAS
    # ──────────────────────────────────────────────────────────────────────────
    # Error de retorno en posición (Loop Closure Drift en mm)
    delta_p = (p_end - p_start) * 1000.0  # mm
    loop_error_3d = float(np.linalg.norm(delta_p))

    # Error angular en SO(3)
    loop_error_rot_deg = quaternion_geodesic_distance_deg(q_start, q_end)
    euler_start = quaternion_to_euler_deg(q_start[0], q_start[1], q_start[2], q_start[3])
    euler_end = quaternion_to_euler_deg(q_end[0], q_end[1], q_end[2], q_end[3])
    delta_euler = euler_end - euler_start

    # Trayectoria dinámica
    total_arc_length = 0.0
    if len(pos_dyn) > 1:
        diffs = np.diff(pos_dyn, axis=0)
        total_arc_length = float(np.sum(np.linalg.norm(diffs, axis=1)))

    mean_inliers = float(np.mean(inliers_dyn)) if len(inliers_dyn) > 0 else 0.0
    total_lost_frames = int(np.sum(lost_dyn)) if len(lost_dyn) > 0 else 0
    tracking_lost_pct = (total_lost_frames / max(1, len(lost_dyn))) * 100.0

    # Criterios de Aceptación UMI-Grade (RealSense D435i + IMU Bosch BMI085)
    # 1. Error de Retorno <= 3.0 mm (precisión milimétrica de cierre de bucle)
    # 2. Jitter en reposo <= 2.5 mm (ruido estéreo y cuantización MEMS en mesa de trabajo)
    pass_loop = loop_error_3d <= 3.0
    pass_jitter = jitter_pos_s1 <= 2.5
    pass_tracking = tracking_lost_pct == 0.0
    overall_pass = pass_loop and pass_jitter and pass_tracking

    # ──────────────────────────────────────────────────────────────────────────
    # IMPRESIÓN DEL REPORTE
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{Colors.HEADER}{Colors.BOLD}═══════════════════════════════════════════════════════════════════════════{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}                   RESULTADOS DE LA PRUEBA: {label}                        {Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}═══════════════════════════════════════════════════════════════════════════{Colors.ENDC}")
    print(f"  Excursión máxima alcanzada    : {max_dist*100.0:.2f} cm")
    print(f"  Distancia total recorrida     : {total_arc_length*100.0:.2f} cm")
    print(f"  Inliers promedio en movimiento: {mean_inliers:.1f}")
    print(f"  Pérdida de tracking           : {tracking_lost_pct:.1f}% ({total_lost_frames} frames)")
    print(f"  ─────────────────────────────────────────────────────────────────────────")
    
    jitter_color = Colors.OKGREEN if pass_jitter else Colors.FAIL
    print(f"  Jitter estático inicial (σ)   : {jitter_color}{Colors.BOLD}{jitter_pos_s1:.4f} mm{Colors.ENDC} (Límite: <= 2.50 mm) -> {'[PASA]' if pass_jitter else '[EXCEDE]'}")
    print(f"  Jitter estático final (σ)     : {jitter_pos_s2:.4f} mm")
    print(f"  ─────────────────────────────────────────────────────────────────────────")
    
    loop_color = Colors.OKGREEN if pass_loop else Colors.FAIL
    print(f"  {Colors.BOLD}ERROR DE RETORNO (LOOP CLOSURE){Colors.ENDC}: {loop_color}{Colors.BOLD}{loop_error_3d:.3f} mm{Colors.ENDC} (Criterio: <= 3.00 mm) -> {'[PASA]' if pass_loop else '[EXCEDE]'}")
    print(f"    - Desglose cartesiano       : ΔX = {delta_p[0]:+.3f} mm, ΔY = {delta_p[1]:+.3f} mm, ΔZ = {delta_p[2]:+.3f} mm")
    print(f"  Error angular de retorno      : {loop_error_rot_deg:.3f}°")
    print(f"    - Desglose Euler            : ΔRoll = {delta_euler[0]:+.2f}°, ΔPitch = {delta_euler[1]:+.2f}°, ΔYaw = {delta_euler[2]:+.2f}°")
    print(f"{Colors.HEADER}{Colors.BOLD}═══════════════════════════════════════════════════════════════════════════{Colors.ENDC}")
    
    verdict_str = f"{Colors.OKGREEN}{Colors.BOLD}CERTIFICADO / APTO PARA RECOLECCIÓN{Colors.ENDC}" if overall_pass else f"{Colors.WARNING}{Colors.BOLD}REQUIERE AJUSTE PARAMÉTRICO{Colors.ENDC}"
    print(f"  Veredicto final               : {verdict_str}\n")

    # Guardar en archivo de historial JSON
    result_data = {
        'label': label,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'max_excursion_cm': round(max_dist * 100.0, 2),
        'total_distance_cm': round(total_arc_length * 100.0, 2),
        'jitter_static_start_mm': round(jitter_pos_s1, 4),
        'jitter_static_end_mm': round(jitter_pos_s2, 4),
        'loop_error_3d_mm': round(loop_error_3d, 3),
        'delta_x_mm': round(delta_p[0], 3),
        'delta_y_mm': round(delta_p[1], 3),
        'delta_z_mm': round(delta_p[2], 3),
        'loop_error_rot_deg': round(loop_error_rot_deg, 3),
        'mean_inliers': round(mean_inliers, 1),
        'tracking_lost_pct': round(tracking_lost_pct, 2),
        'pass_loop': bool(pass_loop),
        'pass_jitter': bool(pass_jitter),
        'overall_pass': bool(overall_pass)
    }

    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except Exception:
            history = []
            
    history.append(result_data)
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

    # ──────────────────────────────────────────────────────────────────────────
    # TABLA COMPARATIVA HISTÓRICA ACUMULADA
    # ──────────────────────────────────────────────────────────────────────────
    print(f"{Colors.BOLD}TABLA COMPARATIVA DE CONFIGURACIONES:{Colors.ENDC}")
    header = f"{'Configuración':<22} | {'Loop Err (mm)':<13} | {'Jitter (mm)':<11} | {'Err Rot (°)':<11} | {'Inliers':<8} | {'Veredicto':<10}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for h in history:
        v_tag = f"{Colors.OKGREEN}PASS{Colors.ENDC}" if h['overall_pass'] else f"{Colors.FAIL}FAIL{Colors.ENDC}"
        print(f"{h['label']:<22} | {h['loop_error_3d_mm']:<13.3f} | {h['jitter_static_start_mm']:<11.4f} | {h['loop_error_rot_deg']:<11.3f} | {h['mean_inliers']:<8.1f} | {v_tag:<10}")
    print("-" * len(header) + "\n")

    # Cierre limpio sin SIGABRT
    try:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser(description="Benchmark de Odometría Visual-Inercial (VIO RTAB-Map)")
    parser.add_argument('--label', type=str, default="conf 0 (Baseline)", help="Etiqueta identificadora de la configuración")
    parser.add_argument('--static-duration', type=float, default=3.0, help="Duración de medición estática en segundos")
    parser.add_argument('--history-file', type=str, default="experiments/vio_benchmark_history.json", help="Ruta del archivo histórico JSON")
    args = parser.parse_args()

    run_evaluation(label=args.label, static_duration=args.static_duration, history_file=args.history_file)

if __name__ == '__main__':
    main()
