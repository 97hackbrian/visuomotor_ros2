#!/usr/bin/env python3
"""
Benchmark experimental de Odometría Visual (VO) para RealSense D415 + RTAB-Map.
Mide frecuencia efectiva, tiempo de cómputo por frame, inliers y jitter de pose (posición y rotación).
"""

import sys
import time
import argparse
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from rtabmap_msgs.msg import OdomInfo

def quaternion_to_euler_deg(qx, qy, qz, qw):
    """Convierte cuaternión a ángulos de Euler (Roll, Pitch, Yaw) en grados."""
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1.0:
        pitch = np.copysign(np.pi / 2.0, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.degrees([roll, pitch, yaw])

class VOBenchmarkNode(Node):
    def __init__(self, duration_s=6.0, label="VO"):
        super().__init__('vo_benchmark_node')
        self.duration_s = duration_s
        self.label = label
        
        self.odom_times = []
        self.positions = []
        self.orientations_euler = []
        
        self.info_times = []
        self.time_estimations = []
        self.bundle_times = []
        self.inliers_list = []
        self.features_list = []
        self.lost_count = 0
        self.total_info = 0
        
        self.create_subscription(Odometry, '/rtabmap/odom', self._odom_cb, 20)
        self.create_subscription(OdomInfo, '/odom_info', self._info_cb, 20)
        
        self.start_wall_time = None
        self.is_done = False
        self.get_logger().info(f"Iniciando Benchmark VO [{self.label}] durante {self.duration_s}s...")

    def _odom_cb(self, msg):
        now = time.perf_counter()
        if self.start_wall_time is None:
            self.start_wall_time = now
            
        elapsed = now - self.start_wall_time
        if elapsed > self.duration_s:
            self.is_done = True
            return
            
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        
        self.odom_times.append(now)
        self.positions.append([pos.x, pos.y, pos.z])
        euler = quaternion_to_euler_deg(ori.x, ori.y, ori.z, ori.w)
        self.orientations_euler.append(euler)

    def _info_cb(self, msg):
        if self.start_wall_time is None:
            return
            
        self.total_info += 1
        if msg.lost:
            self.lost_count += 1
            
        self.inliers_list.append(msg.inliers)
        self.features_list.append(msg.features)
        self.time_estimations.append(msg.time_estimation * 1000.0) # ms
        self.bundle_times.append(msg.local_bundle_time * 1000.0)  # ms (0 si el ajuste de haces está apagado)

    def get_results(self):
        if len(self.odom_times) < 5:
            return None
            
        dts = np.diff(self.odom_times)
        rates = 1.0 / dts[dts > 0]
        
        pos_arr = np.array(self.positions)
        rot_arr = np.array(self.orientations_euler)
        
        # Jitter en posición (std dev en mm)
        std_x = np.std(pos_arr[:, 0]) * 1000.0
        std_y = np.std(pos_arr[:, 1]) * 1000.0
        std_z = np.std(pos_arr[:, 2]) * 1000.0
        jitter_pos_3d = np.sqrt(std_x**2 + std_y**2 + std_z**2)
        
        # Jitter en rotación (std dev en grados)
        std_roll = np.std(rot_arr[:, 0])
        std_pitch = np.std(rot_arr[:, 1])
        std_yaw = np.std(rot_arr[:, 2])
        jitter_rot_3d = np.sqrt(std_roll**2 + std_pitch**2 + std_yaw**2)
        
        time_est_mean = np.mean(self.time_estimations) if self.time_estimations else 0.0
        time_est_std = np.std(self.time_estimations) if self.time_estimations else 0.0
        inliers_mean = np.mean(self.inliers_list) if self.inliers_list else 0.0
        features_mean = np.mean(self.features_list) if self.features_list else 0.0
        lost_rate = (self.lost_count / max(1, self.total_info)) * 100.0

        return {
            'label': self.label,
            'samples': len(self.odom_times),
            'duration_actual_s': self.odom_times[-1] - self.odom_times[0],
            'rate_mean_hz': float(np.mean(rates)),
            'rate_std_hz': float(np.std(rates)),
            'rate_min_hz': float(np.min(rates)),
            'rate_max_hz': float(np.max(rates)),
            'time_estimation_mean_ms': float(time_est_mean),
            'time_estimation_std_ms': float(time_est_std),
            # Percentiles: con 33 ms por frame a 30 fps, lo que importa es la cola (p90, p99), no la media
            **{f'time_estimation_p{q}_ms': float(np.percentile(self.time_estimations, q)) if self.time_estimations else 0.0
               for q in (50, 90, 99)},
            'time_estimation_max_ms': float(np.max(self.time_estimations)) if self.time_estimations else 0.0,
            'local_bundle_p50_ms': float(np.median(self.bundle_times)) if self.bundle_times else 0.0,
            'inliers_mean': float(inliers_mean),
            'features_mean': float(features_mean),
            'jitter_pos_x_mm': float(std_x),
            'jitter_pos_y_mm': float(std_y),
            'jitter_pos_z_mm': float(std_z),
            'jitter_pos_3d_mm': float(jitter_pos_3d),
            'jitter_roll_deg': float(std_roll),
            'jitter_pitch_deg': float(std_pitch),
            'jitter_yaw_deg': float(std_yaw),
            'jitter_rot_3d_deg': float(jitter_rot_3d),
            'tracking_lost_percent': float(lost_rate)
        }

def run_benchmark(duration_s=6.0, label="VO"):
    rclpy.init()
    node = VOBenchmarkNode(duration_s=duration_s, label=label)
    
    start_wait = time.time()
    # Esperar hasta que empiece a recibir o timeout
    while rclpy.ok() and node.start_wall_time is None and (time.time() - start_wait < 15.0):
        rclpy.spin_once(node, timeout_sec=0.1)
        
    if node.start_wall_time is None:
        print(f"[{label}] ERROR: No se recibieron mensajes de odometría en 10s.")
        node.destroy_node()
        rclpy.shutdown()
        return None
        
    while rclpy.ok() and not node.is_done:
        rclpy.spin_once(node, timeout_sec=0.05)
        
    res = node.get_results()
    node.destroy_node()
    rclpy.shutdown()
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark RTAB-Map VO")
    parser.add_argument('--duration', type=float, default=6.0, help="Duración en segundos")
    parser.add_argument('--label', type=str, default="Baseline", help="Etiqueta del experimento")
    args = parser.parse_args()
    
    res = run_benchmark(duration_s=args.duration, label=args.label)
    if res:
        import json
        print("\n" + "="*60)
        print(f" RESULTADOS BENCHMARK: {res['label']}")
        print("="*60)
        print(f"  Muestras analizadas : {res['samples']} en {res['duration_actual_s']:.2f} s")
        print(f"  Frecuencia Odometría: {res['rate_mean_hz']:.2f} +/- {res['rate_std_hz']:.2f} Hz (Rango: [{res['rate_min_hz']:.1f}, {res['rate_max_hz']:.1f}] Hz)")
        print(f"  Tiempo Cómputo CPU  : {res['time_estimation_mean_ms']:.2f} +/- {res['time_estimation_std_ms']:.2f} ms")
        print(f"  Tiempo por frame    : p50 {res['time_estimation_p50_ms']:.1f} / p90 {res['time_estimation_p90_ms']:.1f} / p99 {res['time_estimation_p99_ms']:.1f} / máx {res['time_estimation_max_ms']:.1f} ms (ajuste de haces p50 {res['local_bundle_p50_ms']:.1f} ms)")
        print(f"  Inliers Medios      : {res['inliers_mean']:.1f} / Features: {res['features_mean']:.1f}")
        print(f"  Jitter Posición 3D  : {res['jitter_pos_3d_mm']:.4f} mm (X={res['jitter_pos_x_mm']:.4f}, Y={res['jitter_pos_y_mm']:.4f}, Z={res['jitter_pos_z_mm']:.4f})")
        print(f"  Jitter Angular 3D   : {res['jitter_rot_3d_deg']:.4f} deg (Roll={res['jitter_roll_deg']:.4f}, Pitch={res['jitter_pitch_deg']:.4f}, Yaw={res['jitter_yaw_deg']:.4f})")
        print(f"  Pérdida de Tracking : {res['tracking_lost_percent']:.1f} %")
        print("="*60)
        print(f"JSON_OUTPUT:{json.dumps(res)}")
    else:
        sys.exit(1)
