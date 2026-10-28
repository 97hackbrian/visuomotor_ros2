#!/usr/bin/env python3
"""
Herramienta Interactiva de Calibración del Pseudo-Sensor de Fuerza Propioceptivo.

Mide la posición articular real (/joint_states) del gripper en dos estados:
1. Vacío (cierre total sin objeto) -> pseudo_force_empty ~= 0.0
2. Con Cubo (cierre mecánico contra el objeto) -> pseudo_force_clamped > 0.0

Calcula el umbral óptimo discriminante (pseudo_force_clamped_threshold)
para inyectar en rl_training_params.yaml.
"""

import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class GripperCalibratorNode(Node):
    def __init__(self):
        super().__init__("gripper_calibrator_node")
        self.gripper_pub = self.create_publisher(
            Float64MultiArray, "/position_controller/commands", 10
        )
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)

        self.current_pos = None
        self.gripper_joint_pattern = "gripper"
        self.get_logger().info("Nodo de calibración iniciado. Esperando /joint_states...")

    def _joint_cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            if self.gripper_joint_pattern in name:
                self.current_pos = float(pos)
                break

    def command_and_measure(self, cmd_val: float, duration_s: float = 2.5) -> float:
        msg = Float64MultiArray()
        msg.data = [float(cmd_val)]
        self.get_logger().info(f"Enviando comando {cmd_val} y esperando {duration_s}s...")

        start = time.time()
        while time.time() - start < duration_s:
            self.gripper_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(0.05)

        return self.current_pos


def main():
    rclpy.init()
    node = GripperCalibratorNode()

    # Esperar primera lectura
    while node.current_pos is None and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        time.sleep(0.05)

    print("\n" + "=" * 70)
    print(" 🛠️  CALIBRADOR DEL PSEUDO-SENSOR DE FUERZA PROPIOCEPTIVO")
    print("=" * 70)
    print("Este asistente medirá la discrepancia elástica (pos_actual - pos_cmd)")
    print("para determinar el umbral óptimo de agarre.\n")

    # Paso 1: Abrir
    input("Presiona [ENTER] para abrir el gripper completamente...")
    pos_open = node.command_and_measure(0.0, duration_s=2.0)
    print(f"👉 Posición con Gripper ABIERTO (cmd=0.0): {pos_open:.6f}\n")

    # Paso 2: Cerrar en Vacío
    input("Asegúrate de que NO haya ningún objeto entre las pinzas y presiona [ENTER]...")
    cmd_close = -0.010
    pos_empty = node.command_and_measure(cmd_close, duration_s=2.5)
    pseudo_force_empty = pos_empty - cmd_close
    print(f"👉 Posición CERRADO AL VACÍO (cmd={cmd_close}): {pos_empty:.6f}")
    print(f"👉 Pseudo-Fuerza en VACÍO: {pseudo_force_empty:.6f} m\n")

    # Paso 3: Abrir para colocar objeto
    print("Abriendo el gripper para colocar el cubo...")
    node.command_and_measure(0.0, duration_s=1.5)

    # Paso 4: Cerrar con Cubo
    input("Coloca el CUBO entre las pinzas en la simulación y presiona [ENTER]...")
    pos_clamped = node.command_and_measure(cmd_close, duration_s=2.5)
    pseudo_force_clamped = pos_clamped - cmd_close
    print(f"👉 Posición CERRADO CON CUBO (cmd={cmd_close}): {pos_clamped:.6f}")
    print(f"👉 Pseudo-Fuerza con CUBO: {pseudo_force_clamped:.6f} m\n")

    # Cálculo del Umbral
    threshold = (pseudo_force_empty + pseudo_force_clamped) / 2.0

    print("=" * 70)
    print(" 📊 RESULTADOS DE CALIBRACIÓN")
    print("=" * 70)
    print(f"• Pseudo-Fuerza en Vacío (sin cubo):   {pseudo_force_empty:.6f} m")
    print(f"• Pseudo-Fuerza con Cubo (agarre real): {pseudo_force_clamped:.6f} m")
    print(f"• Separación / Margen de Detección:     {(pseudo_force_clamped - pseudo_force_empty):.6f} m")
    print(f"\n🎯 UMBRAL ÓPTIMO RECOMENDADO (pseudo_force_clamped_threshold):")
    print(f"   {threshold:.6f}")
    print("=" * 70)
    print("\nActualiza este valor en: visuomotor_ros2/config/rl_training_params.yaml")
    print(f"   pseudo_force_clamped_threshold: {threshold:.6f}\n")

    # Regresar a abrir
    node.command_and_measure(0.0, duration_s=1.5)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
