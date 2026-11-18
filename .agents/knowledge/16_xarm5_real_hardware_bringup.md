# 16. Bringup del Manipulador Real xArm5 (UFACTORY) en Jetson Orin Nano

## 1. Hardware y Hallazgos Previos en la Jetson
* **Brazo:** UFACTORY xArm5 (5 GDL, 3 kg de carga). Controlador en `192.168.1.228` (3 sesiones antiguas en `.239`).
* **Historial (host, no contenedor):** `~/dev_ws/src/xarm_ros2` (rama `foxy`, ROS 2 Foxy) con 206 ejecuciones de
  `xarm_driver_node` entre feb–oct 2025. Errores observados: C2 (Emergency IO), C22 (autocolisión),
  C14 (servo J4), C23 (límite articular).
* **Código previo sin versionar:** `~/dev_ws/src/xarm_ros2/xarm_examples/` (teleoperación con mando, home articular
  `[0, -0.82, -0.28, 1.11, 0]` rad "adaptado al gripper", gripper por acción `GripperCommand`).
* El bringup de simulación del repo (`lite6_cartesian.launch.py`) corresponde a un **Lite6 (6 GDL)**, no al xArm5.

## 2. Arquitectura Implementada
```
policy_node --target_frame_raw--> pose_smoother --target_frame--> xarm5_bridge --> /xarm/set_servo_cartesian (modo 1)
policy_node --/position_controller/commands--> xarm5_bridge --> set_gripper_position | set_vacuum_gripper
/xarm/robot_states (report_type dev) --> xarm5_bridge --> TF link_base->xarm_tcp
/xarm/joint_states (drive_joint)     --> xarm5_bridge --> /visuomotor/gripper_joint_states (escala de la política)
```
* `visuomotor_bringup/visuomotor_bringup/xarm5_interface.py`: lógica pura (sin ROS), cubierta por
  `tests/test_xarm5_interface.py` (21 pruebas: convención RPY, proyección 5 GDL, caja de trabajo, límites de paso,
  unidades, gripper).
* `visuomotor_bringup/scripts/xarm5_bridge.py`: nodo con `MultiThreadedExecutor`; llamadas síncronas a servicios
  mediante `threading.Event` (no `spin_until_future_complete` dentro de callbacks).
* `visuomotor_bringup/launch/xarm5_real.launch.py` + `config/xarm5/xarm5_real_params.yaml` (única fuente de valores;
  `gripper_type` define también `add_gripper`/`add_vacuum_gripper` del driver).

## 3. Convenciones del Driver `xarm_api` (verificadas en el código fuente)
* Posiciones en **mm**, ángulos en **rad** (`is_radian=true`); RPY sobre ejes fijos: `R = Rz(yaw)·Ry(pitch)·Rx(roll)`.
* `set_servo_cartesian`, `set_gripper_*` y `set_vacuum_gripper` están **desactivados por defecto** en
  `xarm_params.yaml`; el YAML del proyecto los habilita en el bloque `ufactory_driver.services`.
* Gripper xArm: `drive_joint` publicado 0 (abierto) .. 0.85 (cerrado). El driver solo publica la posición del
  gripper de forma continua con **firmware >= 2.7.101** (puerto 30000); con firmware anterior un hilo republica la
  posición *inicial* cada 500 ms hasta que se ejecuta la **acción** `xarm_gripper/gripper_action`. El servicio
  `set_gripper_position` no publica nada. Por eso el puente usa la acción `control_msgs/GripperCommand`
  (posición en escala `drive_joint`), igual que `xarm_examples/Nodo_gripper.py` que ya funcionó con este brazo.
* Cada servicio del driver se crea solo si `services.<nombre>: true` (bajo el namespace `hw_ns`).
* Interfaces (`MoveCartesian`, `RobotMsg`) idénticas entre las ramas `foxy` y `humble`.

## 4. Restricción Cinemática de 5 GDL
J1 vertical, J2–J4 horizontales paralelos, J5 sobre el eje de la herramienta: no es posible orientación 6D arbitraria.
`orientation_mode: top_down_yaw` fija roll=π, pitch=0 (garra vertical, alcanzable en todo el volumen) y conserva el
yaw de la política. Una política entrenada con el Lite6 (6 GDL) puede pedir inclinaciones que se descartan:
considerar datasets/simulación con xArm5 para cerrar esta brecha (ver módulo 10, Sim-to-Real).

## 5. Seguridad
Arranque desarmado y `dry_run: true`; `~/arm`, `~/disarm`, `~/go_home` (std_srvs/Trigger); caja + radio XY;
5 cm/s y 0.5 rad/s iniciales (paso calculado con el tiempo real entre comandos, acotado al período nominal, para que
el jitter del ejecutor no supere el límite); yaw acotado a `[yaw_min, yaw_max]` e interpolado sin cruzar ±π;
rechazo de objetivos > 25 cm del TCP medido o con NaN/Inf; watchdog de objetivo (0.5 s) y de respuesta de
`set_servo_cartesian` (0.5 s, con número de secuencia para descartar respuestas tardías); desarme ante `err != 0`,
pérdida de `robot_states` o 5 fallos seguidos. Control y publicación de TF van en grupos de callbacks separados.
`simulate_robot_state: true` permite probar la cadena completa sin hardware (exige `dry_run: true`).

## 5b. Verificación sin hardware
* `tools/xarm5/fake_xarm_driver.py`: driver falso con las mismas interfaces que `xarm_api` (rechaza servo fuera de
  modo 1 o con error, como el controlador) e inyección de fallos (`/fake_xarm/inject_error|fail_servo|pause_states|slow_servo`).
* `tools/xarm5/check_xarm5_bridge.py`: 17 comprobaciones del puente en modo REAL contra el driver falso
  (armado, unidades, límites, orientación, watchdog, gripper, errores, timeouts, pérdida de estado, home).
* Pendiente con el robot real: confirmar la frecuencia de `robot_states` con `report_type: dev`
  (`ros2 topic hz /xarm/robot_states`) y la versión de firmware (afecta a la publicación continua del gripper).

## 5c. Datos medidos en el brazo real (2026-11-17, sin movimiento, puente en dry-run)
* Controlador `192.168.1.228`, Jetson `eth0` `192.168.1.5/24`; puertos 502, 30001-30003 y 18333 abiertos.
* `VERSION: v2.5.1`, `DETAIL: 5,5,XF1305,DC1305` (modelo XF1305 = "XF13"). Firmware < 2.7.101: el estado del
  gripper solo se publica vía la acción (decisión del §3 confirmada).
* `report_type: dev` → `/xarm/robot_states` a ~99 Hz; `/xarm/joint_states` ~92 Hz.
* TCP offset configurado en el controlador: `[0, 0, 126 mm, 0, 0, 0]` (la pose reportada no es la brida).
* J1 reportado en +2π (6.283 rad): un home articular con J1 = 0 giraría la base 360°.

## 5d. Primer movimiento real y caída al deshabilitar motores (2026-11-17)
* Prueba +50 mm en X, 2 cm/s, orientación congelada, modo servo: llegó a x=279.9 mm (objetivo 279.9) sin desvío en
  Y/Z, `err=0` en todo el trayecto. La cadena puente → `set_servo_cartesian` queda validada en hardware.
* **Al deshabilitar motores con `motion_enable(8, 0)` (para aplicar frenos) el brazo CAYÓ** antes de quedar frenado:
  TCP de (279.9, 0.4, 165.9) a (221.8, 0.4, 116.4) mm, pitch +0.145 rad (J2 -0.74→-0.50, J4 1.11→1.02).
  Luego quedó estable (`mt_able=0`, `mt_brake=0`). No usar `motion_enable(8, 0)` con el brazo en el aire hasta revisar
  frenos/procedimiento con UFACTORY; la pose inicial de la sesión (z=165.9) pudo ser también resultado de una caída previa.

## 5e. Teleoperación RViz2 remota, frenos y home2 (2026-11-18)
* **Teleop:** `scripts/xarm5_teleop_marker.py` (Jetson) crea el marcador interactivo `/xarm5_teleop` sobre el TCP
  (TF `link_base→xarm_tcp`); controles X/Y/Z + yaw (marco fijo); menú Seguir/Detener/Sincronizar/Armar/Desarmar/
  home2/Comprobar frenos. Publica `/target_frame` a 20 Hz solo con *Seguir* + puente ARMADO; se apaga solo si el
  puente se desarma. RViz2 corre en otro equipo con `launch/xarm5_rviz.launch.py` + `rviz/xarm5_teleop.rviz`.
* **URDF:** `robot_state_publisher` en la Jetson con `xarm_device.urdf.xacro` (dof 5) y `joint_states` remapeado a
  `/xarm/joint_states`; `xacro` 2.0.8 se compila en `/opt/xarm_ws`. El equipo remoto necesita `xarm_description` para
  las mallas.
* **Red:** Jetson con dos interfaces (eth0 robot / wlan0 laboratorio). `deploy/jetson/config/cyclonedds.xml` fija la
  interfaz DDS con `DDS_IFACE` (defecto wlan0) y admite `DDS_PEER` (unicast). `ROS_DOMAIN_ID=42` para el robot real
  (decisión del usuario 2026-11-18; coincide con la simulación: con el brazo conectado, simulación/políticas solo
  con `ROS_LOCALHOST_ONLY=1`). Humble (Jetson) ↔ Jazzy (otro equipo): interoperabilidad DDS no garantizada
  oficialmente; alternativa: RViz2 en un contenedor Humble en el otro equipo.
* **Frenos:** SDK (`xarm_report.cc`): `motor_brake_states[i] = (mt_brake >> i) & 1`, listo solo si todos los ejes = 1
  → bit 1 = freno liberado; `mt_able` bit 1 = motor habilitado. El puente verifica ambos tras `motion_enable` (armar,
  go_home*) y durante el movimiento; `~/check_brakes` solo lee. Fake driver con fallo `stuck_brake` (J3).
* **home2:** articulaciones `[6.2822, -0.5001, -0.2717, 1.0231, -0.0022]`, TCP (0.2204, 0.0008, 0.1150) m,
  rpy (3.1246, 0.2508, 0.0171). Entre dos lecturas con frenos puestos la pose varió ~1.5 mm (posible deriva de freno).

## 6. Contenedor Jetson
`deploy/jetson/Dockerfile.jetson` compila en `/opt/xarm_ws` (underlay) `xarm_ros2@ac736c92` (humble, SDK C++
`f367943`) y `control_msgs@4418bbf` (4.9.0; ausente en la imagen base), y añade `transforms3d==0.4.2` para
`pose_smoother`. Procedimiento de puesta en marcha por etapas en `deploy/jetson/README.md` (sección 3b).
