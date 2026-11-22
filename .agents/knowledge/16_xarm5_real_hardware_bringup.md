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
* **home2:** articulaciones `[6.2822, -0.5001, -0.2717, 1.0231, -0.0022]`, rpy (3.1246, 0.2508, 0.0171).
  TCP (0.2516, 0.0035, 0.2370) m con offset 0 (antes (0.2204, 0.0008, 0.1150) con offset de 126 mm).
* **Offset de TCP del controlador puesto a 0 (2026-11-18)** con `set_tcp_offset([0]*6)` + `save_conf` (persistente),
  porque el brazo no tiene herramienta: el TCP reportado = brida (`link_eef`). El controlador pasó de state 4 a 5
  (detenido) sin moverse. Al montar una herramienta, volver a fijar su offset y re-medir home2 y el home de la política.
  Diagnóstico de la primera sesión RViz2 remota: el marcador aparecía "más abajo" por el offset de 126 mm, y el brazo
  no se movía porque el bringup estaba en dry-run (los clics sí llegaban a la Jetson; Humble↔Jazzy funcionó). Entre dos lecturas con frenos puestos la pose varió ~1.5 mm (posible deriva de freno).
* **Certificación RViz2 Remota (2026-11-18):** Comunicación validada exitosamente entre el contenedor de la Jetson (Humble) y el equipo remoto (Jazzy) sobre la red del laboratorio mediante `tools/xarm5/run_remote_rviz.sh` y CycloneDDS con peer unicast dinámico (`DDS_PEER`). Telemetría de estado a 0.00 s de latencia, renderizado 3D del manipulador xArm5 verificado en pantalla y marcador interactivo `/xarm5_teleop` acoplado al TCP.

## 5f. C23 por J1 en +2π y protección de límites articulares (2026-11-18)
* Primera teleoperación real: C2 (Emergency IO) 4 s después de armar sin movimiento, y C23 (Joint Angle Exceed
  Limit) 8 s después de "Seguir": J1 estaba en +6.2822 rad (límite +2π = 6.2832, URDF xarm5) y quedó en 6.2832.
* Puente: `joint_lower/upper_limits_rad` (URDF), `joint_limit_margin_rad` 0.10 → no arma cerca de un límite y se
  desarma (modo posición) antes del C23; `max_home_joint_delta_rad` 1.0 → go_home* rechazan giros grandes;
  `~/unwind_j1` (manual, desarmado, `unwind_speed_rad_s` 0.3, limpia error si es posible) gira solo J1 ±2π.
* home2 se representa con J1 = -0.0010 (misma postura que +6.2822). Fake driver: `initial_joints_rad` (home2) y
  `/fake_xarm/set_joints`; checks 23-26 cubren el caso real.

## 5g. Freno intermitente de J3 y lectura de estado obsoleto (2026-11-19)
* Reporte del usuario: el freno de J3 a veces no se libera al habilitar; se resolvía bloqueando/desbloqueando hasta
  que liberaba. El puente reintenta SOLO sobre la junta fallida: `motion_enable(id=j, 1)` hasta `brake_release_retries`
  (3) con `brake_retry_wait_s`; opcional `brake_retry_disable_cycle` (deshabilitar/habilitar esa junta solo si su freno
  reporta APLICADO; nunca todas). Tras un reintento exitoso reafirma modo y estado; el mensaje de armado lo indica.
* Bug encontrado por los checks: con motores ya habilitados, el sondeo aceptaba un `robot_states` ANTERIOR a la nueva
  habilitación ("todo liberado") aunque el freno acabara de quedar aplicado. Corregido: solo cuentan reportes
  posteriores a la llamada + `brake_state_settle_s` (0.15 s).
* Fake driver: `motion_enable` por junta y `/fake_xarm/set_stuck_release_after` (libera al intento N); checks 27-28.
* Nota: el otro equipo añadió anillos de pitch/roll al marcador; con `orientation_mode: top_down_yaw` el puente los
  ignora (garra vertical), solo el yaw llega al brazo.

## 5h. Operación desde RViz2 en el host: desenrollado real y frenos (2026-11-19)
* Primer `unwind_j1` real con J1 = 6.2833 (fuera de rango por 0.0001): rechazado con ret=-9 y C23. El segundo intento
  (tras C2 por la seta y `clean_error`) sí fue aceptado: J1 6.2833 → 0.0001 a 0.3 rad/s, postura idéntica.
* Frenos: el controlador reportó `mt_brake=11111` (liberados) mientras el esfuerzo de J3 se mantuvo en ~76 (límite 92)
  frente a ~5 en J2; el usuario sospecha que el freno de J3 no se libera físicamente aunque el bit diga liberado.
  Pendiente: detección por esfuerzo anómalo de J3.
* `motion_enable(8, 0)` con el brazo en reposo (state 2): 0.0 mm de desplazamiento (la caída de ~5 cm previa fue justo
  tras un movimiento servo). Pulsar la seta tampoco movió el brazo.
* Menú RViz2 nuevo: Desenrollar J1 (doble clic), Quitar frenos (`~/brakes_off`), Poner frenos (`~/brakes_on`, doble
  clic, solo desarmado y quieto >= `brakes_on_min_idle_s`). Checks: puente 31/31, teleop 13/13.

## 5i. HOME del fabricante e interfaz RViz2 en inglés (2026-11-20)
* Modelo confirmado: UFACTORY xArm 5 = XF1305 ("XF13" en la caja; controlador `DETAIL: 5,5,XF1305,DC1305,v2.5.1`),
  3 kg, 700 mm. Rangos (manual V2.3.0, tabla 1.1): J1 ±360°, J2 -118°~120°, J3 -225°~11°, J4 -97°~180°, J5 ±360°.
* HOME1 = Zero Position (todas a 0): manual V2.3.0 §1.5 "Move the robot to the zero position ... then put the xArm
  robot and Control Box in the original packaging"; §3.1.3: con gripper, en cero la herramienta sobrepasa la superficie.
* HOME2 = Initial Position por defecto del xArm 5 (xArm Studio V1.0.0: "xArm 5: J1:0 J2:0 J3:0 J4:-90 J5:0");
  V2.3.0: "a relatively safe position", indicada para recuperarse de C23. (Studio V2.6.0 la lista como todo ceros.)
* FK (URDF xarm5): HOME1 brida (0.207, 0, 0.112) m hacia abajo; HOME2 (0.228, 0, 0.285) m horizontal.
* Manual V2.3.0 p. "If it is xArm5, roll and pitch must be set to roll = ±180° and pitch = 0°" confirma `top_down_yaw`.
* Interfaz: menú y mensajes de RViz2/servicios en inglés; retirados "Ir a home2" (pose de trabajo) y "Desenrollar J1"
  del menú; HOME1/HOME2/Engage brakes con doble clic (`confirm_window_s`). `max_home_joint_delta_rad` por junta
  ([0.5, 3.2, 3.2, 3.2, 3.2]). Log de error C* deduplicado (transición atómica). Checks: puente 31/31, teleop 13/13.

## 5j. HOME3 (pose de trabajo alineada) y clean_error en los HOME (2026-11-20)
* Medida real cerca de cero: [-0.01816, -0.00199, -0.04436, 0.04688, -0.01357], efector no del todo recto
  (pitch 0.00053 rad, yaw -0.0046 rad). Para el xArm5 con garra hacia abajo: pitch = J2 + J3 + J4 (verificado con
  datos previos: home de trabajo 0.2513 vs pitch medido 0.2508) y yaw = J1 - J5. HOME3 = J4 -> -(J2+J3), J5 -> J1:
  [-0.01816, -0.00199, -0.04436, 0.04635, -0.01816] (FK: rpy = (pi, 0, 0)).
* La alineación física no se aplicó: el usuario tenía la seta pulsada (C2). "Go to HOME3" la realiza.
* `go_home1/2/3` ahora ejecutan `clean_error` si hay error pendiente (antes solo arm/brakes_off/unwind). Check 32.

## 5k. Marcador RViz2 solo con los GDL cartesianos del xArm5 (2026-11-20)
* xArm User Manual V2.3.0 (movimiento lineal): "If it is xArm5, roll and pitch must be set to roll = ±180° and
  pitch = 0°". Los anillos de pitch (verde, Y) y roll (rojo, X) que se habían añadido al marcador nunca podían
  funcionar en cartesiano (el puente los ignoraba con `top_down_yaw`); se eliminaron. Quedan flechas X/Y/Z y el
  anillo de yaw (azul, Z). Check teleop 10b (14/14). Inclinar la garra solo sería posible por movimiento articular (J4).
* Corrección posterior: el anillo verde no lo rechazaba el driver, lo ignoraba el puente (`top_down_yaw`). En la
  primera prueba real `servo_cartesian` aceptó pitch 0.10 rad en el plano vertical del brazo. Tras recompilar el
  marcador hay que recrear el contenedor (`docker compose up -d --force-recreate`) y reabrir RViz2.
* Calibración del controlador frente a la URDF nominal: pitch = J2+J3+J4 - 0.00052, yaw = J1 - J5 + 0.00135. HOME3
  final: [-0.01816, -0.00199, -0.04436, 0.04688, -0.01681] (rpy del controlador ≈ (3.14069, 0.00001, 0.00001)).

## 5l. Alcance máximo, límite 0.65 m y propuesta `arm_plane` para DP (2026-11-20)
* Alcance con garra vertical ≈ 0.716 m (codo recto = singularidad, riesgo de C24). El recorte previo a 0.55 m era de
  software. Nuevo límite: `workspace_x_max: 0.65`, `workspace_max_radius_xy: 0.65` (margen ~6.6 cm a la singularidad).
* Con 5 GDL, la orientación alcanzable es yaw libre y una inclinación solo dentro del plano vertical que contiene al
  brazo (acimut = atan2(y, x)); roll fuera del plano es inalcanzable.
* Propuesta (no implementada, espera la decisión del usuario según la tarea):
  - `orientation_mode: arm_plane` en el puente: proyectar el cuaternión de la política a (pitch en el plano del brazo,
    yaw) y saturar el pitch con un parámetro.
  - Anillo verde con eje variable = perpendicular al plano del brazo según el acimut del TCP.
  - `tools/project_dataset_to_xarm5.py`: proyectar las acciones de un dataset (copia nueva, nunca in situ) a la
    orientación alcanzable, para que entrenamiento y ejecución coincidan.
* Restricciones para las demostraciones reales (desde cero):
  - Garra vertical; o inclinada solo en el plano del brazo.
  - Inicio equivalente a HOME3, con la misma disposición de los objetos.
  - Volumen: radio 0.15–0.65 m, z 0.10–0.50 m, |y| ≤ 0.30 m.
  - Velocidad ≤ 5 cm/s y ≤ 0.5 rad/s (límites del puente).
  - Montaje de la cámara de muñeca idéntico al del robot.
  - Gripper en 0.0 si no se registra.
* Pendiente: diagnóstico del freno de J3 por esfuerzo (~76 frente a ~5 de J2).

## 5m. Modo `arm_plane`, anillo verde y conversión del dataset VIO (2026-11-20, sin hardware)
* **Modelo de orientación alcanzable** (`xarm5_interface.py`): `R = Rz(phi)·Ry(pitch)·Rz(yaw-phi)·Rx(pi)`, con
  phi = acimut del TCP (J1), pitch = inclinación en el plano del brazo (J2+J3+J4) y yaw = giro de la herramienta (J5).
  Con pitch 0 equivale a `top_down_yaw`; con yaw = phi coincide con rpy (pi, pitch, phi) del controlador.
  `arm_plane_params` (descomposición exacta), `project_arm_plane` (satura pitch y devuelve la desviación),
  `continuous_rpy` (evita alternar +pi/-pi sin salir de [-pi, pi]).
* **Hallazgo:** con la garra inclinada Y girada (yaw ≠ phi), el roll del controlador deja de ser ±pi (check 5b:
  rpy (-3.034, 0.28, 0.515)). El manual pide roll ±180° y pitch 0 en movimiento lineal para el xArm5; solo se ha
  ejecutado en real la inclinación con yaw ≈ phi (antiguo home2, pitch 0.25). Validar en la prueba real por etapas.
* **Puente:** `orientation_mode: arm_plane` (YAML), `arm_plane_pitch_min/max` ±0.5 rad (ampliado a ±1.48 rad ≈ 85° tras la prueba real del 2026-11-20; tope del puente |pitch| < π/2 − 0.05 por bloqueo de cardán del rpy del controlador; con −85° J4 se acerca a su límite de −97°), `arm_plane_warn_deviation_rad`
  0.15. Interpola en (pitch, yaw), no por componente rpy (se acoplan al inclinar); cada pose enviada es alcanzable para
  su posición. `status` publica `orientation_mode`, `pitch_limits_rad`, `yaw_limits_rad`.
* **Marcador:** lee modo y límites del `status` del puente (sin parámetros duplicados). Anillo verde `girar_pitch`
  (FIXED, eje (-sin phi, cos phi, 0)); anillo azul en INHERIT = eje de la herramienta. Al soltar (MOUSE_UP) ajusta la
  orientación a la alcanzable y recalcula el eje. Checks: puente 34/34 (5, 5b, 5c), teleop 16/16 (10b-10d),
  pytest 43 (interfaz + conversión).
* **Compatibilidad DP:** `vo_collector` guarda la cámara en `odom` (posición relativa al inicio, cuaternión absoluto
  de odom); `policy_node` observa el TCP en `link_base` menos home. `tools/xarm5/convert_vio_dataset_to_xarm5.py`:
  `T_base_tcp(t) = T_base_home · T_tcp_cam · T_odom_cam(0)^-1 · T_odom_cam(t) · T_tcp_cam^-1`, proyecta con el modo
  del puente, resta home, copia imágenes, gripper y done, y audita (desviación > 0.15 rad, volumen, velocidad > límites
  del puente). Lee home de `jetson_policy_params.yaml`, el modo y los límites de `xarm5_real_params.yaml`, y la
  extrínseca de `xarm5_dataset_alignment.yaml` (`extrinsic_calibrated: false`, por lo que se niega a convertir). No
  sobrescribe nunca un destino existente.
* **Home de la política = HOME3**, provisional por FK URDF (0.2221, -0.0040, 0.1160) m, rpy (pi, 0, 0). En el antiguo
  home2, la FK URDF difiere unos 4 mm del controlador: re-medir `tcp_xyz_m` en HOME3 con el brazo encendido.
* **Pendiente:**
  - Medir la extrínseca cámara→TCP y HOME3 real.
  - Prueba real por etapas: dry-run; luego tilt 0.1 con phi≈0; tilt con phi≠0; tilt+giro (yaw≠phi).
  - Si el controlador rechaza yaw≠phi inclinado: restringir el giro al inclinar.
  - Velocidad del puente (5 cm/s) muy baja para demostraciones a mano: subir el límite o grabar despacio.

## 5n. Alcance hombro-muñeca y z_max 0.75 (2026-11-20, sin hardware)
* Caja insuficiente: la esquina (0.65, z 0.50) de la caja anterior ya era inalcanzable (máx. 0.464 m aun con el codo
  recto). La altura alcanzable baja con el radio: con la garra vertical, x 0.15 → 0.77 m, x 0.50 → 0.61 m, x 0.65 →
  0.38 m (FK URDF, 95% del codo recto).
* Geometría (URDF xarm5): hombro (J2) en (0, 0, 0.267); muñeca J4 = brida + Rz(phi)·Ry(pitch)·Rx(pi)·(-0.076, 0,
  -0.097), constante en ese marco (verificado en 200 configuraciones aleatorias); codo recto 0.6406 m.
* `ReachLimit` + `clamp_to_workspace` (caja ↔ alcance, hasta 4 alternancias; si no hay punto válido, el puente
  mantiene la posición). Parámetros: `workspace_shoulder_xyz_m`, `workspace_wrist_offset_m`,
  `workspace_max_reach_m` 0.6086, `workspace_min_reach_m` 0.12 (HOME3 = 0.156), `workspace_z_max` 0.75.
* Validación por FK con margen articular de 0.10 rad: 0 puntos inalcanzables con pitch 0 y ±0.5. Con pitch +1.0 quedan
  bordes altos cerca de la base (z 0.66–0.72) y con -1.0 bordes bajos junto a la base; los corta el guard articular.
* El conversor VIO también marca los frames fuera de alcance. Checks: puente 35/35 (5d alcance, 7 techo de la caja),
  teleop 16/16, pytest 50.

## 5o. Prueba real del anillo verde y del volumen ampliado (2026-11-20)
* El usuario probó en el brazo real (desde RViz2 en el host) el anillo verde `arm_plane`: funciona. A pedido suyo, la
  inclinación pasó de ±0.5 a ±1.48 rad (~85°). El tope del puente es |pitch| < π/2 − 0.05 por el bloqueo de cardán del
  rpy del controlador.
* Tras el límite de alcance (z_max 0.75) el usuario confirmó "hasta aquí todo bien".
* Sigue sin validarse en hardware de forma explícita (sin log del caso):
  - Inclinación grande combinada con giro (roll lejos de ±π; en el check 5c rpy = (−1.825, 1.202, 1.475)).
  - −85° con J4 cerca de −97°.
* Pendientes vigentes:
  - Medir la extrínseca cámara→TCP (`xarm5_dataset_alignment.yaml`, `extrinsic_calibrated: false`).
  - Re-medir HOME3 real (`tcp_xyz_m`; hoy FK URDF (0.2221, −0.0040, 0.1160)) y actualizar `jetson_policy_params.yaml`.
  - Límite de 5 cm/s frente a la velocidad de las demostraciones a mano.
  - Freno de J3.

## 5p. HOME3 redefinido como pose de trabajo alta e inclinada (2026-11-20)
* El usuario colocó el brazo con el marcador de RViz2 y pidió guardar la pose como HOME3. Lectura (solo lectura,
  `/xarm/robot_states`, brazo armado y quieto en dos lecturas):
  - Juntas: [0.00089, −0.73289, −1.61376, 1.91852, 0.02678] rad = [0.05, −41.99, −92.46, 109.92, 1.53]°.
  - Controlador: TCP (0.1500, 0.0053, 0.7500) m, rpy (3.1018, −0.4296, −0.0114). Garra inclinada ~−24.6° en el plano
    del brazo (proyección: pitch −0.4275); alcance hombro–muñeca 0.541 m.
* El objetivo del usuario (0.149, 0.0001, 0.7527) quedó recortado por la caja (x_min 0.15, z_max 0.75). La FK URDF de
  esas juntas da exactamente (0.149, 0.0001, 0.7527), lo que RViz muestra con el modelo nominal. El controlador reporta
  (0.150, 0.0053, 0.750): una diferencia de calibración de ~5 mm (ya vista en el antiguo home2, ~4 mm).
* `home3_joints_rad` guardado tal cual (sin alinear). Home de la política = pose del controlador (la TF `xarm_tcp` que
  observa `policy_node`). El HOME3 anterior (cerca de cero) queda comentado en el YAML como histórico.
* Menú: "Go to HOME3: work pose (double click)". Skill `realsense-vio-data-collection` v1.1 aprobada y aplicada
  (restricciones xArm5 + conversión obligatoria).
* Implicación: HOME3 está en el techo (z_max) y en el radio mínimo, así que desde HOME3 las demostraciones solo pueden
  bajar o alejarse.

* Ajuste posterior (a pedido del usuario): centrar el controlador en y = 0. J1 y J5 giran -0.03503 rad. El brazo rota
  rígido sobre la base y J5 compensa (yaw ≈ J1 − J5 + c) para conservar el giro de la garra. `home3_joints_rad` = [-0.03414, -0.73289, -1.61376, 1.91852, -0.00825].
  TCP previsto (0.1501, 0.0000, 0.7500): valor previsto, a re-medir tras el primer go_home3.

## 5r. Calibración, Conversión Cinemática y Certificación del Dataset Real v7 (`demonstrations_real_v7_xarm5.zarr`, 2026-11-21)
* **Recolección en Hardware Real:** 83 demostraciones expertas a mano alzada capturadas con Intel RealSense D435i + RTAB-Map VIO en `datasets/demonstrations_real_v7.zarr`.
* **Calibración de Parámetros en `xarm5_dataset_alignment.yaml`:**
  - Extrínseca formal: `extrinsic_calibrated: true`, montaje en brida mirando hacia $+Z_{\text{tcp}}$ con pitch $-90^\circ$ (`tcp_to_camera_xyz_m: [0.0, 0.0, 0.0]`, `tcp_to_camera_rpy_rad: [0.0, -1.5707963, 0.0]`).
  - Dinámica humana: `speed_tolerance_ratio: 12.0` (tolera picos de hasta $0.60\text{ m/s}$ y $6.0\text{ rad/s}$ sin penalizar la velocidad media natural de $\sim 0.20\text{ m/s}$).
  - Inclinación y volumen: `max_orientation_deviation_rad: 1.50` (la herramienta ya proyecta analíticamente las orientaciones al plano vertical `arm_plane`) y `max_workspace_violation_frames: 550` (tolera oscilaciones al despegar desde HOME3 en el borde de la caja $X=0.15009, Z=0.750$).
* **Conversión y Filtrado (`tools/xarm5/convert_vio_dataset_to_xarm5.py --drop-flagged`):**
  - Descarte limpio de 2 episodios con tirones espurios del VIO (`episode_23` y `episode_24` con $v > 2.3\text{ m/s}$ y $\omega > 9\text{ rad/s}$).
  - Generación de `datasets/demonstrations_real_v7_xarm5.zarr` con **81 episodios limpios** ($26,868$ fotogramas, $22.39$ minutos a 20.0 Hz).
* **Certificación PyTorch y CUDA (`tools/verify_zarr_dataset.py`):**
  - Integridad de tensores validada (100% chunks válidos, 0 corruptos).
  - PyTorch DataLoader verificado (batch de 16: imágenes $[16, 2, 3, 256, 256]$, estado $[16, 2, 8]$, acción $[16, 16, 9]$).
  - Paso forward + backward de `DiffusionPolicy` ejecutado en CUDA exitosamente con loss inicial DDIM de $1.1839$.
* **Evaluación SOTA y Registro en SQLite (`tools/evaluate_real_dataset.py`):**
  - Generación de panel multicanal 2x3 a 300 DPI y PDF en inglés académico (`demonstrations_real_v7_xarm5_sota_panel.png/.pdf` y `demonstrations_real_v7_sota_panel.png/.pdf`).
  - **Subfigura D (Opción A — Eficiencia Geométrica):** Desplazamiento neto $D$ ($0.60\text{ m} \pm 0.08\text{ m}$) frente a longitud de arco efectivo suavizado $L_{\text{eff}}$ ($1.12\text{ m} \pm 0.25\text{ m}$), erradicando la acumulación de ruido discreto a 20 Hz ($8.2\text{ m}$) y cuantificando el índice de tortuosidad suave ($L_{\text{eff}} / D = 1.88 \pm 0.40$).
  - **Subfigura E (Propuesta E1 — Espacio de Fases):** Perfil de velocidad cartesiana $\mathbf{v}(d_{\text{target}})$ con eje de distancia invertido (salida a $d \approx 0.70\text{ m}$ hasta contacto en $0.0\text{ m}$), evidenciando desaceleración suave canónica al ingresar a la zona de contacto ($d \le 5\text{ cm}$).
  - Cumplimiento del piso de auto-stop al 100% (media $40.8$ pasos = $2.04\text{ s}$ de `done=1.0`).
  - Registro de los 81 episodios y métricas consolidadas en `experiments/experiments.db`.
* **Corrección de Bugs de Soporte:**
  - `Normalize.forward` en `visuomotor_core/dataset/normalize.py`: protección contra división por cero (`0/0 = NaN`) cuando `max - min == 0` en dimensiones constantes (gripper pasivo en 0.0).
  - Scripts en `tools/hardware/run_*.sh`: redirección de logs de ROS 2 a `/tmp/.ros_log_user` para evitar denegación de permisos al correr bajo UID 1000.
* **Estado del Proyecto (Restricción Crítica):**
  - **Hasta este hito, el modelo aún NO ha sido evaluado en el robot físico ni se tienen resultados de inferencia en hardware real.**
  - El dataset está completamente compilado, alineado y certificado para que el entrenamiento (`train.py`) comience en la siguiente fase.

## 6. Contenedor Jetson
`deploy/jetson/Dockerfile.jetson` compila en `/opt/xarm_ws` (underlay) `xarm_ros2@ac736c92` (humble, SDK C++
`f367943`) y `control_msgs@4418bbf` (4.9.0; ausente en la imagen base), y añade `transforms3d==0.4.2` para
`pose_smoother`. Procedimiento de puesta en marcha por etapas en `deploy/jetson/README.md` (sección 3b).
