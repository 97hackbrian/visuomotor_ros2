#!/usr/bin/env bash
# Etapa C (DENTRO del contenedor de la Jetson): armar el puente en modo real, lanzar la política, vigilar y parar.
# Lo llama tools/xarm5/real_trials/run_trial.sh después del chequeo previo. Salidas en /tmp/vtrials.
R=$(timeout 15 ros2 service call /xarm5_bridge/arm std_srvs/srv/Trigger 2>&1 | grep -E "message")
echo "arm: $R"
echo "$R" | grep -q "servo mode" || { echo "ABORTO: el puente no se armó en modo servo"; exit 2; }
# setsid: la política en su propio grupo de procesos (un proceso en segundo plano de un script ignora SIGINT;
# se detiene con SIGTERM/SIGKILL al grupo y se verifica que no quede ningún publicador)
setsid ros2 run visuomotor_ros2 policy_node --ros-args \
    --params-file /ros_ws/install/visuomotor_ros2/share/visuomotor_ros2/config/policy_params.yaml \
    -p checkpoint_path:=/ros_ws/model > /tmp/vtrials/policy.log 2>&1 &
PG=$!
python3 /tmp/vtrials/monitor.py --max-s "${MAX_S:-45}" --z-min "${Z_MIN:-0.15}"
# Desarme redundante (el vigilante ya desarma; esto cubre un fallo del vigilante). No pone frenos.
timeout 10 ros2 service call /xarm5_bridge/disarm std_srvs/srv/Trigger 2>&1 | grep -E "message" | sed 's/^/desarme redundante: /'
kill -TERM -- -$PG 2>/dev/null; for i in $(seq 1 20); do kill -0 -- -$PG 2>/dev/null || break; sleep 0.5; done
kill -0 -- -$PG 2>/dev/null && { echo "forzando cierre de la política"; kill -KILL -- -$PG; }
sleep 1
python3 -c "import rclpy, time; from rclpy.node import Node; rclpy.init(); n=Node('chk'); time.sleep(1.5); print('publicadores /target_frame_raw tras parar:', n.count_publishers('/target_frame_raw')); n.destroy_node(); rclpy.shutdown()"
timeout 8 ros2 topic echo --once /xarm5_bridge/status 2>/dev/null | head -1
grep -E "CUDA Graph|RESET finalizado|Auto-Stop Neuronal|ERROR" /tmp/vtrials/policy.log | cut -c1-170 | head -8
