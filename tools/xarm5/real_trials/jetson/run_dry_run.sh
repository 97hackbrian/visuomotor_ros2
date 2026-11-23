#!/usr/bin/env bash
# Etapa B (DENTRO del contenedor): con el puente en dry-run, armar (no envía comandos), lanzar la política 25 s,
# registrar command_preview / target_frame_raw / done_flag / TCP medido, parar y desarmar. El brazo no se mueve.
ST=$(timeout 8 ros2 topic echo --once /xarm5_bridge/status 2>/dev/null | head -1)
echo "estado: $ST"
echo "$ST" | grep -q "dry_run=True" || { echo "ABORTO: el puente NO está en dry-run"; exit 2; }
echo "$ST" | grep -q " err=0" || { echo "ABORTO: el controlador reporta error"; exit 3; }
timeout 15 ros2 service call /xarm5_bridge/arm std_srvs/srv/Trigger 2>&1 | grep -E "message"
setsid ros2 run visuomotor_ros2 policy_node --ros-args \
    --params-file /ros_ws/install/visuomotor_ros2/share/visuomotor_ros2/config/policy_params.yaml \
    -p checkpoint_path:=/ros_ws/model > /tmp/vtrials/policy_dry_run.log 2>&1 &
PG=$!
for i in $(seq 1 90); do grep -q "RESET finalizado" /tmp/vtrials/policy_dry_run.log && break; sleep 1; done
python3 /tmp/vtrials/record_dry_run.py 25
kill -TERM -- -$PG 2>/dev/null; for i in $(seq 1 20); do kill -0 -- -$PG 2>/dev/null || break; sleep 0.5; done
kill -0 -- -$PG 2>/dev/null && kill -KILL -- -$PG
timeout 15 ros2 service call /xarm5_bridge/disarm std_srvs/srv/Trigger 2>&1 | grep -E "message"
timeout 8 ros2 topic echo --once /xarm5_bridge/status 2>/dev/null | head -1
