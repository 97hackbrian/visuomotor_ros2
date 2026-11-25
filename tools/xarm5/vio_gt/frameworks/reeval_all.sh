#!/usr/bin/env bash
# Reevalúa contra el robot TODAS las fuentes de odometría de una sesión, con las mismas reglas (evaluate_vio_gt.py),
# y escribe el resumen comparativo. Fuentes que no existan se saltan.
#   tools/xarm5/vio_gt/frameworks/reeval_all.sh experiments/vio_gt/2026-11-24
set -u
REPO=$(cd "$(dirname "$0")/../../../.." && pwd); cd "$REPO"
SESSION=$1; S=$(ls -d $SESSION/runs/*static_board | head -1); HE=$S/hand_eye.json
FW=$SESSION/frameworks
EV="python3 tools/xarm5/vio_gt/evaluate_vio_gt.py --hand-eye $HE"
LABELS="live"
for R in $SESSION/runs/*approach*; do
  N=$(basename $R); BAG=$(basename $(cat $R/bag_path.txt)); F=datasets/vio_frames/$N
  CMDS="$EV --run $R --label live"
  for C in baseline bundle_adjustment guess_motion optical_flow frame_to_frame no_kalman; do
    [ -d $SESSION/replay/$BAG/$C/odom ] && CMDS="$CMDS
$EV --run $R --odom-bag $SESSION/replay/$BAG/$C/odom --odom-topic /replay/odom --label $C"
  done
  for M in rgbd rgbd_inertial mono_inertial; do for K in live final; do
    T=$FW/orbslam3/$M/${N}_$K.tum
    [ -s $T ] && [ $(wc -l < $T) -gt 50 ] && CMDS="$CMDS
$EV --run $R --traj-tum $T --traj-frame optical --frame-times $F/rgb.txt --label orbslam3_${M}_$K"
  done; done
  [ -s $FW/open3d/$N.tum ] && CMDS="$CMDS
$EV --run $R --traj-tum $FW/open3d/$N.tum --traj-frame optical --frame-times $F/rgb.txt --label open3d_rgbd"
  [ -s $FW/droid/$N.tum ] && CMDS="$CMDS
$EV --run $R --traj-tum $FW/droid/$N.tum --traj-frame optical --frame-times $F/rgb.txt --label droid_rgbd"
  # DPVO (monocular) excluido: sin memoria en 8 GB de GPU salvo en la corrida corta; ver context.md, lección 48
  false && CMDS="$CMDS
$EV --run $R --traj-tum $FW/dpvo/$N.tum --traj-frame optical --frame-times $F/rgb.txt --scale-free --label dpvo_mono_scaled"
  echo "$CMDS" | while read -r CMD; do
    docker exec -u 1000 -w /ros visuomotor_ros2 bash -c "source /opt/ros/jazzy/setup.bash && $CMD" 2>&1 \
      | grep -E "ATE anclada|Error|Traceback" | sed "s|^|[$N] $(echo $CMD | grep -o -- '--label [a-z0-9_]*' | cut -d' ' -f2): |"
  done
done
ALL="live,baseline,bundle_adjustment,guess_motion,optical_flow,frame_to_frame,no_kalman,orbslam3_rgbd_live,orbslam3_rgbd_final,orbslam3_rgbd_inertial_live,orbslam3_mono_inertial_live,open3d_rgbd,droid_rgbd"
docker exec -u 1000 -w /ros visuomotor_ros2 python3 tools/xarm5/vio_gt/summarize_vio_gt.py --session $SESSION --labels $ALL \
  | sed -n '/## Comparación/,$p'
