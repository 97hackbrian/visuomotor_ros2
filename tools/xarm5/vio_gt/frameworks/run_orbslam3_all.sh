#!/usr/bin/env bash
# ORB-SLAM3 (imagen visuomotor_orbslam3:v1) en modos rgbd, rgbd_inertial y mono_inertial sobre todas las corridas
# exportadas, y evaluación contra el robot (en vivo y trayectoria final). Reanudable: salta lo ya hecho.
#   tools/xarm5/vio_gt/frameworks/run_orbslam3_all.sh experiments/vio_gt/2026-11-24 [modos]
set -u
REPO=$(cd "$(dirname "$0")/../../../.." && pwd); cd "$REPO"
SESSION=$1; MODES=${2:-rgbd rgbd_inertial mono_inertial}
O=$SESSION/frameworks/orbslam3; S=$(ls -d $SESSION/runs/*static_board | head -1)
[ -f $O/d435i_settings.yaml ] || docker exec -u 1000 -w /ros visuomotor_ros2 python3 \
  tools/xarm5/vio_gt/frameworks/orbslam3/make_orbslam3_settings.py datasets/vio_frames/$(basename $S)/calib.json $O/d435i_settings.yaml
for MODE in $MODES; do
  mkdir -p $O/$MODE
  for R in $SESSION/runs/*approach*; do
    N=$(basename $R); F=datasets/vio_frames/$N; P=$O/$MODE/$N
    if [ ! -f ${P}_live.txt ] || [ ! -s ${P}_live.txt ]; then
      timeout 1800 docker run --rm -u 1000 -v "$REPO":/ros -w /ros visuomotor_orbslam3:v1 /opt/runner/build/vio_runner \
        /opt/ORB_SLAM3/Vocabulary/ORBvoc.txt $O/d435i_settings.yaml $MODE $F $P > ${P}_stdout.log 2>&1
    fi
    docker exec -u 1000 -w /ros visuomotor_ros2 bash -c "python3 tools/xarm5/vio_gt/frameworks/orbslam3/orbslam3_to_tum.py $P $F/calib.json > /dev/null && source /opt/ros/jazzy/setup.bash && \
      for k in live final; do [ -f ${P}_\$k.tum ] && [ -s ${P}_\$k.tum ] && python3 tools/xarm5/vio_gt/evaluate_vio_gt.py --run $R --hand-eye $S/hand_eye.json \
        --traj-tum ${P}_\$k.tum --traj-frame optical --frame-times $F/rgb.txt --label orbslam3_${MODE}_\$k 2>&1 | grep 'ATE anclada' | sed \"s|^|[$N] $MODE \$k: |\"; done"
  done
done
