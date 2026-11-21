# BRIEFING — 2026-11-20T20:13:30Z

## Mission
Audit in depth the active RealSense D435i + RTAB-Map VIO configuration, parameters, node architecture, launch files, telemetry benchmarks, and integration with data collection in visuomotor_ros2.

## 🔒 My Identity
- Archetype: Explorer
- Roles: Active Repo VIO Explorer, System Auditor, Synthesis
- Working directory: /home/hackbrian/Documents/gits/visuomotor_ros2/.agents/teamwork/teamwork_explorer_survey_3
- Original parent: eeabad66-1519-4aef-8ca9-7d5ccc498e8c
- Milestone: VIO Architecture & Empirical Telemetry Survey

## 🔒 Key Constraints
- Read-only investigation — do NOT implement
- Strictly READ-ONLY regarding repository code and datasets
- DO NOT modify, create, or delete any source code files or datasets
- DO NOT make any git commits
- Write outputs ONLY within /home/hackbrian/Documents/gits/visuomotor_ros2/.agents/teamwork/teamwork_explorer_survey_3/

## Current Parent
- Conversation ID: eeabad66-1519-4aef-8ca9-7d5ccc498e8c
- Updated: 2026-11-20T20:07:30Z

## Investigation State
- **Explored paths**:
  - `ORIGINAL_REQUEST.md`
  - `AGENTS.md` (Sections 17, 19, 20)
  - `.agents/skills/realsense-vio-data-collection/SKILL.md`
  - `.agents/knowledge/15_visual_odometry_rtabmap_realsense.md`
  - `visuomotor_bringup/config/realsense/d435i_camera_params.yaml`
  - `visuomotor_bringup/config/realsense/rtabmap_vio_params.yaml`
  - `visuomotor_bringup/config/realsense/rtabmap_vio_params_best_umi.yaml`
  - `visuomotor_bringup/launch/realsense_d435i_vio.launch.py`
  - `visuomotor_bringup/launch/realsense_d435i_vio_collection.launch.py`
  - `visuomotor_data_collection/config/vo_data_collection_params.yaml`
  - `visuomotor_data_collection/visuomotor_data_collection/vo_collector_node.py`
  - `experiments/vio_benchmark_history.json`
  - `tools/benchmark_vio_pose.py`
  - `tools/hardware/run_benchmark.sh`
  - `tools/hardware/run_vo_container.sh`
  - `tools/hardware/run_data_collection.sh`
  - `tools/hardware/benchmark_vo.py`
  - `tools/hardware/setup_realsense_udev.sh`
  - `tests/test_relative_coordinates.py`
  - `tools/verify_dataset_frequency.py`
- **Key findings**:
  - RealSense D435i operates with active IR stereo projector (`emitter_enabled: 1`), 50 mm baseline, 640x480 at 30 FPS, hardware timestamp sync, and depth aligned to color.
  - Bosch BMI085 IMU sampled at 200 Hz with linear interpolation unification (`unite_imu_method: 2`).
  - Madgwick filter calibrated with gain $\beta=0.035$, drift damping $\zeta=0.002$, world frame ENU, gravity initialization (`wait_imu_to_init: true`).
  - RTAB-Map `rgbd_odometry`: Frame-to-Map (`Odom/Strategy: 0`, `MaxSize: 5000`), GFTT/ORB (`FeatureType: 8`, 1200 features), subpixel interpolation ($5\times 5$, 15 iters), `BundleAdjustment: 0` (preserves native physical metric scale without scale drift), `GuessMotion: false` (eradicates phantom inertial drag and yaw drift), Kalman filter enabled, origin strictly at lens $(0,0,0)$ (`AlignWithGround: false`), depth clamped to 0.20–1.80 m.
  - Certified empirical benchmark (`conf 4 — SOTA UMI Final Fusion`): return error of $2.307\text{ mm}$ ($\le 3.00\text{ mm}$ criterion), resting jitter $1.6144\text{ mm}$ (89% reduction vs baseline), $0.0\%$ tracking loss, $390\text{--}450$ inliers, $13.98\text{ ms}$ CPU computation per frame.
  - Integration with `vo_collector_node.py`: 20.0 Hz timer, in-memory RAM resizing to $256\times 256$, invariant Delta-Home coordinates (step 0 at $(0,0,0)$), upper hemisphere quaternion ($q_w \ge 0$), 8D state without causal leakage (`done` excluded), 9D action with guaranteed $\ge 2.0\text{ s}$ (40 frames) auto-stop floor.
  - Network shielding: `ROS_LOCALHOST_ONLY=1` under `ROS_DOMAIN_ID=0` confines $46\text{ MB/s}$ to SHM/loopback, preventing Wi-Fi saturation.
- **Unexplored areas**:
  - None within the scope of Explorer 3. Full audit completed.

## Key Decisions Made
- Fully documented all 5 thematic areas with exact file paths and line numbers.
- Generated `analysis.md` and `handoff.md` following the 5-component handoff standard.

## Artifact Index
- `analysis.md` — Detailed technical analysis with line references and architecture tables
- `handoff.md` — 5-component handoff report (Observation, Logic Chain, Caveats, Conclusion, Verification Method)
- `progress.md` — Liveness heartbeat and milestone tracking
- `DISPATCH.md` — Inbound message log
