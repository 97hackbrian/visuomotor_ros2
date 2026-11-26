# Evaluation of the v8 handheld dataset (`demonstrations_real_v8` → `demonstrations_real_v8_xarm5`)

This folder applies the same audit as `datasets/evaluations_real_v7_xarm5_v2/` to v8, so the two can be compared
(2026-11-26). The figures (PNG at 300 DPI and PDF) stay local, as in the v7 folder; the `.md` and `.json` reports are
versioned.

## v7 (dataset of the model that worked on the robot) vs v8

| Metric | v7 (`v7_xarm5_v2`) | v8 (`v8_xarm5`) |
|---|---|---|
| Demonstrations recorded → in the training set | 83 → 76 | 252 → 230 (2 days: 114 + 116) |
| Frames at 20 Hz | 23 731 | 95 429 |
| Pose source | live RTAB-Map odometry | DROID-SLAM RGB-D offline on the raw bags (live odometry only guides) |
| Per-frame timestamps | no | yes |
| Null poses stored as poses | 4 frames (2 episodes) | 0 |
| Jumps > 30 mm between frames | 926 | live odometry 709; **DROID 0** |
| Persistent discontinuities > 20 mm | 784 in 71 of 83 episodes | live 652 in 96 of 252; **DROID 0** |
| Noise removed by the trajectory filter (median RMS) | 8.8 mm | live 3.5 mm; **DROID 0.9 mm** |
| Done frames moving faster than 2 cm/s | 21.0 % (72 of 83 episodes only at the 40-frame floor) | **11.9 %** (1 of 230 at the floor) |
| Start pose | assumed HOME3 | measured (ArUco PnP + IMU): median 44 mm (day 1), 57 mm (day 2) from HOME3 |
| Initial optical-axis tilt (ArUco, independent) | 32.1° ± 5.9° | 31.4° ± 6.8° |
| Coverage | mostly the right side of the table | 9 cells × 6 blade orientations, 3–6 demonstrations per combination |
| Final TCP height p5 / p50 / p95 | 0.169 / 0.280 / 0.413 m | **0.305 / 0.335 / 0.365 m** (IQR 2.2 cm vs 9.7 cm) |
| Frames beyond the bridge reach (97 %), measured extrinsic | 3.3 % (`v7_xarm5_v3`) | **21 %** |
| Frames beyond the straight elbow (unreachable) | 0.8 % (`v7_xarm5_v3`) | **15.2 %** (80 of 230 demonstrations) |
| Demonstrations fully within the bridge reach | 64 of 75 (`v7_xarm5_v3`) | 129 of 230 (137 with ≤ 5 % of frames beyond) |
| Integrity check (`verify_zarr_dataset.py`) | pass | pass: 230/230 episodes, 95 429 images decoded, DiffusionPolicy step on CUDA (loss 1.242) |
| Consistency with the v7 training dataset (`verify_training_dataset.py`) | reference | LISTO; every episode has DROID poses (`experiments/v8_validation/demonstrations_real_v8/verify.json`) |

**Main caveat for training: reach.** In about a third of the v8 demonstrations the hand moves forward high before
descending. During that part of the approach the robot cannot follow (wrist beyond the straight elbow). The endpoints
are fine: only 12 of 230 end outside the bridge box. The converter flags only 25 demonstrations, because its threshold
is "more than 50 % of the frames outside". All are kept, so the training set contains path segments that the robot
bridge will clip. Within-reach coverage (fig. 5, right) is missing mainly 180° and ±90° at the far and side cells.

## Files

| File | Content | Tool |
|---|---|---|
| `quality_report.md`, `quality_summary.json`, `fig1_data_quality_diagnosis`, `fig2_trajectories_base_frame`, `fig3_curation_audit`, `fig5_coverage_cells_orientations` | Noise, speed, start pose and tilt (fig. 1); TCP paths, v7 vs v8 on the same axes (fig. 2); per-demonstration curation (fig. 3); coverage, all and within reach (fig. 5) | `tools/xarm5/analyze_v8_dataset_quality.py` |
| `raw_diagnostics.md/.json`, `fig4_raw_recording_diagnostics` | Raw recording with the **live odometry** poses | `tools/xarm5/analyze_vio_dataset_quality.py --raw-diagnostics` |
| `droid_poses/raw_diagnostics.md/.json`, `droid_poses/fig4_raw_recording_diagnostics` | The same diagnostics on the **DROID** poses (the ones used for training) | same tool, on the DROID build (`build_dataset_from_bags.py`) |
| `workspace_summary.json`, `fig_xarm5_workspace_vs_demos` | Reach of the xArm5 vs bridge box vs demonstrations (current limits: 0.70 m / ±0.35 m / 97 %) | `tools/xarm5/analyze_xarm5_workspace.py` |
| `workspace_summary_reference_v7_xarm5_v3.json` | The same reach analysis on v7 with the measured extrinsic (fair comparison) | same tool |
| `verification_report.json`, `verification.log` | Integrity, numerics, done labels, DataLoader and one training step | `tools/verify_zarr_dataset.py` (measured-start episodes: step 0 is the start offset from HOME3) |

## Regenerate (container, after `tools/xarm5/make_v8_training_set.sh`)

```bash
python3 tools/xarm5/build_dataset_from_bags.py --src datasets/demonstrations_real_v8.zarr --dst datasets/_work_eval_v8/demonstrations_real_v8_droid.zarr
python3 tools/xarm5/analyze_v8_dataset_quality.py --raw datasets/demonstrations_real_v8.zarr \
    --droid datasets/_work_eval_v8/demonstrations_real_v8_droid.zarr --train datasets/demonstrations_real_v8_xarm5.zarr \
    --reference datasets/demonstrations_real_v7_xarm5_v2.zarr \
    --reference-summary datasets/evaluations_real_v7_xarm5_v2/quality_summary.json --out-dir datasets/evaluations_real_v8_xarm5
python3 tools/xarm5/analyze_vio_dataset_quality.py --raw datasets/demonstrations_real_v8.zarr --raw-diagnostics --out-dir datasets/evaluations_real_v8_xarm5
python3 tools/xarm5/analyze_vio_dataset_quality.py --raw datasets/_work_eval_v8/demonstrations_real_v8_droid.zarr --raw-diagnostics --out-dir datasets/evaluations_real_v8_xarm5/droid_poses
python3 tools/xarm5/analyze_xarm5_workspace.py --dataset datasets/demonstrations_real_v8_xarm5.zarr --out-dir datasets/evaluations_real_v8_xarm5
python3 tools/verify_zarr_dataset.py --dataset datasets/demonstrations_real_v8_xarm5.zarr --out-json datasets/evaluations_real_v8_xarm5/verification_report.json --log-file datasets/evaluations_real_v8_xarm5/verification.log
rm -rf datasets/_work_eval_v8
```
