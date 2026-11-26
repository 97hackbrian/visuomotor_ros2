# Data-quality audit: `demonstrations_real_v8` → `demonstrations_real_v8_xarm5`

* Demonstrations: 252 recorded → 230 in the training set (95429 frames; per day: 2026-11-25 114, 2026-11-26 116). Excluded: 19 discarded (operator or validation), 3 with a non-physical DROID jump (episode_237, episode_248, episode_249); 35 kept as REVISAR (mostly the final hold still moving).
* Poses: DROID-SLAM RGB-D offline on the raw bags (the live RTAB-Map odometry only guides the operator). Live odometry lost tracking in 0.5 % of the frames (median per demonstration); live vs DROID disagreement p90: median 11.0 mm.
* Noise removed by the trajectory filter (median RMS): live odometry 3.5 mm, DROID 0.9 mm (v7, live VIO: 8.8 mm).
* Median camera speed: live 0.049 m/s, DROID 0.036 m/s, DROID + filter 0.034 m/s (p99 0.113 m/s; robot bridge limit 0.05 m/s).
* Real start pose vs HOME3 (median TCP offset): 2026-11-25 44 mm; 2026-11-26 57 mm. Every demonstration is anchored with its measured start pose (ArUco PnP + IMU).
* Initial optical-axis tilt from the ArUco tags: 31.4° ± 6.8° (n=151); v7: 32.1° ± 5.9°. The odometry is reset per demonstration: no drift.
* Coverage: 9 cells × 6 blade orientations; per combination 3–6 demonstrations in the training set, 0–6 within the robot's reach.
* Reach: 137 of 230 demonstrations stay within the bridge reach (≤ 5 % of frames beyond it); 80 have frames beyond the straight elbow (physically unreachable), mostly the start of the approach (the hand moves forward high before descending). The converter flags only 25 (> 50 % of the frames outside); all are kept in the training set.

![Diagnosis](fig1_data_quality_diagnosis.png)

![Trajectories](fig2_trajectories_base_frame.png)

![Curation](fig3_curation_audit.png)

![Coverage](fig5_coverage_cells_orientations.png)
