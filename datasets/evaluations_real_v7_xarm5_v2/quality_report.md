# Data-quality audit: `demonstrations_real_v7` → `demonstrations_real_v7_xarm5_v2`

* Episodes: 83 raw → 76 kept (dropped: episode_23, episode_24, episode_25, episode_30, episode_31, episode_32, episode_35).
* VIO noise removed (median RMS): 8.8 mm; filter: Hampel + 1.5 Hz zero-phase Butterworth (order 4).
* Mean TCP speed: raw 0.197 m/s → filtered 0.050 m/s (p99 0.138 m/s; bridge limit 0.05 m/s).
* Gravity misalignment at t=0 (median): Assumed mount rpy (0, -π/2, 0): 57.1°; Data-consistent mount rpy (π, -π/2, 0): 13.4°.
* ArUco (independent) optical-axis tilt at t=0: 32.1° ± 5.9° (n=59), image roll 0.1° (upright); HOME3 camera axis 24.6°. VIO-based tilt reaches 87° because the odometry frame is not reset between episodes and drifts.

![Diagnosis](fig1_vio_data_quality_diagnosis.png)

![Trajectories](fig2_trajectories_base_frame.png)

![Curation](fig3_curation_audit.png)
