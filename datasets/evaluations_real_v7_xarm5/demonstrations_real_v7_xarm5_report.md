# SOTA Evaluation Report: `demonstrations_real_v7_xarm5`

**Date:** 2026-11-22 04:15:12  
**Dataset:** `/ros/datasets/demonstrations_real_v7_xarm5.zarr`  
**Sensor:** Intel RealSense D435i (Active IR Stereo + IMU) + RTAB-Map VIO  
**Total Episodes:** 81  
**Total Frames:** 26,868 (22.39 minutes @ 20 Hz)  

## 1. Global Summary Statistics Table

| Dimension / Metric | Mean Value (μ ± σ) | Median | Range [Min, Max] |
| :--- | :--- | :--- | :--- |
| **Steps per Episode** | 331.7 ± 92.6 | 309 | [226, 728] |
| **Temporal Duration (s)** | 16.59s ± 4.63s | 15.45s | [11.30s, 36.40s] |
| **Net Displacement ($D$)** | 0.600 m ± 0.082 m | - | [0.388 m, 0.757 m] |
| **Effective Path ($L_{eff}$)** | 1.123 m ± 0.255 m | - | [0.644 m, 2.298 m] |
| **Smooth Tortuosity ($L_{eff}/D$)** | 1.88 ± 0.40 | 1.84 | Direct & smooth reaching |
| **Mean Cartesian Velocity** | 0.199 m/s ± 0.055 m/s | - | Peak max: 1.810 m/s |
| **Smoothness (Log Dim. Jerk)** | 19.76 ± 0.52 | - | Expert human smoothness |
| **Terminal Stop Floor (done=1.0)** | 40.8 steps (2.04s) | 40 steps | Min: 40 steps (2.00s) |
| **Auto-Stop Compliance Rate** | **100.0%** | 100% | ≥ 40 frames guaranteed |

## 2. 3D Spatial Bounding Box (Workspace Envelope)

* **X:** [-0.194, 0.295] m (Span: 0.489 m)
* **Y:** [-0.325, 0.308] m (Span: 0.633 m)
* **Z:** [-0.720, 0.017] m (Span: 0.736 m)

## 3. Multichannel SOTA Visual Panel

![Multichannel SOTA Panel](demonstrations_real_v7_xarm5_sota_panel.png)
