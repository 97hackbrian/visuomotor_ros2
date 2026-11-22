# SOTA Evaluation Report: `demonstrations_real_v7`

**Date:** 2026-11-22 04:38:49  
**Dataset:** `/ros/datasets/demonstrations_real_v7.zarr`  
**Sensor:** Intel RealSense D435i (Active IR Stereo + IMU) + RTAB-Map VIO  
**Total Episodes:** 83  
**Total Frames:** 28,039 (23.37 minutes @ 20 Hz)  

## 1. Global Summary Statistics Table

| Dimension / Metric | Mean Value (μ ± σ) | Median | Range [Min, Max] |
| :--- | :--- | :--- | :--- |
| **Steps per Episode** | 337.8 ± 99.4 | 315 | [226, 728] |
| **Temporal Duration (s)** | 16.89s ± 4.97s | 15.75s | [11.30s, 36.40s] |
| **Net Displacement ($D$)** | 0.600 m ± 0.081 m | - | [0.388 m, 0.757 m] |
| **Effective Path ($L_{eff}$)** | 1.144 m ± 0.287 m | - | [0.644 m, 2.298 m] |
| **Smooth Tortuosity ($L_{eff}/D$)** | 1.92 ± 0.45 | 1.85 | Direct & smooth reaching |
| **Mean Cartesian Velocity** | 0.200 m/s ± 0.054 m/s | - | Peak max: 13.645 m/s |
| **Smoothness (Log Dim. Jerk)** | 19.85 ± 0.76 | - | Expert human smoothness |
| **Terminal Stop Floor (done=1.0)** | 40.8 steps (2.04s) | 40 steps | Min: 40 steps (2.00s) |
| **Auto-Stop Compliance Rate** | **100.0%** | 100% | ≥ 40 frames guaranteed |

## 2. 3D Spatial Bounding Box (Workspace Envelope)

* **X:** [-0.540, 0.301] m (Span: 0.841 m)
* **Y:** [-0.642, 0.065] m (Span: 0.707 m)
* **Z:** [-0.588, 0.381] m (Span: 0.968 m)

## 3. Multichannel SOTA Visual Panel

![Multichannel SOTA Panel](demonstrations_real_v7_sota_panel.png)
