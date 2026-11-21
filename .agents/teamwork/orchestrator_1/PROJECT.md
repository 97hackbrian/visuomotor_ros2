# Project: SOTA Visual SLAM / VIO Research & Comparison (UMI vs RTAB-Map VIO)

## Architecture
- Target Research Document: `docs/research/umi_vs_rtabmap_vio_comparison.md`
- Literature Sources: `bio/umi.pdf`, `bio/pepper_harvesting.pdf`, `bio/fruit_bearing.pdf`, `bio/action_diffusion.pdf`, `bio/dp3d.pdf`, `bio/dppo.pdf`
- Codebase & Config References:
  - `visuomotor_bringup/config/realsense/d435i_camera_params.yaml`
  - `visuomotor_bringup/config/realsense/rtabmap_vio_params.yaml`
  - `visuomotor_bringup/launch/realsense_d435i_vio.launch.py`
  - `visuomotor_bringup/launch/realsense_d435i_vio_collection.launch.py`
  - `visuomotor_data_collection/visuomotor_data_collection/vo_collector_node.py`
  - `.agents/skills/realsense-vio-data-collection/SKILL.md`
  - `AGENTS.md` (Section 17: Odometría Visual e Inercial con RTAB-Map)

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | UMI SLAM Pipeline Deep Audit | Extraction of SLAM backend (ORB-SLAM3/DROID-SLAM/COLMAP), fisheye optics, mirror gripper tracking, scale ambiguity resolution, offline batching | M1 | bio/umi.pdf & UMI literature |
| 2 | Harvesting & Non-structured SOTA Audit | Extraction of sensing, odometry, stereovision vs monocular, occlusion management, outdoor unstructured robotics | M1 | bio/pepper_harvesting.pdf & bio/fruit_bearing.pdf |
| 3 | Active Repo VIO Architecture Audit | Extraction of RealSense D435i + IMU BMI085 (200Hz Madgwick), RTAB-Map rgbd_odometry (20Hz F2M, OdomF2M/BA=0, Odom/GuessMotion=false), Kalman filter, return error 2.307mm, jitter 1.61mm | M1 | Active codebase & telemetry |
| 4 | Comparative Matrix & Analysis | Multi-axial technical comparison: latency/frequency, metric scale, drift & return error, sensor robustness, mechanical overhead, portability, DP compatibility | M2 | Synthesis of M1 |
| 5 | Full Technical Report Generation | Comprehensive academic/engineering report in `docs/research/umi_vs_rtabmap_vio_comparison.md` adhering to rigorous scientific standards | M2 | Worker implementation |
| 6 | Adversarial Technical Review & Verification | Empirical and theoretical review of claims, citations, mathematical justifications and matrix | M3 | Reviewers & Challengers |
| 7 | Integrity & Forensic Audit | Verification that no code files were modified, no git commits made, report is authentic and exhaustive | M4 | teamwork_auditor |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Survey & Technical Investigation | Parallel exploration of papers and active repository | none | DONE |
| 2 | Technical Report Compilation | Synthesis and drafting of `docs/research/umi_vs_rtabmap_vio_comparison.md` | M1 | DONE |
| 3 | Peer Review & Adversarial Verification | Dual review and challenge of findings, metrics, and conclusions | M2 | DONE |
| 4 | Integrity Forensic Audit & Gate | Verification of non-violation of code, authentic synthesis | M3 | DONE |

## Code Layout & Permissions
- Working metadata: `.agents/teamwork/*`
- Deliverable target file: `docs/research/umi_vs_rtabmap_vio_comparison.md`
- Restricted files (NO MODIFICATION ALLOWED):
  - `visuomotor_core/**`
  - `visuomotor_bringup/**`
  - `visuomotor_ros2/**`
  - `visuomotor_data_collection/**`
  - `visuomotor_msgs/**`
  - Git repository state (`.git/**`)
