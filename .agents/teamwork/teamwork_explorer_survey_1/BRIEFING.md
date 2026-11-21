# BRIEFING — 2026-11-20T20:15:40Z

## Mission
Audit bio/umi.pdf (Chi et al.) and open-source UMI pipeline to extract rigorous technical evidence on Visual SLAM, optics/fisheye, scale ambiguity, batch processing, and failure modes.

## 🔒 My Identity
- Archetype: explorer
- Roles: Literature Explorer / Technical Auditor
- Working directory: /home/hackbrian/Documents/gits/visuomotor_ros2/.agents/teamwork/teamwork_explorer_survey_1/
- Original parent: eeabad66-1519-4aef-8ca9-7d5ccc498e8c
- Milestone: UMI SLAM Literature Audit & Technical Analysis

## 🔒 Key Constraints
- Read-only investigation — do NOT implement
- Strictly read-only regarding repository code and datasets
- No git commits, no source code modification
- Write outputs ONLY within /home/hackbrian/Documents/gits/visuomotor_ros2/.agents/teamwork/teamwork_explorer_survey_1/

## Current Parent
- Conversation ID: eeabad66-1519-4aef-8ca9-7d5ccc498e8c
- Updated: not yet

## Investigation State
- **Explored paths**: `bio/umi.pdf`, `arxiv:2402.10329v2` (full manuscript & appendices §A.0-A through §A.0-F), official UMI GitHub codebase (`real-stanford/universal_manipulation_interface`), `scripts_slam_pipeline/` (00 to 07), Docker container (`orb_slam3:latest`).
- **Key findings**:
  1. SLAM Backend: Monocular-Inertial ORB-SLAM3 (Campos et al., 2021) based on Steffen Urban's GoPro SLAM.
  2. Critical Modifications: "Map as Initialization" (active SLAM continuation after relocalization to handle dynamic manipulation without map freezing), "Marker-Enhanced Initialization" (fiducial tag locks $s=1.0$ and speeds up initialization during mapping), and `slam_mask.png` (excludes gripper jaws, fingers, mirrors, and operator hands).
  3. Optics: 155° fisheye via GoPro Max Lens Mod, modeled with Kannala-Brandt 8-parameter model (`KannalaBrandt8`). Raw fisheye images are passed directly to Diffusion Policy (rectifying to 69° pinhole collapses success from 100% to 55%).
  4. Side Mirrors: Symmetrically placed physical mirrors provide implicit stereo parallax. Digital reflection transform (horizontal flip + crop swap) resolves chirality inversion, improving success from 85% to 100%. Continuous finger stroke (0-80 mm) is tracked via ArUco tags.
  5. Scale Resolution: Multi-tiered via Marker PnP geometry, GoPro IMU accelerometer specific force/gravity integration, and CAD gripper stroke.
  6. Batch Workflow: Scripts 00 to 07 process MP4s offline inside Docker (~60-100s per 30s episode). Physical robot runs ZERO visual SLAM at inference time, using joint encoders/forward kinematics for relative SE(3) execution.
  7. Accuracy & Failures: 6.1 mm ATE / 10.1 mm RPE against OptiTrack MoCap. Vulnerable to relocalization dropouts, motion blur, textureless surfaces, and reflective artifacts.
- **Unexplored areas**: None regarding UMI SLAM. Investigation complete.

## Key Decisions Made
- Extracted and verified full text of `bio/umi.pdf` and arXiv:2402.10329v2 HTML appendices.
- Completed comprehensive 8-section technical report in `analysis.md`.
- Ready to author structured 5-component `handoff.md`.

## Artifact Index
- /home/hackbrian/Documents/gits/visuomotor_ros2/.agents/teamwork/teamwork_explorer_survey_1/DISPATCH.md — Dispatch log
- /home/hackbrian/Documents/gits/visuomotor_ros2/.agents/teamwork/teamwork_explorer_survey_1/BRIEFING.md — Working memory
- /home/hackbrian/Documents/gits/visuomotor_ros2/.agents/teamwork/teamwork_explorer_survey_1/progress.md — Liveness heartbeat
- /home/hackbrian/Documents/gits/visuomotor_ros2/.agents/teamwork/teamwork_explorer_survey_1/analysis.md — Technical Analysis (520 lines, verified citations)
- /home/hackbrian/Documents/gits/visuomotor_ros2/.agents/teamwork/teamwork_explorer_survey_1/handoff.md — Handoff Report
