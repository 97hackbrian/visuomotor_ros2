# Model card — `train_real_v7x5v2_relative_ee_e300` (epoch 200)

**Status: trained and verified offline / in ROS 2 simulation only. NOT yet evaluated on the physical xArm5.**

| Item | Value |
|---|---|
| Dataset | `demonstrations_real_v7_xarm5_v2` (68 train / 8 held-out episodes) |
| Task | approach the pruning shears on the table from HOME3 (no grasp) |
| Action representation | `relative_ee` (UMI-style, relative to the current EE pose) |
| Quaternion convention | `home` (reference [0.9768, -0.0098, 0.213, 0.0207]) |
| Training | 300 epochs x 329 steps, batch 64, lr 1.0e-04 warmup+cosine, EMA True, ColorJitter True |
| Held-out ADE / ADE executed / FDE | 11.6 / 7.6 / 18.5 mm (hold-pose baseline 18.6 / 11.0 mm) |
| Held-out orientation error | 1.74 deg |
| Held-out done F1 / progress ratio | 0.70 / 1.00 |
| Replay (Jetson path) command error | 11.6 mm mean, 24.2 mm p95, 1.70 deg |
| Auto-stop (replay, default params) | rate 100%, median delay -0.6 s, stop-to-goal 10.6 mm |
| Recommended auto-stop (set in jetson_policy_params.yaml) | threshold 0.6, patience 5 steps → replay: 100% stops, 0 premature (>1 s), median delay -0.28 s, stop-to-goal 8 mm |
| ROS 2 closed loop (simulated bridge) | direction cos sim 0.99, auto-stop True, bridge rejections 0 |

Files: `model.safetensors`, `config.json` (carries action representation and quaternion convention), `training_meta.json`. Report figure: `final_model_report.png`.

## Deployment requirements (see deploy/jetson/README.md)
* Same D435i mounted on the flange, optical axis along the tool, image **upright at HOME3** (top of image pointing away
  from the robot base), 640x480 colour stream; `camera_topic` must match the Jetson driver.
* Jetson parameters: `deploy/jetson/config/jetson_policy_params.yaml` (`quaternion_hemisphere: auto`,
  `image_resize_method: cv2_area`, `relative_coordinates: true`, HOME3 home pose, auto-stop 0.6 / 5).
* Code: rebuild the Jetson image (or use `docker-compose.dev.yml`) so `visuomotor_core` supports `relative_ee`.
* Start in dry-run, then arm at 5 cm/s; keep the object within |y| <= 0.30 m (bridge box).
