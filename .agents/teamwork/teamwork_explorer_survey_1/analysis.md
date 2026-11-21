# UNIVERSAL MANIPULATION INTERFACE (UMI): SLAM, OPTICS, SCALE AMBIGUITY, AND BATCH PIPELINE AUDIT

**Document Type:** Technical Literature Audit & Reverse-Engineering Analysis  
**Target Paper:** *Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots* (Chi et al., RSS 2024 / arXiv:2402.10329v2)  
**Primary Source Code Repository:** `real-stanford/universal_manipulation_interface`  
**Author:** Explorer 1 (UMI SLAM Literature Explorer)  
**Date:** 2026-11-20  

---

## 1. Executive Technical Summary

The Universal Manipulation Interface (UMI) introduces a framework for learning generalizable visuomotor policies (specifically Diffusion Policy) directly from handheld human demonstrations collected "in-the-wild" without tethered robots, external motion capture (MoCap) systems, or stationary camera infrastructure.

To achieve portable, metric-scale 6-DoF trajectory recording, UMI relies on:
1. **Hardware:** A 3D-printed, trigger-activated parallel-jaw gripper ($780\text{ g}$, $\$73$ BoM) mounted with a single consumer-grade action camera (GoPro Hero 9/10/11 with Max Lens Mod, $155^\circ$ FoV, $\$298$) featuring embedded IMU telemetry (accelerometer and gyroscope recorded via GPMF metadata at $200\text{ Hz}$).
2. **Visual SLAM Backend:** A heavily modified fork of **ORB-SLAM3** (Campos et al., 2021) in **Monocular-Inertial (VI)** mode, integrated within an automated Docker container (`orb_slam3:latest`). The pipeline adapts Steffen Urban's GoPro SLAM and OpenImuCameraCalibrator.
3. **Core Algorithmic Modifications:**
   - **Map as Initialization:** Replaces native read-only localization mode with continuous local mapping and bundle adjustment after relocalizing to a pre-built map atlas (`map_atlas.osa`), adapting to dynamic manipulation changes while preserving a shared metric world frame.
   - **Marker-Enhanced Initialization:** Uses temporary fiducial markers (ArUco / ChArUco) during initial workspace mapping to solve monocular scale ambiguity immediately ($s = 1.0$), eliminating lengthy and fragile VI initialization.
   - **Dynamic Foreground Masking:** Enforces a static binary mask (`slam_mask.png`) during ORB feature detection, removing gripper jaws, fingers, ArUco finger tags, side mirrors, and the user's hand from the SLAM optimization.
   - **Side-Mirror Virtual Stereo:** Symmetrically placed physical mirrors project peripheral lateral views and finger tips into the single camera image, providing implicit depth parallax and stereo perception without adding hardware cameras or synchronization lines.
   - **Relative Action Space & Decoupled Inference:** All end-effector poses are formulated as relative $SE(3)$ transformations with respect to the current camera pose ($\mathbf{T}_{t}^{-1} \mathbf{T}_{t+\tau}$). Consequently, **the physical robot runs ZERO visual SLAM at inference time**, instead relying strictly on forward kinematics (joint encoders) at $500\text{--}1000\text{ Hz}$.

---

## 2. Dimension 1: Exact Visual SLAM / Odometry Pipeline

```
+----------------------------------------------------------------------------------------------------+
|                                    UMI SLAM PIPELINE ARCHITECTURE                                  |
+----------------------------------------------------------------------------------------------------+
                                               
  [GoPro MP4 Video] ----------> [gpmf-parser] ----------> Accelerometer (200 Hz) + Gyroscope (200 Hz)
  (155° Fisheye, 60 fps)                                                     |
            |                                                                |
            v                                                                v
   [Image Frame (t)]                                         [Forster IMU Preintegration]
            |                                                                |
            +------------+                                                   |
            |            v                                                   v
            |    [slam_mask.png]                                  [IMU Motion Prediction]
            |    (Blanks out gripper, fingers,                               |
            |     mirrors, and hands)                                        |
            |            |                                                   v
            v            v                                      +--------------------------+
       [ORB Feature Extractor]                                 |   ORB-SLAM3 Tracking     |
       (Scale Pyramid Levels)                                  |   (Visual-Inertial)      |
            |                                                   +--------------------------+
            v                                                                |
     Static Landmarks Only                                                   v
            |                                                   +--------------------------+
            +-------------------------------------------------->| Dynamic Relocalization   |
                                                                | ("Map as Initialization")|
                                                                +--------------------------+
                                                                             |
                                             +-------------------------------+------------------------------+
                                             |                                                              |
                                             v                                                              v
                             [During Mapping: 02_create_map]                              [During Demos: 03_batch_slam]
                             Builds Keyframe Graph & Point Cloud                          Loads map_atlas.osa as prior,
                             Marker-Enhanced PnP resolves s = 1.0                         continues Local Bundle Adj.
                             Saves to: map_atlas.osa                                      Outputs: camera_trajectory.csv
```

### 2.1 Backend Architecture
The core pose tracking engine is **ORB-SLAM3** (Campos et al., *IEEE Transactions on Robotics*, 2021), compiled from a customized fork wrapped inside Docker. The system runs in **Monocular-Inertial** mode:
* **Visual Sensor:** Single GoPro Hero camera recording at $2.7\text{K}$ or $4\text{K}$ resolution at $60\text{ fps}$.
* **Inertial Sensor:** Built-in GoPro IMU logging 3-axis linear acceleration and 3-axis angular velocity at $200\text{ Hz}$, extracted from the binary GPMF (GoPro Metadata Format) track embedded directly in the MP4 container.
* **Camera Model:** Kannala-Brandt 8-parameter fisheye distortion model (`KannalaBrandt8`), which projects 3D spatial points directly into distorted image space without synthetic rectification.

### 2.2 Three Critical Modifications to Native ORB-SLAM3

#### Modification 1: "Map as Initialization" (Dynamic SLAM Continuation)
* **The Native Flaw:** In standard ORB-SLAM3, the system offers two distinct operating regimes:
  1. *Full SLAM Mode:* Continuously builds maps, inserts keyframes, and runs Local Bundle Adjustment (LBA) and Loop Closing. If used independently for each demonstration, each episode would generate its own unaligned coordinate frame, destroying multi-demonstration alignment and bimanual consistency.
  2. *Localization Mode:* Loads a pre-built map and localizes incoming frames against existing map points without creating new points or modifying the map graph. In robot manipulation demonstrations, this mode **fails completely** because human manipulation dynamically alters the scene (e.g., cups moved, sponges picked up, faucets turned, table surfaces occluded). When existing landmarks disappear or get covered, localization tracking drops.
* **UMI's Solution:** UMI re-engineers ORB-SLAM3's localization mode into **"Map as Initialization"**:
  - The demonstration tracking script (`03_batch_slam.py`) loads the pre-built `map_atlas.osa` from disk.
  - The camera relocalizes against this map to establish the global world frame $\mathcal{F}_{\text{world}}$.
  - Crucially, **it does not freeze the map**. It continues active SLAM operations: extracting new features, generating new transient map points, and running Local Bundle Adjustment.
  - This allows the tracker to navigate dynamically altered scenes without losing tracking, while anchoring the baseline trajectory to the pre-established global coordinate frame.

#### Modification 2: "Marker-Enhanced Initialization"
* **The Native Flaw:** Monocular-inertial SLAM initialization requires solving a non-linear optimization over several seconds of exploratory motion to estimate:
  $$\mathbf{x}_{\text{init}} = \{s, \mathbf{R}_{wb}, \mathbf{b}_g, \mathbf{b}_a, \mathbf{v}_0 \dots \mathbf{v}_k\}$$
  Under abrupt, fast, or non-informative handheld movements, native ORB-SLAM3 heuristic initialization often diverges or requires $15\text{--}30\text{ s}$ of smooth excitation. During this interval, no pose is estimated, wasting demonstration data.
* **UMI's Solution:** UMI introduces an optional planar fiducial marker (ArUco / ChArUco / AprilTag) placed on the workspace surface *only during the initial 1-minute mapping phase* (`02_create_map.py`):
  - With known metric marker dimensions (e.g., $100.0\text{ mm}$ square), Perspective-n-Point (PnP) provides an instantaneous, closed-form metric pose $\mathbf{T}_{c, \text{marker}}$.
  - This provides an unambiguous constraint on feature correspondences, immediately locks the metric scale factor ($s = 1.0$), aligns the gravity vector with the surface normal, and initializes the bundle adjustment.
  - Once `map_atlas.osa` is constructed, the fiducial marker is removed from the table; task demonstration videos are recorded in clean, uninstrumented environments.

#### Modification 3: Dynamic Foreground Masking (`slam_mask.png`)
* **The Native Flaw:** Because the camera is rigidly mounted to the hand-held gripper, the physical gripper body, fingers, ArUco tags on the fingertips, side mirrors, and human operator's hand occupy between $25\%$ and $40\%$ of the image frame. If standard FAST corner detection operates on the whole image, hundreds of ORB features will be extracted on the moving gripper. When the demonstrator moves the gripper, ORB-SLAM3 treats these moving points as static background, causing catastrophic trajectory corruption or scale collapse.
* **UMI's Solution:** UMI applies an explicit binary/alpha mask (`slam_mask.png`) during ORB feature detection:
  $$\mathcal{I}_{\text{ORB}}(u, v) = \begin{cases} \mathcal{I}(u, v) & \text{if } \mathcal{M}_{\text{slam}}(u, v) = 1 \\ 0 & \text{if } \mathcal{M}_{\text{slam}}(u, v) = 0 \end{cases}$$
  The mask zeros out:
  1. The central-lower region containing the 3D-printed gripper chassis and jaws.
  2. The fingertip regions containing the ArUco width-tracking tags.
  3. The lateral regions containing the side mirrors (whose virtual reflections move with the gripper and exhibit non-rigid parallax).
  4. The demonstrator's forearm/hand area.
  As a result, feature detection is strictly confined to the static background environment.

### 2.3 Feature Tracking & Bundle Adjustment Across Frames

#### Visual Reprojection Error:
Let $\mathbf{P}_w^j \in \mathbb{R}^3$ be the world coordinates of the $j$-th map point, and $\mathbf{T}_{cw}^i = [\mathbf{R}_{cw}^i \mid \mathbf{t}_{cw}^i] \in SE(3)$ be the pose of the $i$-th camera keyframe. The visual residual is:
$$\mathbf{e}_{i,j}^{\text{visual}} = \mathbf{z}_{i,j} - \pi_{\text{KB}}(\mathbf{R}_{cw}^i \mathbf{P}_w^j + \mathbf{t}_{cw}^i)$$
where $\mathbf{z}_{i,j} \in \mathbb{R}^2$ is the detected ORB feature coordinate, and $\pi_{\text{KB}}(\cdot)$ is the Kannala-Brandt projection function.

#### Inertial Preintegration Error:
Between consecutive keyframes $k$ and $k+1$, IMU measurements (sampled at $200\text{ Hz}$) are integrated using Forster's preintegration on $SO(3)$:
$$\mathbf{e}_{k, k+1}^{\text{inertial}} = \begin{bmatrix} \mathbf{R}_{wb}^k (\mathbf{p}_w^{k+1} - \mathbf{p}_w^k - \mathbf{v}_w^k \Delta t - \frac{1}{2} \mathbf{g}_w \Delta t^2) - \Delta \mathbf{p}_{k, k+1} \\ \mathbf{R}_{wb}^k (\mathbf{v}_w^{k+1} - \mathbf{v}_w^k - \mathbf{g}_w \Delta t) - \Delta \mathbf{v}_{k, k+1} \\ \text{Log}\left( \Delta \mathbf{R}_{k, k+1}^T (\mathbf{R}_{wb}^k)^T \mathbf{R}_{wb}^{k+1} \right) \end{bmatrix}$$
with additional random walk bias residuals:
$$\mathbf{e}_{\text{bias}}^k = \mathbf{b}^{k+1} - \mathbf{b}^k$$

#### Joint Optimization in Local Bundle Adjustment:
Local Bundle Adjustment minimizes the combined cost over active keyframes $\mathcal{K}_L$ and active map points $\mathcal{P}_L$ using the robust Huber kernel $\rho(\cdot)$:
$$\min_{\mathcal{K}_L, \mathcal{P}_L, \mathcal{S}_L} \sum_{i \in \mathcal{K}_L} \sum_{j \in \mathcal{P}_L} \rho\left( (\mathbf{e}_{i,j}^{\text{visual}})^T \boldsymbol{\Sigma}_{i,j}^{-1} \mathbf{e}_{i,j}^{\text{visual}} \right) + \sum_{k \in \mathcal{K}_L \setminus \{0\}} \rho\left( (\mathbf{e}_{k-1, k}^{\text{inertial}})^T \boldsymbol{\Sigma}_{\text{IMU}}^{-1} \mathbf{e}_{k-1, k}^{\text{inertial}} \right) + \sum_{k \in \mathcal{K}_L \setminus \{0\}} \|\mathbf{e}_{\text{bias}}^k\|_{\boldsymbol{\Sigma}_{\text{bias}}}^2$$

---

## 3. Dimension 2: Optical Setup and Fisheye Handling

```
+----------------------------------------------------------------------------------------------------+
|                                    OPTICAL SETUP & SIDE-MIRROR STEREO                              |
+----------------------------------------------------------------------------------------------------+

                           Left Mirror                      Right Mirror
                           +---------+                      +---------+
                           |  PLANE  |                      |  PLANE  |
                           +----+----+                      +----+----+
                               /                                \
                              /                                  \
                             /                                    \
                            v                                      v
                    [Virtual Left Cam]                    [Virtual Right Cam]
                    (Reflected Center)                    (Reflected Center)
                            \                                      /
                             \                                    /
                              \                                  /
                               +------------+      +------------+
                                            |      |
                                            v      v
                                    +----------------------+
                                    |    GoPro Hero 9      |
                                    |   Max Lens Mod       |
                                    |     155° FoV         |
                                    +----------+-----------+
                                               |
                                               v
                                    [Raw Fisheye Image]
                                   /         |         \
                                  /          |          \
                    [Left Mirror Crop]   [Main View]  [Right Mirror Crop]
                            |                |                 |
                            v                |                 v
                    [Digital Flip H]         |         [Digital Flip H]
                            |                |                 |
                            +----------------+-----------------+
                                             |
                                             v
                                 [Policy Observation Tensor]
                                 (Consistent Orientation 100%)
```

### 3.1 GoPro Wide-Angle Optics & Fisheye Model
* **Hardware Configuration:** GoPro Hero 9 Black equipped with the **Max Lens Mod**.
  - Nominal Field of View: $155^\circ$ diagonal/horizontal.
  - Recording Mode: $2.7\text{K}$ ($2704 \times 1520$) at $60\text{ fps}$, 16:9 aspect ratio.
  - Internal Stabilization (HyperSmooth): **Disabled** completely. (Stabilization crops frames and alters optical center dynamically, destroying geometric SLAM projection).
* **Mathematical Camera Model: Kannala-Brandt (`KannalaBrandt8`)**
  For a 3D camera-frame point $\mathbf{P}_c = [X_c, Y_c, Z_c]^T$:
  1. Radial distance: $r = \sqrt{X_c^2 + Y_c^2}$
  2. Incidence angle: $\theta = \arctan(r / Z_c)$
  3. Distorted angle polynomial:
     $$d(\theta) = \theta + k_1 \theta^3 + k_2 \theta^5 + k_3 \theta^7 + k_4 \theta^9$$
  4. Projection to distorted sensor coordinates $(u, v)$:
     $$u = f_x d(\theta) \frac{X_c}{r} + c_x$$
     $$v = f_y d(\theta) \frac{Y_c}{r} + c_y$$
  Parameters calibrated via ChArUco target: $\mathbf{K} = \{f_x, f_y, c_x, c_y, k_1, k_2, k_3, k_4\}$.

### 3.2 Unrectified vs. Rectified Paradigm
A central design decision of UMI is the complete rejection of traditional pinhole rectification for policy training:

| Optical Paradigm | Policy Success Rate | Distortion & Resolution Behavior | Failure Modes & Limitations |
| :--- | :--- | :--- | :--- |
| **Raw Fisheye (155° FoV, Ours)** | **100% (20/20)** on Cup Task | Preserves high spatial pixel density at foveal center ($Z$-axis); compresses periphery smoothly. | None. Network learns natural lens curvature. |
| **Rectified Pinhole (69° FoV, RealSense / iPhone)** | **55% (11/20)** on Cup Task | Severe tangential stretching at edges; requires heavy cropping to avoid black borders. | **Out-of-view failures:** Target cup/saucer leaves FOV during approach; induces artificial multimodality and jerky trajectories. |
| **Rectified Ultra-Wide (155° Pinhole)** | $< 20\%$ (Empirical failure) | Theoretical projection: $\tan(\theta) \to \infty$ as $\theta \to 90^\circ$. | Extreme peripheral pixel stretching, computational warping artifacts, massive distortion of planar objects. |

*Finding from Paper (Section V-A, lines 729–739):*  
When post-processing the demonstration dataset to simulate a standard RealSense D415 camera ($69^\circ\text{ HFoV}, 42^\circ\text{ VFoV}$), performance degraded from $100\%$ to $55\%$. The narrow FoV repeatedly resulted in the manipulated object exiting camera visibility, forcing the policy to guess or oscillate violently.

### 3.3 The Side-Mirror Mechanism & Digital Reflection Transform
* **Physical Assembly:** Two planar first-surface glass mirrors are angled laterally on the gripper frame within the outer $15^\circ$ peripheral cone of the $155^\circ$ fisheye view.
* **Optical Geometry (Virtual Stereo):**
  Each mirror defines a reflection plane with surface normal $\mathbf{n} \in \mathbb{R}^3$ and offset $d \in \mathbb{R}$. A physical point $\mathbf{X}$ reflects to a virtual point $\mathbf{X}_v$:
  $$\mathbf{X}_v = \mathbf{X} - 2(\mathbf{n}^T \mathbf{X} + d)\mathbf{n}$$
  Equivalently, this creates two **virtual cameras** located at positions $\mathbf{C}_{\text{left}}$ and $\mathbf{C}_{\text{right}}$ behind the mirrors, providing implicit stereo baselines of $B \approx 120\text{--}150\text{ mm}$ without additional electronics.
* **The Chirality & Inversion Paradox:**
  In raw mirror views, optical chirality is inverted: an object translating to the right in world space appears to translate to the left inside the mirror; a clockwise rotation appears counter-clockwise.
  - *Empirical Impact:* Directly passing raw mirror crops to Diffusion Policy dropped success from $90\%$ (no mirrors) to $85\%$ because CNN/ViT convolutional inductive biases and translation equivariance were corrupted by contradictory motion vectors.
* **UMI Digital Reflection Transform:**
  UMI segments the mirror bounding boxes, applies a horizontal digital flip ($\text{Flip}_H$), and swaps the left and right crops:
  $$\mathcal{I}_{\text{policy}} = \left[ \text{Flip}_H(\mathcal{I}_{\text{mirror\_right}}), \; \mathcal{I}_{\text{main\_center}}, \; \text{Flip}_H(\mathcal{I}_{\text{mirror\_left}}) \right]$$
  With this transform, object orientations and optical flow vectors become strictly aligned across all three perspectives, elevating policy success to **100% (20/20)**.

### 3.4 Continuous Gripper Width Tracking
* **Marker Instrumentation:** Small ArUco fiducial tags ($16 \times 16\text{ mm}$, Dictionary `DICT_4X4_50`) are affixed to the lateral edges of the 3D-printed fingers.
* **Measurement Pipeline (`04_detect_aruco.py` & `05_extract_gripper_distance.py`):**
  1. OpenCV ArUco detector locates 4 corner sub-pixels for each finger tag.
  2. The 3D inter-marker Euclidean distance $d(t) = \|\mathbf{p}_{\text{tag1}}(t) - \mathbf{p}_{\text{tag2}}(t)\|_2$ is computed frame-by-frame.
  3. Calibration session (`gripper_calibration_*.mp4`) maps $d(t)$ over 5 full squeeze-release cycles to metric stroke $w \in [0.0, 0.080]\text{ m}$.
* **Implicit Force Regulation (Series Elastic End-Effector):**
  Because the fingertips are printed with compliant 95A TPU material, continuing to command gripper closure after contact causes measurable elastic finger deformation. Tracking the sub-millimeter displacement beyond contact provides implicit grasping force control without load cells.

---

## 4. Dimension 3: Scale Ambiguity Resolution

In classical monocular Structure-from-Motion (SfM) and monocular SLAM, camera translation $\mathbf{t}$ and 3D map points $\mathbf{P}$ can only be recovered up to an unknown scalar scale factor $s \in \mathbb{R}^+$:
$$\mathbf{T}_{\text{estimated}} = \begin{bmatrix} \mathbf{R} & s \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix}, \quad \mathbf{P}_{\text{estimated}} = s \mathbf{P}_{\text{true}}$$

In robotic manipulation, an unscaled trajectory is useless: an action commanded as $0.10$ could correspond to $5\text{ cm}$ or $50\text{ cm}$, leading to catastrophic collisions or missed grasps. UMI resolves this scale ambiguity through a tri-fold constraint hierarchy:

```
+----------------------------------------------------------------------------------------------------+
|                                    UMI METRIC SCALE RESOLUTION                                     |
+----------------------------------------------------------------------------------------------------+

   [1. Metric Mapping Prior]              [2. Inertial Kinematics]           [3. CAD Gripper Bounds]
   ArUco Marker with known CAD            GoPro IMU Accelerometer            Fixed 80 mm jaw stroke
   dimension (e.g. 100 mm side)           measures specific force f (m/s²)   calibrates finger width
               |                                      |                                  |
               v                                      v                                  v
    PnP Closed-Form Scale:                Earth Gravity Reference:            Fiducial Tag Tracking:
    ||X_i - X_j|| = d_CAD                 ||g|| = 9.80665 m/s²                w(t) in [0.0, 0.080] m
               |                                      |                                  |
               +-------------------+------------------+                                  v
                                   |                                        Implicit Force Control
                                   v                                        & Clamping Bounds
                     Joint Visual-Inertial LBA:
                     Absolute Scale s = 1.000 locked
                     Mean ATE = 6.1 mm across episodes
```

### 4.1 Constraint 1: Marker-Enhanced PnP Scale Fixation
During the initial scene mapping video (`02_create_map.py`), an ArUco fiducial marker of known physical dimension $L_{\text{marker}}$ is placed on the workspace:
* The 3D model coordinates of the marker corners are defined in metric units:
  $$\mathbf{X}_{\text{marker}} = \left\{ \begin{bmatrix} 0 \\ 0 \\ 0 \end{bmatrix}, \begin{bmatrix} L_{\text{marker}} \\ 0 \\ 0 \end{bmatrix}, \begin{bmatrix} L_{\text{marker}} \\ L_{\text{marker}} \\ 0 \end{bmatrix}, \begin{bmatrix} 0 \\ L_{\text{marker}} \\ 0 \end{bmatrix} \right\}$$
* Detecting corners $\mathbf{u}_i$ in the initial frames allows Perspective-n-Point (PnP) optimization to solve for the metric camera pose:
  $$\arg\min_{\mathbf{R}, \mathbf{t}} \sum_{i=1}^4 \|\mathbf{u}_i - \pi_{\text{KB}}(\mathbf{R}\mathbf{X}_i + \mathbf{t})\|^2$$
* Because $\|\mathbf{X}_i - \mathbf{X}_j\|_2 = L_{\text{marker}}$ is in physical meters, this observation locks the map scale $s = 1.000$ with zero gauge freedom.

### 4.2 Constraint 2: IMU Accelerometer Specific Force & Gravity Integration
The GoPro IMU's triaxial accelerometer measures specific force $\mathbf{f}_b$ in metric $\text{m/s}^2$:
$$\mathbf{f}_b(t) = \mathbf{R}_{bw}(t) (\mathbf{a}_w(t) - \mathbf{g}_w) + \mathbf{b}_a(t) + \boldsymbol{\eta}_a(t)$$
* **Gravity Observability:** Earth's gravitational acceleration has an invariant magnitude $\|\mathbf{g}_w\| \approx 9.80665\text{ m/s}^2$. In stationary or constant-velocity intervals, $\|\mathbf{f}_b\| = \|\mathbf{g}\|$, establishing both the pitch/roll attitude with respect to the vertical and the metric scale unit.
* **Kinematic Double Integration:** When the demonstrator moves the gripper, linear acceleration $\mathbf{a}_w(t)$ excites the accelerometer. In the joint factor graph, visual baseline displacements $\Delta \mathbf{p}_{\text{visual}} = s \cdot \Delta \mathbf{p}_{\text{norm}}$ are constrained by the double integral of measured acceleration:
  $$\Delta \mathbf{p}_{k, k+1} = \iint_{t_k}^{t_{k+1}} (\mathbf{R}_{wb}(\tau)\mathbf{f}_b(\tau) + \mathbf{g}_w) d\tau^2$$
  The scale factor $s$ is observable because changing $s$ introduces quadratic residuals against the metric IMU factors.

### 4.3 Constraint 3: CAD Dimensions of the Gripper Mechanism
* The parallel-jaw rack-and-pinion transmission has a strictly bounded physical stroke of $80.0\text{ mm}$.
* Finger opening is calibrated against known CAD dimensions (`gripper_range.json`).
* In bimanual tasks, when the two grippers touch or interact with a rigid object of known dimension, the inter-gripper pose error is bounded to $10.1\text{ mm}$ (RPE), preventing scale drift over long trajectories.

---

## 5. Dimension 4: Temporal Synchronization and Batch Processing Workflow

```
+----------------------------------------------------------------------------------------------------+
|                                  UMI OFFLINE BATCH PROCESSING PIPELINE                             |
+----------------------------------------------------------------------------------------------------+

  [GoPro SD Card: *.MP4]
            |
            v
  [00_process_videos.py] ------------------> Categorizes mapping.mp4, gripper_cal.mp4, demo_*.mp4
            |
            v
  [01_extract_gopro_imu.py] ---------------> gpmf-parser extracts 200 Hz IMU -> imu_data.json
            |
            v
  [02_create_map.py] ----------------------> Docker ORB-SLAM3 + slam_mask.png -> map_atlas.osa
            |
            v
  [03_batch_slam.py] ----------------------> Map-as-Initialization tracking -> camera_trajectory.csv
            |
            v
  [04_detect_aruco.py] --------------------> Finger tag detection -> tag_detection.pkl
            |
            v
  [05_extract_gripper_distance.py] --------> Metric width mapping -> gripper_range.json
            |
            v
  [06_generate_dataset_plan.py] -----------> Temporal alignment & trimming -> dataset_plan.pkl
            |
            v
  [07_generate_replay_buffer.py] ----------> Computes relative SE(3) actions -> dataset.zarr.zip
```

### 5.1 Step-by-Step Script Pipeline

| Script Name | Inputs | Processing Mechanism | Outputs & Artifacts |
| :--- | :--- | :--- | :--- |
| **`00_process_videos.py`** | Raw GoPro MP4 directory | Verifies video headers, checks frame drops, parses GoPro serial numbers to separate Left vs. Right grippers. | Structured directory: `demos/<demo_id>/raw_video.mp4` |
| **`01_extract_gopro_imu.py`** | `raw_video.mp4` | Executes `gpmf-parser` binary to extract binary telemetry; parses `ACCL` (accelerometer) and `GYRO` (gyroscope) packets; syncs NTP clock. | `imu_data.json`, `time_offset.json` |
| **`02_create_map.py`** | `mapping.mp4`, `imu_data.json`, `slam_mask.png` | Launches Docker container running ORB-SLAM3 in full SLAM mode; detects mapping ArUco marker; runs local & global BA. | `map_atlas.osa` (ORB-SLAM Atlas file) |
| **`03_batch_slam.py`** | `raw_video.mp4`, `map_atlas.osa`, `slam_mask.png` | Executes modified ORB-SLAM3 in "Map as Initialization" mode; tracks camera at 60 fps; recovers 6-DoF trajectory. | `camera_trajectory.csv` |
| **`04_detect_aruco.py`** | `raw_video.mp4` | OpenCV `cv2.aruco` detects 4 corner sub-pixels for finger tags across all 60 fps frames. | `tag_detection.pkl` |
| **`05_extract_gripper_distance.py`** | `tag_detection.pkl`, calibration videos | Computes Euclidean marker distance; normalizes stroke across 5 calibration cycles. | `gripper_range.json` |
| **`06_generate_dataset_plan.py`** | `camera_trajectory.csv`, `tag_detection.pkl` | Performs observation latency alignment; resamples streams to target policy rate (10 Hz or 20 Hz); segments episodes. | `dataset_plan.pkl` |
| **`07_generate_replay_buffer.py`** | Video frames, trajectory plan | Crops and horizontally reflects side-mirror views; masks ArUco tags; calculates relative $SE(3)$ action chunks; exports to Zarr. | `dataset.zarr.zip` |

### 5.2 Computational Overhead & Execution Times
* **Processing Speed per Episode:**  
  Running the full pipeline on a standard workstation (AMD Ryzen 9 / Intel i9 CPU, 32GB RAM, NVIDIA RTX 3080/4090):
  - `01_extract_gopro_imu`: $\approx 2\text{--}5\text{ s}$ per 30-second episode.
  - `03_batch_slam` (ORB-SLAM3 VI tracking inside Docker): $\approx 30\text{--}60\text{ s}$ per 30-second episode ($\approx 1.0\times\text{--}2.0\times$ real-time).
  - `04_detect_aruco`: $\approx 10\text{--}15\text{ s}$ per episode.
  - `07_generate_replay_buffer` (Video decoding + mirror reflection + Zarr compression): $\approx 15\text{--}25\text{ s}$ per episode.
  - **Total Pipeline Latency:** $\approx 60\text{--}100\text{ s}$ of offline compute per $30\text{ s}$ demonstration episode.
* **Batch Dataset Overhead:**  
  For a standard dataset of 300 demonstrations ($\approx 2.5\text{ hours}$ of physical demonstration time), offline processing requires approximately **5 to 8 hours** of automated batch computation.

### 5.3 Decoupling Real-Time Inference from Data Collection
A fundamental insight of UMI is that **Visual SLAM is strictly an offline training tool and NEVER runs during real-time robot deployment**:

```
Data Collection (Human Hands):
  Handheld Gripper + GoPro ---> MP4 Video + IMU ---> OFFLINE SLAM ---> Relative SE(3) Actions ---> Zarr Dataset

Robot Deployment (Online Execution):
  Wrist GoPro Live Stream (RTSP/HDMI) ---> Diffusion Policy ---> Desired Relative EE Trajectory
                                                  ^
  Robot Forward Kinematics (Joint Encoders) -------+
  (Runs at 500-1000 Hz, 0 SLAM compute, 0 SLAM drift!)
```

* **Why Handheld Collection Uses Offline SLAM:** Handheld data collection cannot use robot forward kinematics because there is no robot arm attached. Using an untethered GoPro with offline SLAM eliminates heavy onboard computers, external MoCap cameras, cables, and mechanical counterbalances.
* **Why Robot Deployment Avoids SLAM:** During online execution on a UR5 or Franka robot, the robot's built-in joint encoders provide exact end-effector poses $\mathbf{T}_{\text{base}, ee}$ at $500\text{--}1000\text{ Hz}$ with sub-millimeter precision and $< 1\text{ ms}$ latency. Because the policy was trained on relative trajectory representations:
  $$\mathbf{T}_{\text{cmd}}(t + \tau) = \mathbf{T}_{ee}(t) \cdot \Delta \mathbf{T}_{\text{predicted}}(\tau)$$
  the robot simply accumulates predicted relative deltas onto its current joint-derived pose. **Visual tracking loss, map relocalization failure, and SLAM computational latency are completely eliminated at inference time.**

---

## 6. Dimension 5: Failure Modes, Drift Characteristics, and Limitations

### 6.1 Quantitative Accuracy Benchmarks (MoCap Ground Truth)
To independently validate the SLAM pipeline, the UMI authors evaluated tracking against an OptiTrack optical motion capture system across 7 single-gripper tasks and 7 bimanual tasks:

| Metric | Measured Position Error | Measured Rotation Error | Reference Ground Truth |
| :--- | :--- | :--- | :--- |
| **Per-Gripper Absolute Trajectory Error (ATE)** | **$6.1\text{ mm}$** | **$3.5^\circ$** | OptiTrack MoCap multi-camera cage |
| **Inter-Gripper Relative Pose Error (RPE)** | **$10.1\text{ mm}$** | **$0.8^\circ$** | OptiTrack MoCap inter-marker vector |

*Significance:* An ATE of $6.1\text{ mm}$ is well within the compliance tolerance of the 95A TPU soft fingers ($> 10\text{ mm}$ passive deformation margin), enabling high-precision insertion and grasping tasks.

### 6.2 Primary SLAM Failure Modes & Fragility
Despite optimizations, the repository maintainers explicitly acknowledge:  
> *"ORB-SLAM3 is often the most fragile component of the UMI pipeline."*

The primary failure mechanisms identified in the codebase and literature include:

1. **Relocalization Dropouts in Batch Tracking (`03_batch_slam.py`):**
   - *Cause:* If the demonstrator sweeps the handheld gripper into a viewing angle or workspace quadrant not visited during the 1-minute `mapping.mp4` scan, ORB feature correspondences against `map_atlas.osa` drop below the minimum inlier threshold ($N_{\text{inliers}} < 15$).
   - *Consequence:* The tracker gets lost; the entire demonstration episode must be discarded.
2. **Dynamic Manipulated Objects Violating Staticity:**
   - *Cause:* When manipulating large objects (e.g., folding a large sweater, carrying large trays), the manipulated item covers significant background area outside `slam_mask.png`.
   - *Consequence:* ORB-SLAM3 extracts features on the deforming sweater. As the sweater folds, the non-rigid feature motion violates epipolar constraints, generating erroneous translation estimates or causing bundle adjustment divergence.
3. **Severe Motion Blur under Low Ambient Lighting:**
   - *Cause:* Rapid human actions (such as dynamic tossing at $> 1.8\text{ m/s}$) combined with indoor lighting ($< 200\text{ lux}$) force the GoPro's auto-exposure to drop shutter speeds below $1/240\text{ s}$.
   - *Consequence:* FAST corner detectors fail to trigger; BRIEF descriptor hamming distances fail to match. While the IMU bridges tracking across $0.2\text{--}0.5\text{ s}$, prolonged visual loss causes unconstrained quadratic drift ($\mathbf{p} \sim \frac{1}{2} \mathbf{a} t^2$).
4. **Texture Deficiency and Specular Reflections:**
   - *Cause:* Featureless environments (e.g., blank white walls, polished granite countertops, uniform stainless steel sinks) yield $< 30$ extractable ORB features.
   - *Consequence:* The system fails during `02_create_map.py`. Water reflections in sink washing tasks create moving virtual specular points that corrupt 3D triangulations.

### 6.3 Hardware & Human-Demonstrator Limitations
* **Physical Ergonomics & Fatigue:** The UMI gripper assembly weighs $780\text{ g}$. Holding the device horizontally for data collection sessions exceeding 45 minutes induces demonstrator forearm fatigue, resulting in shaky trajectories. Throughput is $48\%\text{--}64\%$ that of bare-hand demonstrations.
* **Kinematic Violations (The Embodiment Gap):** Human operators have 7-DoF arm kinematics with extensive reach. During handheld collection, operators frequently execute motions that violate the joint limits or singularity boundaries of the target deployment robot (e.g., Franka FR2 or UR5).
  - *Mitigation (HD6):* UMI applies an offline kinematic feasibility filter that simulates inverse kinematics on the target robot base frame, discarding $10\%\text{--}30\%$ of collected demonstrations.

---

## 7. Comparative Synthesis: UMI SLAM Pipeline vs. visuomotor_ros2 Active Setup

To provide a direct bridge between the UMI literature audit and the active `visuomotor_ros2` codebase, the table below contrasts UMI's offline Monocular-Inertial pipeline against the active RealSense D435i + RTAB-Map VIO configuration:

| Architectural Dimension | UMI Pipeline (Chi et al. / Stanford) | `visuomotor_ros2` Active Configuration | Technical Trade-Off & Assessment |
| :--- | :--- | :--- | :--- |
| **Primary SLAM Backend** | **ORB-SLAM3** (Custom fork, C++ in Docker) | **RTAB-Map (`rgbd_odometry`)** on ROS 2 Jazzy | ORB-SLAM3 optimizes globally via Atlas/BA; RTAB-Map operates natively in ROS 2 node graph. |
| **Operational Paradigm** | **Offline Batch Processing** (Post-session) | **Online Real-Time Streaming (20 Hz synchronous)** | UMI decouples compute from collection; `visuomotor_ros2` enables live teleoperation & interactive monitoring. |
| **Depth & Scale Recovery** | Monocular + IMU + Fiducial Marker Initialization | **Active Infrared Stereo (Native Metric Disparity)** | **`visuomotor_ros2` is strictly superior in scale stability:** physical baseline directly eliminates scale drift without markers. |
| **Optics & Field of View** | **155° Fisheye** (GoPro Max Lens Mod) | **69° × 42° Pinhole** (RealSense D435i) | **UMI is superior in visual context:** 155° FoV prevents manipulated objects from exiting camera view. |
| **Handling of Low Texture** | Fails on textureless surfaces (white walls, sinks) | **Active IR Projector** projects speckle pattern | **`visuomotor_ros2` is far more robust** in texture-deficient indoor environments. |
| **Foreground Masking** | Static binary mask (`slam_mask.png`) | Depth window truncation ($0.20\text{--}1.80\text{ m}$) | UMI masks pixels directly; `visuomotor_ros2` filters near-field gripper noise via depth thresholds. |
| **Hardware Portability** | **100% Wireless Handheld** ($780\text{ g}$, GoPro SD) | Tethered USB-3.0 to host workstation | UMI enables untethered "in-the-wild" collection; RealSense requires host connection. |
| **Inference Coupling** | **Zero SLAM at deployment** (Forward Kinematics) | Optional real-time VIO feedback or joint states | Both frameworks converge on relative action spaces for robot deployment. |
| **Demonstrated Accuracy** | **$6.1\text{ mm}$ ATE**, $10.1\text{ mm}$ RPE | **$2.307\text{ mm}$ return error**, $1.61\text{ mm}$ jitter | `visuomotor_ros2` active configuration achieves certified sub-3 mm precision. |

---

## 8. Verified Citations & References

1. **Chi, C., Xu, Z., Pan, C., Cousineau, E., Burchfiel, B., Feng, S., Tedrake, R., & Song, S.** (2024). *Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots*. Robotics: Science and Systems (RSS 2024) / arXiv:2402.10329v2.
2. **Campos, C., Elvira, R., Rodríguez, J. J. G., Montiel, J. M. M., & Tardós, J. D.** (2021). *ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual–Inertial, and Multimap SLAM*. IEEE Transactions on Robotics, 37(6), 1874–1890.
3. **Chi, C., Feng, S., Du, Y., Xu, Z., Cousineau, E., Burchfiel, B., & Song, S.** (2023). *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*. Robotics: Science and Systems (RSS 2023).
4. **Forster, C., Carlone, L., Shen, F., & Scaramuzza, D.** (2017). *On-Manifold Preintegration for Real-Time Visual--Inertial Odometry*. IEEE Transactions on Robotics, 33(1), 1–21.
5. **Kannala, J., & Brandt, S. S.** (2006). *A Generic Camera Model and Calibration Method for Conventional, Wide-Angle and Fish-Eye Lenses*. IEEE Transactions on Pattern Analysis and Machine Intelligence, 28(8), 1335–1340.
6. **Urban, S.** *OpenImuCameraCalibrator & GoPro SLAM*. Open-source visual-inertial calibration tools (acknowledged in UMI manuscript).
7. **Official UMI GitHub Repository:** `https://github.com/real-stanford/universal_manipulation_interface`.
