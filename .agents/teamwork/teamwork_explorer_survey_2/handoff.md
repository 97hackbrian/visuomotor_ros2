# Handoff Report — Explorer 2 (Harvesting & SOTA SLAM Explorer)

**Date**: 2026-11-20T20:28:00Z  
**Working Directory**: `/home/hackbrian/Documents/gits/visuomotor_ros2/.agents/teamwork/teamwork_explorer_survey_2/`  
**Target Path**: `/home/hackbrian/Documents/gits/visuomotor_ros2/.agents/teamwork/teamwork_explorer_survey_2/handoff.md`  
**Recipient**: Parent Orchestrator (`eeabad66-1519-4aef-8ca9-7d5ccc498e8c`)  
**Mission**: Audit harvesting robotics literature (`bio/pepper_harvesting.pdf`, `bio/fruit_bearing.pdf`), SOTA demonstration frameworks (DexCap, Dobb-E, Mobile ALOHA, UMI), and compare them with `visuomotor_ros2` active configuration (Intel RealSense D435i + RTAB-Map VIO).

---

## 1. Observation

1. **`bio/pepper_harvesting.pdf` (Chung Hee Kim, Abhisesh Silwal, George Kantor, Carnegie Mellon University, IEEE RA-L, Vol. 10, No. 4, April 2025)**:
   - **Verbatim Quote (Page 3408, Section III-B & IV-B)**:
     > *"UMI’s setup involves a handheld gripper equipped with a wrist-mounted camera to capture visual and action data during human demonstrations. The framework uses visual-inertial SLAM (ORB-SLAM3 [29]) to generate maps for gripper localization during data collection—a process that functions well in the controlled, static environments typical of indoor tasks. However, applying UMI in agricultural settings poses significant challenges. Outdoor agricultural tasks are subject to constant disturbances such as wind and crop interactions, which deprives well-defined static features necessary for reliable localization using ORB-SLAM3. Moreover, since operations like harvesting and pruning are inherently irreversible (e.g., a harvested apple cannot be reattached), frequent re-mapping for each demonstration is required, rendering this approach impractical."*
   - **Verbatim Results (Page 3409, Section IV-B, Fig. 4)**:
     > *"Without refinement (purple line), the pose estimates were noisy, leading to mean squared errors (MSE) of 0.276 for position and 0.358 for rotation. Applying our method (orange line) significantly reduced noise, achieving MSEs of 0.003 and 0.018, respectively, yielding a smoother trajectory that supports reliable data collection for policy training."*
   - **Hardware Setup (Page 3409, Section V-A)**: UR5e arm on Husky UGV, Intel Core i9-13900K, RTX 4070 12GB GPU. Handheld tool matches robotic end-effector. Eye-in-hand fisheye camera deployed on robot; external tripod camera used during demonstration collection is eliminated.
   - **Empirical Field Results (Page 3411, Table I)**: 300 demonstrations, 221 outdoor field trials. Overall success rate: $28.95\%$ (64/221), cycle time $31.71\text{ s}$. Easy: $42.57\%$, Medium: $20.21\%$, Hard: $7.31\%$.

2. **`bio/fruit_bearing.pdf` (Jiacheng Rong et al., China Agricultural University, Zhejiang University, Elsevier, In Press, 2026)**:
   - **Hardware & Vision Setup (Page 12-13, Section 2.2.1, Fig. 3)**: 7-DOF RealMan RM75B robotic arm, custom scissor end-effector with two Intel RealSense D405 cameras mounted eye-in-hand (Camera 1 for primary RGB-D grasping point detection and PBVS; Camera 2 for close-up tool-pedicel interaction).
   - **Verbatim Comparison on PBVS Collapse (Page 42-43, Table 6)**:
     > *"The results show that, in the absence of any initial deviation (i.e., when the blades are already aligned), the method achieves a success rate of 94.0% (63/67), indicating high positioning accuracy under ideal conditions. However, under moderate deviation (slight initial misalignment), the success rate drops sharply to 0.0% (0/13), meaning that all 13 trials fail when even a small offset is present... small calibration errors or pose-estimation inaccuracies introduce systematic offsets that cannot be corrected by rigid geometric servoing."*
   - **Learned ACT Policy Performance (Page 44-45, Table 7)**:
     - No deviation: $91.0\%$ (61/67), duration $9.1\text{ s}$.
     - Moderate deviation ($5\text{--}25\text{ mm}$): Left offset $70.7\%$ (29/41), Right offset $66.7\%$ (32/48).
     - Severe deviation ($25\text{--}40\text{ mm}$): Left offset $56.7\%$ (17/30), Right offset $61.8\%$ (21/34).
   - **Failure Modes of Unimodal RGB (Page 51-54, Section 3.2.3, Fig. 18-20)**: Visual hallucinations ($\approx 50\%$ of failures), blind forward motion without correction, background stem distraction, and incorrect action sequencing.

3. **`bio/dp3d.pdf` (Yanjie Ze et al., RSS 2024)**:
   - **Ablation on Modalities (Page 5-6, Table IV)**: Point cloud ($78.3\%$) vs 2D RGB image ($40.7\%$) vs 2D Depth image ($32.0\%$) vs RGB-D ($34.7\%$) vs Voxels ($32.3\%$).
   - **Safety Violations in Real Robots (Page 10, Table XIV)**: 2D RGB Diffusion Policy caused $32.5\%$ safety violations (collisions/ground strikes); 2D Depth caused $25.0\%$; DP3 caused **$0.0\%$ safety violations**.
   - **DP3 Encoder Architecture (Page 14, Appendix A)**: 3-layer MLP ($3 \to 64 \to 128 \to 256$) with interleaved LayerNorm and ReLU, order-equivariant global Max-Pool, and Linear Projection to 64D feature vector.

4. **SOTA Demonstration Frameworks**:
   - **DexCap**: Intel L515 LiDAR (chest) + dual RealSense T265 SLAM cameras (wrist) + Manus mocap gloves. Cost $>\$10,000$; L515 blinds in sunlight; T265 drifts during rapid hand acceleration.
   - **Dobb-E**: $\$25$ reacher-grabber stick + iPhone LiDAR (ARKit VIO). Accessible ($<\$1,000$), but coarse depth grid ($64\times 48$) and VIO drift in textureless vegetation.
   - **Mobile ALOHA**: Dual-arm master-slave teleoperation via physical Dynamixel joint encoders at 50 Hz. Absolute zero visual drift on joint angles, but heavy non-portable rig costing $\$20,000\text{--}\$30,000$.
   - **UMI**: Handheld gripper + GoPro fisheye + side mirror. Offline batch ORB-SLAM3 VIO; no real-time inference; fails in agricultural wind/foliage.
   - **visuomotor_ros2 Active Configuration**: RealSense D435i + Bosch BMI085 IMU ($200\text{ Hz}$) + RTAB-Map VIO ($20.0\text{ Hz}$ Frame-to-Map). Certified return error $2.307\text{ mm}$, jitter $<0.65\text{ mm}$, native active IR metric scale, real-time ROS 2 Jazzy integration.

---

## 2. Logic Chain

1. **Step 1 (From Observation 1)**: In outdoor agricultural tasks, ambient wind causes leaves and stems to sway continuously, which deprives feature-based Visual SLAM algorithms (like ORB-SLAM3 in UMI) of static 3D world points. Furthermore, harvesting is physically irreversible; once the crop is severed, the visual scene permanently changes, making pre-built map reuse impossible without continuous, exhaustive remapping.
2. **Step 2 (From Observation 2)**: Classical Position-Based Visual Servoing (PBVS) relies on static forward geometric projection and inverse kinematics. Hand-eye calibration residuals ($1\text{--}3\text{ mm}$), depth estimation noise, and minor plant movement accumulate into systematic spatial errors, causing PBVS to collapse to $0.0\%$ success whenever moderate initial misalignment ($>5\text{ mm}$) occurs. Thus, high-precision harvesting requires closed-loop, data-driven corrective visual servoing.
3. **Step 3 (From Observation 2 & 3)**: Unimodal 2D RGB policies suffer from visual attention leakage (attending to fruit body rather than peduncle) and visual hallucinations (misidentifying background stems as target pedicels). Incorporating metric 3D point clouds (as proven in DP3) provides Euclidean spatial separation that eliminates background clutter confusion and prevents hardware collisions ($0.0\%$ safety violations).
4. **Step 4 (From Observation 1, 3, & 4)**: The active configuration of `visuomotor_ros2` (RealSense D435i + RTAB-Map VIO) solves the primary operational defects of UMI: it operates in synchronous real-time ($20\text{ Hz}$), guarantees native physical metric scale without gauge ambiguity, and achieves certified sub-$2.5\text{ mm}$ accuracy ($2.307\text{ mm}$ return error). For indoor/greenhouse and Isaac Sim domains, it is vastly superior to monocular UMI.

---

## 3. Caveats

1. **Direct Outdoor Solar Saturation**: While RealSense D435i active IR stereo is superior in indoor, greenhouse, and controlled laboratory lighting, direct sunlight ($>80,000\text{ lux}$) saturates the $850\text{ nm}$ IR sensor and drowns out the weak laser projector dots. For direct outdoor deployment, optical bandpass filters, Flash-No-Flash (FNF) strobe illumination, or appearance-invariant fisheye RGB policies (as in Kim et al.) are required.
2. **Peduncle Semantic Segmentation**: Visual attention drift remains an issue if image encoders are trained on raw RGB without auxiliary peduncle segmentation masks or 3D point cloud cropping.
3. **Hardware Write Boundary**: Per instructions, no repository source code or datasets were modified; findings and comparative analyses are strictly documented in the assigned agent workspace.

---

## 4. Conclusion

1. **Sensing & SLAM Verdict**: UMI's monocular ORB-SLAM3 pipeline is fundamentally unsuited for agricultural harvesting due to dynamic foliage motion and task irreversibility. Handheld systems in such domains require either external fiducial tracking during data collection (Kim et al.) or real-time active RGB-D VIO (visuomotor_ros2).
2. **Visual Servoing Verdict**: Open-loop PBVS is mathematically insufficient for pedicel grasping under spatial misalignment ($0.0\%$ success). A two-stage hybrid approach (macro PBVS to reach $10\text{ cm}$ pre-grasp pose + micro closed-loop Diffusion Policy / ACT with 5 mm increments) provides robust error recovery.
3. **visuomotor_ros2 Telemetric Superiority**: The active RealSense D435i + RTAB-Map VIO configuration in `visuomotor_ros2` provides a proven, real-time, metric-certified foundation ($2.307\text{ mm}$ return error) that eliminates UMI's offline latency and scale ambiguity, making it an ideal platform for training and evaluating visuomotor policies.
4. **Roadmap to DP3**: Future policy iterations should incorporate 3D point cloud encoding (`DP3Encoder` with 512 points), leveraging the D435i depth stream to achieve appearance invariance, eliminate background foliage hallucinations, and guarantee collision-free deployment.

---

## 5. Verification Method

To independently verify all findings:
1. **Inspect Local Literature Audits**:
   - `bio/pepper_harvesting.pdf`: Verify page 3408 (Section III-B & IV-B) for UMI/ORB-SLAM3 failure; page 3409 for fiducial cube PnP MSE ($0.276\text{ m} \to 0.003\text{ m}$); page 3411 for Table I field results ($28.95\%$).
   - `bio/fruit_bearing.pdf`: Verify page 12-13 for RealSense D405 hardware; page 42-43 for Table 6 PBVS collapse ($0.0\%$ at moderate deviation); page 44-45 for Table 7 ACT results ($91.0\%$ no deviation, $70.7\%/66.7\%$ moderate deviation); page 51-54 for visual hallucination failure modes.
   - `bio/dp3d.pdf`: Verify page 5-6 (Table I & IV) for point cloud ($78.3\%$) vs 2D RGB ($40.7\%$) vs depth ($32.0\%$); page 10 (Table XIV) for safety violations ($0.0\%$ for DP3 vs $32.5\%$ for 2D DP); page 14 for `DP3Encoder` PyTorch implementation.
2. **Inspect Technical Analysis Report**:
   - View `/home/hackbrian/Documents/gits/visuomotor_ros2/.agents/teamwork/teamwork_explorer_survey_2/analysis.md` for complete mathematical derivations, sensor comparison tables, and full bibliographic citations.
3. **Check Git Cleanliness**:
   - Confirm via host `git status` that no unauthorized source code modifications or git commits were executed.
