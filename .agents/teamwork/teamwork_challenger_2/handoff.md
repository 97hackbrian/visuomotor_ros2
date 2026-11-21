# Adversarial Verification Report & Handoff — Challenger 2
**Role:** Adversarial Policy & Harvesting Challenger (Specialist & Critic)  
**Target Document:** `/home/hackbrian/Documents/gits/visuomotor_ros2/docs/research/umi_vs_rtabmap_vio_comparison.md`  
**Primary Sources Audited:**
- `bio/fruit_bearing.pdf` (Rong et al., Elsevier / CAU 2026)
- `bio/pepper_harvesting.pdf` (Kim et al., IEEE RA-L / CMU 2025)
- `bio/dp3d.pdf` (Ze et al., RSS 2024)
- `bio/umi.pdf` (Chi et al., RSS 2024)

---

## 1. Observation

Direct, empirical observations obtained via forensic extraction (`pdftotext`, Python AST and string audit):

### Obs 1.1: Citation and Framing of Rong et al. (`bio/fruit_bearing.pdf`)
* **Target Document Claims:**
  - Line 45: `Follaje dinámico + Viento + Tarea Irreversible (Corte) ──> Colapso total de ORB-SLAM3 y PBVS (0.0% éxito)`
  - Lines 368–379: Table contrasting "Éxito PBVS Rígido" vs "Éxito ACT / Diffusion":
    ```markdown
    Desviación Inicial de Alineación     Éxito PBVS Rígido       Éxito ACT / Diffusion
    -----------------------------------------------------------------------------
    Sin desviación (0 mm)                  94.0% (63/67)            91.0% (61/67)
    Desviación moderada (5 - 25 mm)         0.0% (0/13)    Izq:     70.7% (29/41)
                                                           Der:     66.7% (32/48)
    Desviación severa (25 - 40 mm)         No Factible     Izq:     56.7% (17/30)
                                                           Der:     61.8% (21/34)
    ```
* **Verbatim Text from `bio/fruit_bearing.pdf`:**
  - Section 3.2.2: *"We next examine the performance of a baseline position-based visual servo approach (assuming ideal inverse kinematics), as summarized in Table 6. This method directly drives the end-effector toward the estimated pedicel pose without incorporating learned feedback. The results show that, in the absence of any initial deviation (i.e., when the blades are already aligned), the method achieves a success rate of 94.0% (63/67)... However, under moderate deviation (slight initial misalignment), the success rate drops sharply to 0.0% (0/13), meaning that all 13 trials fail when even a small offset is present."*
  - Section 3.2.1: Deviation definitions are purely qualitative:
    * *"No deviation: The scissor blades are initially well aligned with the pedicel."*
    * *"Moderate Deviation: The scissor blades are slightly misaligned, requiring minor corrective adjustments to achieve alignment."*
    * *"Severe Deviation: The scissor blades are significantly misaligned, requiring substantial corrective movements before a successful grasp can be achieved."*
  - Quantitative intervals `"5 - 25 mm"` and `"25 - 40 mm"` do NOT appear anywhere in Rong et al. The only mention of $25\text{ mm}$ is in the narrative text of Fig. 17(b) describing an individual severe trial: *"a rightward adjustment of approximately 25 mm"*.
  - Section 2.3: Rong et al. tested **Modified ACT (Action Chunking with Transformers)**, NOT Diffusion Policy. The word "diffusion" appears only once in the entire paper (Section 1, related work citation).
  - Environment: Rong et al. conducted their tests inside a controlled greenhouse (`Hong-Fu International Agricultural Base, Beijing, China`). The word "wind" appears 0 times in `bio/fruit_bearing.pdf`.

### Obs 1.2: Citation and Framing of Kim et al. (`bio/pepper_harvesting.pdf`)
* **Target Document Claims:**
  - Line 45: Attributes "Colapso total de ORB-SLAM3... (0.0% éxito)" to field conditions.
  - Lines 347–358: Claims Kim et al. discovered that ORB-SLAM3 is "completamente inaplicable" and reports: *"PnP sin refinar: MSE_pos = 0.276 m ... Pipeline con SSIM: MSE_pos = 0.003 m ... Reducción del 92x en el error posicional, alcanzando la fidelidad milimétrica necesaria para entrenar Diffusion Policy"*.
* **Verbatim Text from `bio/pepper_harvesting.pdf`:**
  - Section III-B & IV-B: Kim et al. explain qualitatively why UMI's ORB-SLAM3 is impractical for field data collection: *"Outdoor agricultural tasks are subject to constant disturbances such as wind and crop interactions, which deprives well-defined static features necessary for reliable localization using ORB-SLAM3. Moreover, since operations like harvesting and pruning are inherently irreversible... frequent re-mapping for each demonstration is required, rendering this approach impractical."*
  - Section IV-B (Ground Truth Reference):
    *"We quantitatively evaluated our method using the ORB-SLAM3 trajectory as a proxy baseline (see Fig. 4), illustrating the reliability of our localization method. The trajectory was derived in a static environment to avoid the known issue of ORB-SLAM3 breaking in dynamic scenes. Without refinement (purple line), the pose estimates were noisy, leading to mean squared errors (MSE) of 0.276 for position and 0.358 for rotation. Applying our method (orange line) significantly reduced noise, achieving MSEs of 0.003 and 0.018, respectively..."*
  - Kim et al. **did not run ORB-SLAM3 in the field to obtain a 0.0% failure rate**. They trusted ORB-SLAM3 in a static environment as their **ground truth reference** to validate the fiducial cube tracker!
  - Section VIII (Deployment Outcome): Kim et al. reported a real-world field harvesting success rate of **28.95%** with a cycle time of $31.71\text{ s}$ using their trained policy.

### Obs 1.3: Presentation of DP3 (Ze et al., `bio/dp3d.pdf`)
* **Target Document Claims:**
  - Lines 401–414: Cites Table IV (DP3 78.3% vs RGB 40.7%) and Table XIV (Safety Violations: DP3 0.0% vs RGB 32.5%).
  - Line 420: *"el proceso de difusión inversa DDIM nunca genera trayectorias que atraviesen superficies físicas (0.0% colisiones frente al 32.5% en modelos 2D RGB)."*
  - Lines 615–620: *"Aprovechamiento directo del hardware activo RealSense D435i ... Muestreo de N = 512 / 1024 puntos vía Farthest Point Sampling (FPS) ... Impacto Previsto: +24.2% incremento en éxito de agarre y 0.0% colisiones contra el banco físico"*.
* **Verbatim Text from `bio/dp3d.pdf`:**
  - Section IV-B: Real-world perception was captured using an **Intel RealSense L515 solid-state LiDAR camera**, NOT an active IR stereo camera like the D435i.
  - Section IV-B: Visual depth inputs were downscaled to **$84 \times 84$** before unprojecting to 3D point clouds.
  - Table XVI: Standard DP3 achieves an inference speed of only **12.7 FPS** (below 20 Hz) on a high-end desktop GPU. Reaching 25.3 FPS required simplifying the network architecture into **Simple DP3**.
  - Section V-D: *"Interestingly and surprisingly, we find that DP3 rarely violates the safety... It is important to note that our assessment of safety is primarily qualitative. We intend to explore a more theoretical understanding of this observation in our future work."*
  - Table I: The 24.2% value is the **relative improvement** across 72 simulated tasks ($59.8\% \to 74.4\%$), NOT an absolute +24.2 percentage point increase in grasp success rate.

### Obs 1.4: Metric Comparison in Final Verdict
* **Target Document Claims:**
  - Lines 53–56 & Section 7.1: *"logrando una precisión métrica de retorno certificada de $2.307\text{ mm}$ (frente a los $6.1\text{ mm}$ de ATE en UMI)... La configuración activa de visuomotor_ros2 es operativamente superior al pipeline de UMI..."*
  - Table Row 6: Directly contrasts `ATE: 6.1 mm` against `Retorno: 2.307 mm`.
* **Metrological Facts:**
  - UMI's $6.1\text{ mm}$ is **Absolute Trajectory Error (ATE)**: the Root Mean Square Error (RMSE) against ground truth poses tracked continuously by an external 12-camera **OptiTrack Motion Capture** system across the full trajectory.
  - `visuomotor_ros2`'s $2.307\text{ mm}$ is **Return Error / Endpoint Drift** ($\|\mathbf{p}(T) - \mathbf{p}(0)\|$): measured only at the starting cradle where RTAB-Map matches the starting 3D keypoints from its local map cache (`OdomF2M/MaxSize: 5000`), effectively benefiting from local loop-closure. No continuous MoCap ATE was measured during in-flight excursions of 50–70 cm.

---

## 2. Logic Chain

1. **Premise 1 (Citation Fidelity):** A scientific comparative review must accurately distinguish what each cited paper empirically measured from qualitative motivation or speculative extrapolation.
   - *From Obs 1.1:* Rong et al.'s baseline was an open-loop drive toward an initial pose estimate without continuous visual feedback. The document presents this as a general failure of PBVS, invents quantitative bounds `(5 - 25 mm)` not present in the paper, conflates ACT with Diffusion Policy, and blames wind in an indoor greenhouse experiment.
   - *From Obs 1.2:* Kim et al. never tested ORB-SLAM3 in the field to obtain "0.0% éxito". Instead, they used ORB-SLAM3 in a static environment as their gold-standard ground truth to validate their cube tracker. Conflating this into `Colapso total de ORB-SLAM3 y PBVS (0.0% éxito)` distorts Kim et al.'s methodology.
2. **Premise 2 (Engineering Realism in 3D Policy Learning):** Extrapolating DP3 to `visuomotor_ros2` must account for real sensor physics and edge computing constraints.
   - *From Obs 1.3:* Ze et al. achieved high fidelity because they used a ToF LiDAR (L515) and decimated depth to $84 \times 84$. Applying DP3 to raw $640 \times 480$ active stereo depth from a D435i introduces edge fattening, multi-path noise, and extreme computational overhead during Farthest Point Sampling ($O(N \cdot K)$). Furthermore, DP3 ran at 12.7 FPS on an RTX 3090; assuming it will run seamlessly at 20 Hz on an edge Jetson Orin Nano is an unvetted engineering leap.
   - Claiming that DDIM "never generates trajectories that penetrate physical surfaces" transforms a qualitative empirical finding on 4 tasks (80 trials) into a non-existent mathematical guarantee.
3. **Premise 3 (Metrological Rigor):** A comparative verdict is invalid if it compares fundamentally incommensurable metrics.
   - *From Obs 1.4:* Comparing an endpoint return drift of $2.307\text{ mm}$ (assisted by cached map keypoint re-association) against an OptiTrack-verified continuous trajectory RMSE (ATE of $6.1\text{ mm}$) is a category error. If UMI returned to a known visual marker, its PnP re-localization error would also be sub-millimeter. Conversely, `visuomotor_ros2`'s true mid-flight ATE without MoCap is unverified.
4. **Premise 4 (Deployment Fairness):** Criticizing UMI's offline SLAM as inferior for real-time robot deployment creates a false dichotomy, since UMI's robot deployment relies exclusively on $500\text{--}1000\text{ Hz}$ motor joint encoders, bypassing SLAM entirely.

---

## 3. Caveats

- **No Caveats Regarding Project Code:** The project code files (`visuomotor_core`, `visuomotor_ros2`, etc.) were kept strictly read-only.
- **Empirical Scope:** The audit did not execute a live physical OptiTrack session on `visuomotor_ros2` hardware, but strictly examined the reported telemetry logs and paper texts.

---

## 4. Adversarial Challenge Report

### Risk Assessment: HIGH
While the document exhibits impressive depth, theoretical elegance, and thoroughness, it contains several critical scientific exaggerations, metric conflations, and citation inaccuracies that undermine its academic rigor if submitted for formal peer review.

### Challenge Breakdown

#### [High] Challenge 1: Metric Incommensurability (ATE vs Return Error)
- **Assumption Challenged:** That `visuomotor_ros2` ($2.307\text{ mm}$) is metrologically superior to UMI ($6.1\text{ mm}$).
- **Attack Scenario:** A peer reviewer notes that UMI measured continuous 3D Absolute Trajectory Error (ATE) via OptiTrack MoCap over the entire duration of dynamic tasks, whereas `visuomotor_ros2` measured endpoint return drift to the origin cradle where RTAB-Map re-associated starting features.
- **Blast Radius:** Destroys the credibility of the primary quantitative claim in Section 1, Section 5.4, Section 6, and Section 7.1.
- **Mitigation:** Explicitly differentiate Return Drift ($e_{\text{return}} = 2.307\text{ mm}$) from continuous trajectory ATE ($e_{\text{ATE}} = 6.1\text{ mm}$). State that a full MoCap ATE benchmark for `visuomotor_ros2` is future work, while highlighting that $2.307\text{ mm}$ return drift demonstrates excellent loop-closure stability for table-top manipulation.

#### [High] Challenge 2: Misattribution and Conflation of Agricultural Literature (Kim et al. & Rong et al.)
- **Assumption Challenged:** That ORB-SLAM3 collapsed to 0.0% success in field harvesting, and that PBVS collapsed under dynamic wind and foliage.
- **Attack Scenario:** Reviewers read Kim et al. and find that Kim et al. used ORB-SLAM3 as the ground truth reference in a static room to prove their cube tracker worked, and that Rong et al. tested in a windless greenhouse on a rigid open-loop baseline.
- **Blast Radius:** Accusation of literature misrepresentation and selective quote-mining in Section 1 (ASCII diagram) and Section 4.
- **Mitigation:**
  1. Fix line 45 and Section 4.1: State that Kim et al. justified abandoning SLAM based on the qualitative domain realities of crop deformation, wind, and irreversibility, and used ORB-SLAM3 in a static scene to benchmark their cube tracker. Mention Kim et al.'s actual harvesting success rate ($28.95\%$).
  2. Clarify that Rong et al.'s baseline was a rigid open-loop drive toward an initial pose estimate without closed-loop visual feedback.
  3. Remove the fabricated quantitative intervals `(5 - 25 mm)` and `(25 - 40 mm)`, replacing them with Rong et al.'s qualitative designations and citing the single $25\text{ mm}$ example in Fig. 17(b).
  4. Change "ACT / Diffusion" to "Modified ACT" in Section 4.2 and Table 6/7.

#### [Medium] Challenge 3: Over-idealization of DP3 and Downplayed Hardware Realities
- **Assumption Challenged:** That DP3 can be directly deployed on D435i active stereo depth and Jetson Orin Nano edge hardware with a guaranteed +24.2% boost and 0.0% collisions.
- **Attack Scenario:** Deploying DP3 on D435i depth introduces stereo noise artifacts (flying pixels) not present in L515 LiDAR. Computing FPS on $640 \times 480$ depth on a Jetson Orin Nano drops frame rates to $< 5\text{ Hz}$.
- **Blast Radius:** Sets unrealistic expectations for Phase 2 roadmap; policy fails due to sensor mismatch and computational latency.
- **Mitigation:**
  1. Note that Ze et al. utilized an Intel RealSense L515 ToF LiDAR and downscaled depth to $84 \times 84$.
  2. Highlight the computational cost of Farthest Point Sampling ($O(N \cdot K)$) and inference latency ($12.7\text{ FPS}$ on desktop GPU), noting the necessity of Simple DP3 or optimized TensorRT kernels for 20 Hz edge execution.
  3. Clarify that +24.2% is a relative gain across 72 simulated tasks, and that 0.0% collision was an empirical observation over 80 real trials, not an unconditional mathematical guarantee.

#### [Medium] Challenge 4: The Deployment Strawman Regarding UMI
- **Assumption Challenged:** That UMI's lack of real-time SLAM during inference is an operational drawback.
- **Attack Scenario:** UMI's authors point out that executing 0 Hz SLAM on the robot is a deliberate architectural feature: the robot uses 1000 Hz joint encoders with zero latency and 0.2 mm repeatability.
- **Blast Radius:** Weakens the claim of `visuomotor_ros2`'s "operational superiority in deployment".
- **Mitigation:** Rephrase the comparison: `visuomotor_ros2` is superior for *interactive real-time data collection with immediate feedback* (zero post-processing delay), while UMI is superior for *tetherless in-the-wild collection*, with both relying on deterministic robot state (encoders or calibrated VIO) during deployment.

---

## 5. Conclusion & Formal Verdict

### Formal Verdict: **REQUEST_CHANGES**

The report `docs/research/umi_vs_rtabmap_vio_comparison.md` is a work of exceptional technical quality, but requires specific editorial adjustments to eliminate distortions, metric conflations, and literature misattributions before it can be considered academically flawless.

### Required Changes Checklist for Author/Editor:
1. **Differentiate ATE vs Return Error:** Disclaim that UMI's $6.1\text{ mm}$ is continuous MoCap ATE whereas `visuomotor_ros2`'s $2.307\text{ mm}$ is endpoint return drift to a cached keyframe map.
2. **De-conflate Kim et al. & ORB-SLAM3:** Clarify that Kim et al. used ORB-SLAM3 as a static proxy baseline and that 0.0% failure belongs strictly to Rong et al.'s open-loop baseline. Add Kim et al.'s $28.95\%$ harvesting success rate.
3. **Correct Rong et al. Citations:** Replace "ACT / Diffusion" with "Modified ACT", remove unverified deviation ranges `(5 - 25 mm)`/`(25 - 40 mm)`, and specify that the baseline lacked continuous visual servo feedback.
4. **Contextualize DP3 Realities:** Acknowledge L515 LiDAR vs D435i stereo differences, $84 \times 84$ input downscaling, FPS latency ($12.7\text{ FPS}$ base), relative vs absolute $+24.2\%$, and soften the 0.0% collision claim from a mathematical guarantee to an empirical finding.
5. **Rebalance the Deployment Narrative:** Acknowledge that UMI intentionally uses 1000 Hz joint encoders during robot deployment.

---

## 6. Verification Method

To verify these findings independently, execute:
```bash
# Verify Rong et al. text, lack of wind, Table 6 and Modified ACT
python3 -c "
import subprocess
t = subprocess.check_output(['pdftotext', 'bio/fruit_bearing.pdf', '-']).decode('utf-8', errors='ignore')
assert 'Table 6' in t and '0/13' in t
assert t.count('wind') == 0
assert 'Modified ACT' in t or 'Action Chunking with Transformers' in t
"

# Verify Kim et al. ORB-SLAM3 as proxy baseline and 28.95% success rate
python3 -c "
import subprocess
t = subprocess.check_output(['pdftotext', 'bio/pepper_harvesting.pdf', '-']).decode('utf-8', errors='ignore')
assert 'proxy baseline' in t and '28.95%' in t
"

# Verify Ze et al. (DP3) RealSense L515, 84x84 resolution, and Table XIV
python3 -c "
import subprocess
t = subprocess.check_output(['pdftotext', 'bio/dp3d.pdf', '-']).decode('utf-8', errors='ignore')
assert 'RealSense L515' in t
assert '84 × 84' in t or '84 x 84' in t
assert 'TABLE XIV' in t
"
```
If any of the above assertions fail, this challenge is invalidated. All assertions currently pass with exit code 0.
