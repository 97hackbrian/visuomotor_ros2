# Original User Request

## Approved Task Prompt — 2026-11-20T20:04:02Z

Investigación bibliográfica, técnica y comparativa exhaustiva del estado del arte (SOTA) sobre Visual SLAM / VIO para recolección de demostraciones expertas en manipulación visuomotora (Diffusion Policy), analizando los papers locales en `bio/` (especialmente `bio/umi.pdf` y `bio/pepper_harvesting.pdf`) y repositorios oficiales de UMI y harvesting frente a la configuración activa en tiempo real con RealSense D435i + RTAB-Map VIO en visuomotor_ros2.

Working directory: /home/hackbrian/Documents/gits/visuomotor_ros2
Integrity mode: development
The team must NOT: Modificar archivos de código fuente existentes ni crear commits en Git.

## Verification Resources
- Directorio local `bio/` con los papers en PDF:
  * `bio/umi.pdf`: Paper oficial de UMI (Universal Manipulation Interface, Chi et al.).
  * `bio/pepper_harvesting.pdf`: Paper de manipulación y harvesting agrícola.
  * `bio/fruit_bearing.pdf`: Manipulación y harvesting en entornos no estructurados.
  * `bio/action_diffusion.pdf`: Diffusion Policy fundacional.
  * `bio/dp3d.pdf`: 3D Diffusion Policy (DP3).
  * `bio/dppo.pdf`: Diffusion Policy Policy Optimization.

## Requirements

### R1. Auditoría de papers locales en `bio/` y literatura de UMI / harvesting
Leer y auditar en profundidad los documentos PDF disponibles localmente en `bio/` (priorizando `bio/umi.pdf`, `bio/pepper_harvesting.pdf` y afines), complementándolos con la literatura SOTA en la red y repositorios oficiales de código abierto. Determinar con exactitud qué pipeline de Visual SLAM / Odometría emplean (ORB-SLAM3 modificado, COLMAP, DROID-SLAM u otros), cómo gestionan la óptica ojo de pez de GoPro y el espejo lateral para tracking de garra, cómo resuelven la recuperación de escala métrica 3D sin sensor estéreo nativo, y cómo articulan el procesamiento offline por lotes frente al tiempo real.

### R2. Estudio comparativo técnico frente a la configuración activa de visuomotor_ros2
Realizar una comparativa técnica rigurosa punto por punto entre los métodos de los papers analizados (UMI / harvesting) y la configuración activa del repositorio (Intel RealSense D435i con estéreo activo + IMU Bosch BMI085 a 200 Hz con imu_filter_madgwick, RTAB-Map rgbd_odometry en ROS 2 Jazzy a 20 Hz, Frame-to-Map, ventana de profundidad 0.20-1.80 m, Odom/AlignWithGround: false, OdomF2M/BundleAdjustment: 0 para preservar escala estéreo nativa, Odom/GuessMotion: false, y filtro Kalman temporal, con error de retorno de 2.307 mm y jitter de 1.61 mm). Determinar fundamentadamente si la configuración de visuomotor_ros2 es equivalente, superior o inferior, identificando ventajas operativas, trade-offs de latencia, precisión métrica, robustez ante oclusiones y viabilidad para entrenar Diffusion Policy.

### R3. Generación del reporte técnico documental
Redactar y guardar un informe técnico completo, formal y exhaustivo en `docs/research/umi_vs_rtabmap_vio_comparison.md`, además de presentar un resumen técnico ejecutivo en el mensaje final. El informe debe estructurarse con rigor científico, citando formalmente los papers de `bio/`, referencias externas, repositorios de código y archivos de configuración del proyecto. No modificar ningún archivo de código fuente existente del repositorio ni generar commits en Git.

## Acceptance Criteria

### Análisis de papers en `bio/` y pipeline de UMI / harvesting
- [ ] Extrae y cita directamente la información de los PDFs locales en `bio/` (específicamente `bio/umi.pdf` y `bio/pepper_harvesting.pdf`).
- [ ] Documenta el pipeline exacto de SLAM/Odometría de UMI (detallando el papel exacto de ORB-SLAM3 modificado u otros backends como DROID-SLAM o COLMAP).
- [ ] Explica cómo UMI resuelve la ambigüedad de escala métrica monocular (calibraciones métricas, targets ArUco/ChArUco, IMU o marcadores de pinza).
- [ ] Describe la sincronización temporal y el procesamiento (offline batch processing tras la sesión vs. estimación en tiempo real).
- [ ] Analiza el pipeline sensorial y de odometría/SLAM reportado en los papers de harvesting (`bio/pepper_harvesting.pdf`, etc.) y otros frameworks SOTA (DexCap, Dobb-E, Mobile ALOHA).

### Comparativa técnica frente a RealSense D435i + RTAB-Map VIO
- [ ] Contrasta sistemáticamente la solución de UMI contra la configuración activa de `visuomotor_ros2` en una matriz comparativa detallada que evalúe:
  - Frecuencia y latencia (20 Hz sincrónico en tiempo real vs. pipeline offline de UMI).
  - Escala métrica (estéreo activo con disparidad infrarroja nativa vs. estimación/escala de UMI).
  - Deriva acumulada y error de retorno (comparativa frente al benchmark empírico certificado de 2.307 mm).
  - Robustez sensorial (patrón IR activo en texturas pobres vs. visión RGB monocular gran angular).
  - Costo, portabilidad y carga mecánica de hardware (gripper con GoPro/espejo vs. RealSense en brazo/wrist).
- [ ] Emite una conclusión técnica concluyente, justificada matemática y empíricamente, sobre si la configuración actual es equivalente, mejor o complementaria para entrenar Diffusion Policy.

### Integridad del repositorio y entregable
- [ ] El reporte técnico completo se encuentra escrito en `docs/research/umi_vs_rtabmap_vio_comparison.md`.
- [ ] Ningún archivo de código fuente preexistente (`visuomotor_core`, `visuomotor_bringup`, `visuomotor_ros2`, `visuomotor_data_collection`, `visuomotor_msgs`) ha sido editado.
- [ ] `git status` en el host permanece limpio de commits no autorizados.
