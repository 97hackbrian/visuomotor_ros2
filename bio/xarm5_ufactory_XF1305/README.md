# UFACTORY xArm 5 (XF1305): manuales oficiales

Modelo del laboratorio: **UFACTORY xArm 5, código XF1305** (caja "XF13"; controlador `DETAIL: 5,5,XF1305,DC1305`,
firmware v2.5.1). La portada del Hardware Manual v2.7.1 dice "Apply to Model: XF1305, XI1305, XS1305".

Descargados el 2026-11-22 de la página oficial https://www.ufactory.cc/download/ (sección UFACTORY xArm > Documents).

| Archivo | Documento | URL | sha256 |
|---|---|---|---|
| `xArm_Hardware_Manual_v2.7.1.pdf` | Hardware Manual v2.7.1 (55 p.) | https://www.ufactory.cc/upfiles/download/202611/RE7G3GY52WQE.pdf | `27d6c22cbdd9e5c6a2472d7e232254f03e7fba192abd3e8e5bf9ee9cb1ecb32b` |
| `xArm_Developer_Manual_v2.0.1.pdf` | Developer Manual v2.0.1 (125 p.) | https://www.ufactory.cc/upfiles/download/202608/T7SCNVJFSZR0.pdf | `eac4da444c51f2f66d298c5874731e76252ad03d833b98de3a7cfad8aad5e0c3` |
| `xArm_Quick_Start_Guide.pdf` | Quick Start Guide (7 p.) | https://www.ufactory.cc/upfiles/download/202611/GVZH4V52N5J8.pdf | `20adf949e7e5a52387416d8c7c978129e0131a825c90926032cf4407f8d5a84f` |

La misma página ofrece el modelo 3D del xArm 5 (XF1305) en STEP; no se descargó.

## Datos clave (con página)

| Dato | Valor | Fuente |
|---|---|---|
| Alcance nominal | 700 mm | Developer Manual §4.1 (p. 121) |
| Rango cartesiano | X ±700 mm, Y ±700 mm, Z -400..951.5 mm, roll/pitch/yaw ±180° | Hardware Manual §8.1; Developer Manual §4.1 |
| Envolvente de los eslabones | radio 770 mm, +1037 / -416 mm respecto a la base (sin efector final) | Hardware Manual §2.2.2, figura "working range of xArm5 and xArm6" (p. 12) |
| Rangos articulares | J1 ±360°, J2 -117°..116°, J3 -219°..10°, J4 -97°..180°, J5 ±360° | Hardware Manual, Preface (p. 2) y §8.2 |
| Carga útil máx. | 3 kg | Hardware Manual §8.2 |
| Peso (solo brazo) | 11.3 kg | Hardware Manual §8.2 |
| Velocidad máx. del efector | 1 m/s; juntas 180°/s | Hardware Manual §8.1 |
| Repetibilidad | ±0.1 mm | Hardware Manual §8.1 |
| Brida | DIN ISO 9409-1-A50/63 | Hardware Manual §8.1 |

## Cómo se usan en el repo

- **700 mm es el radio máximo, no el alcance lateral a la altura de la tarea.** La esfera de alcance está centrada en
  el hombro (J2, z = 0.267 m). A una profundidad x, el alcance lateral es aproximadamente √(R² − x²). Con la brida a
  ~0.28 m y la garra vertical (aproximación de `demonstrations_real_v7`), el alcance lateral es ±0.47 m a la
  profundidad de los objetos, con el 95 % de margen que usa el puente. Ver `tools/xarm5/analyze_xarm5_workspace.py` y
  `datasets/evaluations_real_v7_xarm5_v2/fig_xarm5_workspace_vs_demos.png`.
- **La caja `|y| <= 0.30 m` de `xarm5_real_params.yaml` es una decisión de seguridad de la celda, no del brazo.** Se
  puede ampliar tras medir la celda real; el brazo llega más lejos.
- **Rangos articulares v2.7.1.** Los toma el guard `joint_*_limits_rad` del puente desde el 2026-11-22; antes usaba
  la URDF y el manual V2.3.0, con J2 hasta 120° y J3 hasta -225°.
