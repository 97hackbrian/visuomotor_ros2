# Historial del 2026-11-26 (antes de los nombres fijos)

Registros de los armados y conversiones del primer día. Se conservan porque documentan los dos errores encontrados y
corregidos (ver `.agents/knowledge/19_v8_handheld_dataset_pipeline.md`):

| Archivo | Dataset que produjo (ya borrado, a pedido del usuario) | Qué tenía |
|---|---|---|
| `build_droid.log` | `demonstrations_real_v8_droid.zarr` (armado v1) | La referencia de inicio salía del giróscopo y borraba hasta 17 cm de movimiento real al inicio |
| `build_droid_v2.log` | `demonstrations_real_v8_droid_v2.zarr` (armado v2) | Correcto |
| `convert_xarm5.log`, `verify_demonstrations_real_v8_xarm5.json` | `demonstrations_real_v8_xarm5.zarr` (conversión v1) | Muñeca a ±149°: 10 demos con las hojas a 180° con saltos de giro |
| `convert_xarm5_v2.log`, `verify_demonstrations_real_v8_xarm5_v2.json` | `demonstrations_real_v8_xarm5_v2.zarr` (conversión v2) | Correcto (muñeca a ±177°) |

Desde el 2026-11-26, `tools/xarm5/make_v8_training_set.sh` rehace un solo dataset de entrenamiento con nombre fijo,
`datasets/demonstrations_real_v8_xarm5.zarr`, y deja aquí, en la carpeta de arriba, `build.log`, `convert.log`,
`verify.log` y `verify.json` del último armado.

`report_119_validadas.csv`, `summary_119_validadas.json` y `start_poses_119_validadas.json` son la validación sobre
119 demos: 106 OK, 8 REVISAR y 5 REPETIR. Es la evidencia por la que el usuario pidió descartar episode_5, 40, 71,
101 y 117. Los archivos de arriba ya no las incluyen porque están descartadas; las poses de inicio de las otras 114
son idénticas.
