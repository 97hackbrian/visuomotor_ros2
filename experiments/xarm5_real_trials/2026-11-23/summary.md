# Ensayos de la política en el xArm5 real

Modelo: época 200 de `train_real_v7x5v2_relative_ee_e300` (`deploy/jetson/model`), 5 pasos DDIM + CUDA Graph, puente a 5 cm/s. Referencia: últimos frames de 76 demos.

* **Tasa de éxito (automática): 6/7 = 86 %** (IC Wilson 95 %: 49–97 %).
* Sin confirmación del usuario todavía.
* Parada automática: 7/7; tiempo hasta la parada: mediana 23.5 s (demos: 15.3 s de grabación).

| Intento | Posición | Éxito | Parada (s) | Vista final (z u, z v; escala) | Final TCP (m) | Recorrido (mm) |
|---|---|---|---|---|---|---|
| trial_001 | centro (igual a la vista HOME3) | sí | 20.6 | [-1.78, -0.92]; 0.85 | [0.5118, -0.0555, 0.342] | 617.9 |
| trial_002 | centro (repetición) | sí | 23.5 | [-1.29, -1.03]; 0.88 | [0.5081, -0.0501, 0.3342] | 621.6 |
| trial_003 | izquierda | sí | 30.2 | [-2.06, -0.06]; 1.06 | [0.4507, 0.1625, 0.3193] | 604.7 |
| trial_004 | izquierda (repetición) | sí | 28.7 | [-2.13, -0.23]; 1.07 | [0.4674, 0.1678, 0.3196] | 642.0 |
| trial_005 | derecha (extremo) | sí | 20.8 | [-0.24, 0.62]; 1.01 | [0.4877, -0.2458, 0.3244] | 653.0 |
| trial_006 | derecha (extremo, repetición) | sí | 19.9 | [0.14, -0.02]; 0.93 | [0.4877, -0.244, 0.3299] | 636.9 |
| trial_007 | centro-lejos, girada en yaw | no | 29.1 | [3.62, 0.91]; 1.49 | [0.6023, -0.0373, 0.2573] | 720.5 |
