# Ensayos de la política en el xArm5 real

Modelo: época 200 de `train_real_v7x5v2_relative_ee_e300` (`deploy/jetson/model`), 5 pasos DDIM + CUDA Graph, puente a 5 cm/s. Referencia: últimos frames de 76 demos.

* **Llegó y se detuvo sobre la herramienta (parada automática + cámara sobre las tijeras): 8/11 = 73 %** (IC Wilson 95 %: 43–90 %).
* **Terminó como en las demos (criterio estricto de vista final: encuadre y distancia): 6/11 = 55 %** (IC 95 %: 28–79 %).
* La altura frente a las demos del mismo lugar es solo informativa: en las demos la altura de parada sobre un mismo lugar varía ±7 cm.
* Confirmada por el usuario: 1/1.
* Parada automática: 8/11; tiempo hasta la parada: mediana 25.9 s (demos: 15.3 s de grabación).

| Intento | Posición | Sobre la herramienta | Como en las demos | Usuario | Parada (s) | Altura final / rango de las demos vecinas (m) | Vista final (z u, z v; escala) | Final TCP (m) |
|---|---|---|---|---|---|---|---|---|
| trial_001 | centro (igual a la vista HOME3) | sí | sí | — | 20.6 | 0.342 / [0.188, 0.29] | [-1.78, -0.92]; 0.85 | [0.5118, -0.0555, 0.342] |
| trial_002 | centro (repetición) | sí | sí | — | 23.5 | 0.3342 / [0.188, 0.29] | [-1.29, -1.03]; 0.88 | [0.5081, -0.0501, 0.3342] |
| trial_003 | izquierda | sí | sí | — | 30.2 | 0.3193 / [0.24, 0.404] | [-2.06, -0.06]; 1.06 | [0.4507, 0.1625, 0.3193] |
| trial_004 | izquierda (repetición) | sí | sí | — | 28.7 | 0.3196 / [0.24, 0.404] | [-2.13, -0.23]; 1.07 | [0.4674, 0.1678, 0.3196] |
| trial_005 | derecha (extremo) | sí | sí | — | 20.8 | 0.3244 / [0.164, 0.337] | [-0.24, 0.62]; 1.01 | [0.4877, -0.2458, 0.3244] |
| trial_006 | derecha (extremo, repetición) | sí | sí | — | 19.9 | 0.3299 / [0.164, 0.337] | [0.14, -0.02]; 0.93 | [0.4877, -0.244, 0.3299] |
| trial_007 | centro-lejos, girada en yaw | sí | no | sí | 29.1 | 0.2573 / [0.16, 0.335] | [3.62, 0.91]; 1.49 | [0.6023, -0.0373, 0.2573] |
| trial_008 | abajo-derecha (extremo, cerca de la base) | sí | no | — | 28.2 | 0.239 / [0.257, 0.329] | [3.74, -2.3]; 1.61 | [0.431, -0.2625, 0.239] |
| trial_009 | izquierda, tijeras en diagonal | no | no | — | None | 0.2525 / [0.188, 0.335] | [-4.61, -5.1]; 2.0 | [0.5381, -0.0048, 0.2525] |
| trial_010 | izquierda, diagonal (repetición del 9) | no | no | — | None | 0.149 / [0.16, 0.335] | None; None | [0.6061, -0.0794, 0.149] |
| trial_011 | centro, junto al borde frente a la base | no | no | — | None | 0.1498 / [0.21, 0.459] | [3.91, -5.24]; 1.79 | [0.4654, -0.0707, 0.1498] |
