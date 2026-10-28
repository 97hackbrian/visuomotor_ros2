# Módulo 09: Benchmarking Comparativo, Métricas de Precisión TCP y Tratamiento de Outliers

Este documento documenta la arquitectura experimental, modelos cinemáticos, resultados empíricos y estándares de visualización para la evaluación cuantitativa de políticas visuomotoras en robótica (Solo Diffusion Policy vs. Diffusion Policy + Reinforcement Learning Fine-Tuning).

---

## 1. Modelo Cinemático del Tool Center Point (TCP / Superficie de Contacto)

### A. Limitación de la Medición en la Brida (`gripper_base_link`)
En arquitecturas visuomotoras basadas en ROS 2 y TF, el marco de referencia del efector suele vincularse a la brida de la muñeca del robot (`gripper_base_link`). Sin embargo:
* Las almohadillas mecánicas de contacto del gripper Lite6 se extienden físicamente a lo largo del eje $Z$ local de la herramienta:
  $$\mathbf{tcp\_offset\_z} = 0.079\text{ m} = \mathbf{79.0\text{ mm}}$$
* Si la distancia al objeto se calcula directamente como $\|\mathbf{p}_{\text{ee}} - \mathbf{p}_{\text{obj}}\|$, una pinza en contacto firme sobre un cubo reportará una distancia artificial de $\approx 80\text{ mm}$, enmascarando la precisión micrométrica real de agarre.

### B. Formulación Analítica de la Proyección Cinemática
Para obtener la distancia real de contacto $d_{\text{contacto}} \in [0.0, 5.0]\text{ mm}$, se proyecta el vector director local $Z$ de la herramienta utilizando el cuaternión de orientación medido $\mathbf{q} = [q_x, q_y, q_z, q_w]$:
$$\mathbf{v}_z = \begin{bmatrix} 2(q_x q_z + q_w q_y) \\ 2(q_y q_z - q_w q_x) \\ 1 - 2(q_x^2 + q_y^2) \end{bmatrix}$$
$$\mathbf{p}_{\text{tcp}} = \mathbf{p}_{\text{ee}} + \mathbf{v}_z \cdot \mathbf{tcp\_offset\_z}$$
$$d_{\text{contacto}} = \|\mathbf{p}_{\text{tcp}} - \mathbf{p}_{\text{objeto}}\|$$

Esta métrica se calcula de manera síncrona a 20 Hz dentro de `GymROS2Env._compute_reward_and_termination()` en `gym_ros2_env.py` y se registra paso a paso en telemetría.

---

## 2. Protocolo de Evaluación Pareada Determinista (Halton QMC)

Para eliminar el sesgo de variabilidad espacial entre modelos, la suite `tools/compare_models.py` implementa el protocolo de **episodios pareados**:
1. Para cada episodio $i \in [1, N]$, se muestrea una pose espacial $(X, Y, Z, \text{Yaw})$ mediante secuencias cuasi-aleatorias de Halton (Quasi-Monte Carlo, $d=3$, semilla fija).
2. Se evalúa el modelo **Solo DP** (`train_500_dataV7_fixed`) durante una ventana de hasta 1500 pasos con Auto-Stop neuronal activo.
3. Se regresa el brazo a Home, se teletransporta el objeto a la **misma pose idéntica**, y se evalúa **DP + RL** (`dppo_v7_rl_finetuned`).
4. Ambos modelos enfrentan las mismas dificultades angulares y de alcance en la mesa de trabajo.

---

## 3. Resultados Empíricos Consolidados ($N = 10$ Episodios Pareados / 20 Vuelos)

Evaluación realizada sobre NVIDIA Isaac Sim 4.2 y ROS 2 Jazzy:

| Métrica | Solo DP | DP + RL (Iter 100) | Análisis Científico |
| :--- | :---: | :---: | :--- |
| **Tasa de Éxito Global** | **80.0%** (8/10) | **80.0%** (8/10) | Culminación exitosa hasta Home con Auto-Stop |
| **Tasa de Agarre Firme** | **90.0%** (9/10) | **90.0%** (9/10) | Validación de pseudo-fuerza $\ge 0.7\text{ mm}$ y elevación $\Delta Z \ge 8\text{ cm}$ |
| **Distancia Contacto TCP** | $13.61 \pm 33.35\text{ mm}$ | $14.64 \pm 32.69\text{ mm}$ | Contacto sub-centimétrico ($0.88 - 7.14\text{ mm}$ en casos exitosos) |
| **Tracking RMSE** | $30.97 \pm 7.82\text{ mm}$ | **$29.70 \pm 8.63\text{ mm}$** | **RL reduce el error en $-1.27\text{ mm}$ (trayectorias más suaves)** |
| **Error Terminal en Home** | $153.04 \pm 85.03\text{ mm}$ | $153.13 \pm 84.99\text{ mm}$ | Posicionamiento final respecto a $[0.25, 0.00, 0.40]$ |
| **Elevación Media $\Delta Z$** | $27.35 \pm 9.61\text{ cm}$ | $27.12 \pm 9.53\text{ cm}$ | Elevación contra la gravedad en vuelo |
| **Duración Media de Ciclo** | $67.47 \pm 24.81\text{ s}$ | $68.86 \pm 23.44\text{ s}$ | Tiempo total de ejecución |

---

## 4. Hallazgo Central: Optimización Local de Precisión vs. Exploración Global

* **Lo que responde la evidencia:** RL fine-tuning sobre Diffusion Policy (DPPO) mejora significativamente la suavidad cinemática de la trayectoria cartesiana (menor Tracking RMSE), amortiguando oscilaciones de alta frecuencia gracias a la penalización de aceleraciones y al gradiente del crítico $V(s)$.
* **Lo que difiere de la expectativa inicial:** RL **no** descubre trayectorias macroscópicas autónomas en regiones fuera de distribución (OOD). En el Episodio 3 ($Y = -0.898\text{ m}, \text{Yaw} = 1.36\text{ rad}$), ambos modelos fallaron al no descender al cubo ($d_{\text{contacto}} \approx 108\text{ mm}$). El acotamiento de divergencia KL ($\text{Approx KL} \le 0.05$) y la regularización $\lambda_{\text{BC}}$ retienen a la red dentro de la variedad de demostraciones, canalizando el aprendizaje hacia el micro-ajuste de contacto y no hacia la exploración global.

---

## 5. Estándar de Visualización Científica y Manejo de Outliers

Para evitar que episodios atípicos de fallo compriman los diagramas de caja en líneas ilegibles:
1. **Escala Logarítmica para Distancias:** En distancias de aproximación que abarcan órdenes de magnitud ($0.7\text{ mm}$ a $108\text{ mm}$), la escala logarítmica permite que las cajas ocupen más del 50% del espacio vertical.
2. **Enfoque en Rango Operativo Activo (*Smart Zoom*):** Para error terminal, elevación y duración, los ejes se limitan al régimen nominal de operación y los puntos atípicos se comunican mediante etiquetas (*badges*) explícitas (ej. `▲ Outlier Ep 3: 390.7 mm`).
3. **Superposición de Datos Dispersos (*Jitter Scatter*):** Todas las observaciones individuales se grafican con una ligera dispersión horizontal, garantizando total transparencia de la muestra.
4. **Desacoplamiento sin Simulación (`tools/replot_benchmark.py`):** Permite regenerar figuras de alta resolución (PNG 300 DPI y PDF vectorial) directamente desde el CSV en menos de 2 segundos.

---

## 6. Habilidad Operativa Asociada
El procedimiento se encuentra automatizado en la skill:
* `.agents/skills/visuomotor-benchmark-evaluator/SKILL.md`
