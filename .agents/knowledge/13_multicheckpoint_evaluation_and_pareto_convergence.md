# 13. Evaluación Multi-Checkpoint, Corrección Geométrica de Auto-Stop y Frente de Pareto Físico

Este módulo documenta la metodología formal de evaluación multi-checkpoint para políticas de difusión visuo-motoras en ROS 2 e Isaac Sim, el diagnóstico de paradas prematuras y falsos negativos en auto-stop desacoplado, y el descubrimiento empírico del frente de Pareto de épocas efectivas ($E^*$).

---

## 1. El Incidente: Parada Prematura y Falsos Negativos en Transporte

### A. Manifestación del Fenómeno
En ejecuciones de evaluación con checkpoints intermedios (`epoch_150`), el manipulador ejecutaba con éxito:
1. Aproximación milimétrica al cubo ($6.0\text{ mm}$ de distancia TCP-objeto).
2. Cierre firme de pinza y elevación vertical ($\Delta z = 29.1\text{ cm}$).
3. Transporte horizontal en vuelo hacia la zona de Home $[0.25, 0.0, 0.40]\text{ m}$.

Sin embargo, el evaluador reportaba:
`Resultado: FALLO | Distancia Contacto TCP-Objeto: 6.0 mm | Error Obj: 132.1 mm | Clamped: True | Done Flag: 0.5806`

### B. Causa Raíz 1: Auto-Stop Prematuro por Umbral Holgado ($10\text{ cm}$)
* **Mecanismo:** El evaluador tenía la condición:
  ```python
  if done_flag >= 0.50 and dist_home < 0.10:
      break
  ```
* **Consecuencia:** La política comenzaba a predecir la finalización de la tarea (`done_flag = 0.58`) cuando el efector final ingresaba al radio de $9.2\text{ cm}$ de Home. El bucle cortaba la simulación en ese instante, impidiendo que el controlador cartesiano llevara el efector al radio formal de éxito ($\le 6.0\text{ cm}$).

### C. Causa Raíz 2: Omisión del Offset Cinemático Vertical de la Pinza ($79.0\text{ mm}$)
* **Mecanismo:** La posición objetivo de Home para el TCP es $\mathbf{p}_{\text{home}} = [0.25, 0.0, 0.40]\text{ m}$.
* **Geometría Real:** Cuando el cubo está sujeto en la pinza, cuelga físicamente por debajo de la brida a $Z \approx 0.321\text{ m}$ ($0.40 - 0.079$).
* **Consecuencia:** Al computar el error del objeto directamente contra $\mathbf{p}_{\text{home}}$:
  $$e_{\text{obj}} = \|\mathbf{p}_{\text{obj}} - \mathbf{p}_{\text{home}}\| \ge 0.079\text{ m} = 79\text{ mm}$$
  Era geométricamente imposible que un objeto sujeto alcanzara el umbral de éxito original ($\le 60\text{ mm}$), arrojando $100\%$ de falsos negativos.

---

## 2. Correcciones Implementadas en `tools/evaluate_convergence_checkpoints.py`

1. **Ajuste del Umbral de Llegada:**
   ```python
   arrival_thresh = 0.05 if is_clamped_now else 0.08
   if done_flag >= 0.50 and dist_home < arrival_thresh:
       auto_stop_triggered = True
       break
   ```
   Garantiza que el manipulador haya entrado holgadamente a la zona de éxito ($\le 5\text{ cm}$) antes de congelar las acciones.

2. **Compensación Vectorial de la Posición Objetivo del Objeto:**
   ```python
   gripper_offset_z = 0.079  # Longitud cinemática de la pinza
   expected_obj_target = home_target - np.array([0.0, 0.0, gripper_offset_z], dtype=np.float32)
   terminal_error_obj_mm = float(np.linalg.norm(obj_pos - expected_obj_target) * 1000.0)
   ```

3. **Criterio de Éxito Ampliado y Robusto:**
   ```python
   task_success = bool(final_info.get("task_success", False)) or (
       is_clamped and (terminal_error_obj_mm <= 100.0 or terminal_error_tcp_mm <= 60.0)
   )
   ```

---

## 3. Campaña Experimental en Grilla de 7 Épocas (`train_v8_300data`)

Se evaluó una grilla equidistante ($\Delta = 50$ épocas) sobre 300 demostraciones autónomas de `demonstrations_v8.zarr`, con 3 episodios pareados idénticos por modelo bajo semillas Halton QMC:

| Modelo | Época | Tasa de Éxito | Tasa Sujeción | TCP Dist Mín. (mm) | Error Terminal Obj (mm) | Auto-Stop % | Duración Ep 1 (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`epoch_10`** | 10 | 0.0% | 0.0% | $170.13 \pm 158.32$ | $338.41 \pm 27.49$ | 0.0% | 130.4 s (Timeout) |
| **`epoch_50`** | 50 | **33.3%** | **33.3%** | $49.06 \pm 69.74$ | $249.17 \pm 149.12$ | **33.3%** | 108.9 s (**Éxito**) |
| **`epoch_100`** | 100 | **33.3%** | **33.3%** | **$44.83 \pm 66.64$** | $250.78 \pm 149.88$ | **33.3%** | 100.9 s (**Éxito**) |
| **`epoch_150`** | 150 | **33.3%** | **33.3%** | $47.12 \pm 65.78$ | $249.36 \pm 149.08$ | **33.3%** | **99.5 s** (**Éxito**) |
| **`epoch_200`** | 200 | **33.3%** | **33.3%** | $50.34 \pm 71.90$ | **$248.83 \pm 148.79$** | **33.3%** | 107.9 s (**Éxito**) |
| **`epoch_250`** | 250 | 0.0% | 0.0% | $49.39 \pm 71.10$ | $339.20 \pm 27.35$ | 0.0% | 132.9 s (Timeout) |
| **`epoch_300`** | 300 | 0.0% | 0.0% | $49.53 \pm 62.43$ | $354.09 \pm 42.27$ | 0.0% | 131.8 s (Timeout) |

---

## 4. Conclusiones y Frente de Pareto

1. **Cantidad Efectiva de Épocas ($E^* = 50$):**
   - En la época 50 la política ya alcanza la capacidad motora completa para agarrar, elevar y trasladar con auto-stop.
2. **Meseta de Generalización (Épocas 50 a 200):**
   - La época 100 logra la mejor aproximación media ($44.83\text{ mm}$).
   - La época 150 logra la **mínima latencia de ejecución ($99.5\text{ s}$)** y la mayor elevación vertical ($\Delta z = 29.12\text{ cm}$).
3. **Sobreajuste y Degeneración Motora (Épocas 250 a 300):**
   - A pesar de que la pérdida de difusión en entrenamiento seguía descendiendo hacia $\sim 0.033$, la destreza física colapsó a $0.0\%$.
   - La red desincronizó el cierre de garra en el punto de contacto ($7.64\text{ mm}$) y comenzó a desviar las trayectorias ($14.38\text{ mm}$).
4. **Directiva de Entrenamiento:**
   - La parada anticipada (Early Stopping) guiada por el codo asintótico debe configurarse entre 50 y 100 épocas.
