# Módulo 04: Dataloaders Zarr, Normalización y Manejo del Horizonte Temporal

Este documento documenta la ingesta de datos de entrenamiento, los mecanismos de preprocesamiento, la solución formal al truncamiento temporal y las estadísticas de normalización, auditado línea por línea en `visuomotor_core` (`dataset/zarr_dataset.py`, `dataset/normalize.py`, `dataset/dataset.py`, `train.py`).

---

## 1. Topología del Dataset en Formato Zarr

El almacenamiento de demostraciones robóticas se implementa sobre arrays multidimensionales en **Zarr** (utilizando compresión `Blosc(cname='zstd', clevel=5)`). Su estructura en disco organiza cada trayectoria de manera aislada:

```text
demonstrations_vX.zarr/
├── episode_0/
│   ├── image   [N_0, 256, 256, 3]  uint8   (Imágenes RGB de cámara de muñeca)
│   ├── state   [N_0, 8]            float32 (Propiocepción TCP + Gripper actual)
│   └── action  [N_0, 9]            float32 (Comandos futuros + done_flag)
├── episode_1/
│   ├── image   [N_1, 256, 256, 3]  uint8
│   ├── state   [N_1, 8]            float32
│   └── action  [N_1, 9]            float32
└── ...
```

* **Indexación Global Plana:** El dataloader (`zarr_dataset.py`) precarga una lista de tuplas `(ep_name, frame_idx)` recorriendo todos los episodios disponibles. Esto permite a PyTorch indexar el dataset con un `len(dataset)` total sin cargar imágenes masivas en memoria RAM.

---

## 2. El Bug Crítico de Truncamiento del Horizonte (*Horizon Truncation Bug*)

### Diagnóstico Matemático y Raíz del Problema
En arquitecturas de Diffusion Policy con *Action Chunking*, la red no predice un solo paso, sino un horizonte completo de $H_{act} = 16$ acciones futuras.

En dataloaders estándar, para evitar que una consulta en el fotograma terminal intente leer fuera de los límites del array $N$, se implementaba un corte estricto:
```python
# Slicing ingenuo / tradicional:
valid_frames = ep_length - self.h_act
```
O bien:
```python
if frame_idx + self.h_act > ep_length:
    continue # Se descarta la muestra
```

### Consecuencia Devastadora en la Señal de Término:
Si un episodio de recolección dura 200 fotogramas y la bandera de fin de tarea (`done = 1.0`) se grababa en los últimos 15 o 20 fotogramas:
1. El dataloader descartaba el $100\%$ de los fotogramas donde la señal de término existía.
2. La red entrenaba con una distribución donde el vector `action[:, 8]` era **estrictamente 0.0 en el 100% de los batches**.
3. En inferencia en vivo, el modelo jamás emitía un valor de fin de tarea superior a `0.001`, haciendo imposible el auto-stop neuronal.

---

## 3. Solución Arquitectónica: *Copy-Padding* Terminal (`zarr_dataset.py`)

Para permitir que la red aprenda la transición hacia el fin de la tarea y la permanencia en reposo, `zarr_dataset.py` implementa **Copy-Padding** terminal:

```python
total_ep_frames = group['action'].shape[0]
actual_actions = group['action'][frame_idx : min(total_ep_frames, frame_idx + self.h_act)]
actual_len = len(actual_actions)

if actual_len < self.h_act:
    pad_len = self.h_act - actual_len
    # Replicar el último vector de acción disponible (que retiene pose destino y done=1.0)
    pad_actions = np.repeat(actual_actions[-1:], pad_len, axis=0)
    actions = np.concatenate([actual_actions, pad_actions], axis=0)
    
    # Máscara booleana para supervisión de pérdida
    action_is_pad = torch.zeros(self.h_act, dtype=torch.bool)
    action_is_pad[actual_len:] = True
else:
    actions = actual_actions
    action_is_pad = torch.zeros(self.h_act, dtype=torch.bool)
```

### Beneficios Formales del Copy-Padding:
1. **Supervisión de Reposo:** Cuando el robot llega al destino, la red ve que las acciones futuras son estacionarias ($X, Y, Z$ constantes) y que `done` se mantiene en `1.0`. La red aprende no solo a moverse hacia el objetivo, sino a **permanecer quieto en él**.
2. **Máscara de Pérdida (`action_is_pad`):** Permite en `train.py` aplicar `do_mask_loss_for_padding`, evitando que los pasos repetidos artificialmente distorsionen la escala del gradiente si se desea aislar la supervisión.

---

## 4. Regularización por *Proprioception Dropout* (15%)

Auditado en `visuomotor_core/dataset/zarr_dataset.py`:

```python
# Proprioception Dropout (15% probabilidad)
if torch.rand(1).item() < 0.15:
    state = torch.zeros_like(state)
```

### Fundamento Teórico:
* **Problema de Sobre-dependencia Propioceptiva:** Los robots entrenados con estados cinemáticos tienden a apoyarse excesivamente en las coordenadas cartesianas previas, ejecutando trayectorias en lazo abierto (*dead-reckoning*) e ignorando la imagen de la cámara. Si el objeto se mueve o cambia de posición imprevista, la política falla catastróficamente.
* **Solución por Dropout:** En el $15\%$ de los pasos de entrenamiento, el vector propioceptivo se anula por completo con ceros. Esto obliga a la U-Net a extraer la intención de movimiento y la localización métrica exclusivamente a partir de las características visuales aprendidas por la ResNet-18 y SpatialSoftmax.

---

## 5. Pipeline de Normalización (`normalize.py`)

Para garantizar que el proceso de difusión Gaussiana opere con tensores numéricamente equilibrados, todas las observaciones cinemáticas y acciones se normalizan previo al entrenamiento y se desnormalizan en inferencia:

### A. Modo Min-Max (Intervalo $[-1, 1]$)
Configurado por defecto para coordenadas cartesianas, cuaterniones y comandos de garra:
$$x_{\text{norm}} = 2 \cdot \frac{x - x_{\min}}{x_{\max} - x_{\min} + \epsilon} - 1$$
$$x_{\text{orig}} = \frac{(x_{\text{norm}} + 1) \cdot (x_{\max} - x_{\min} + \epsilon)}{2} + x_{\min}$$
* **Salvaguarda Numérica:** $\epsilon = 1.0 \times 10^{-4}$ evita divisiones por cero en dimensiones constantes o estáticas.

### B. Modo Mean-Std (Distribución Estándar $\mathcal{N}(0, 1)$)
$$x_{\text{norm}} = \frac{x - \mu}{\sigma + \epsilon}$$

### C. Almacenamiento Persistente de Estadísticas:
Las estadísticas globales ($x_{\min}, x_{\max}, \mu, \sigma$) se computan una sola vez al inicio del entrenamiento y se guardan como buffers persistentes (`dataset_stats.pt` o dentro del checkpoint del modelo) para que `policy_node.py` en tiempo real cargue **exactamente** la misma escala métrica.

---

## 6. Política de Parcheo Seguro de Datasets (`patch_dataset_done_flag.py`)

* **Regla de Inmutabilidad:** Los datasets crudos generados durante sesiones de recolección (ej. `demonstrations_v7.zarr`) son inmutables y no deben ser modificados in-situ.
* **Flujo de Parcheo Aditivo:**  
  Si un dataset antiguo requiere corrección de la 9ª dimensión, el script de parcheo crea una copia explícita e independiente:
  ```bash
  python3 patch_dataset_done_flag.py \
    --input datasets/demonstrations_v7.zarr \
    --output datasets/demonstrations_v7_done_fixed.zarr \
    --done_frames 20
  ```
  Esto preserva la integridad del historial experimental original.

---

## 7. Tip Metodológico para Machine Learning: El Error Fuera de Bolsa (OOB - Out-of-Bag)

En machine learning tradicional, para medir generalización se suele apartar un conjunto de prueba (*test set*). El Bagging (*Bootstrap Aggregating*) ofrece una alternativa gratuita y honesta sin necesidad de apartar datos:

* **Identificación de Muestras OOB:** Para cada fila $i$, identificamos qué árboles no la incluyeron en su muestra bootstrap (`fuera[bolsa] == False`).
* **Evaluación Fuera de Distribución Local:** Esos árboles específicos nunca vieron a la observación $i$ durante su entrenamiento: para ellos, $i$ es un dato de prueba nuevo.
* **Votación / Agregación:** Hacemos votar a esos árboles sobre la fila $i$ y comparamos la predicción agregada con la etiqueta real $y_i$.
* **Exclusión de Casos Límite:** Si alguna fila quedó dentro de la bolsa de todos los árboles, no puede evaluarse y se excluye del cómputo del error global.

---

## 8. Tip Metodológico para Machine Learning: AdaBoost con Tocones de Decisión

A diferencia de *Bagging* (árboles independientes entrenados en paralelo), **AdaBoost** entrena modelos en secuencia (*boosting*) adaptativo:

### Algoritmo Paso a Paso:
1. **Inicialización:** Comenzamos con un vector de pesos uniforme:
   $$w_n = \frac{1}{N}, \quad \forall n \in \{1, \dots, N\}$$

2. **Iteración en cada ronda $t \in \{1, \dots, T\}$:**
   * **Entrenamiento Débil:** Entrenamos un modelo simple: un **tocón de decisión** (*decision stump*, árbol con una sola pregunta/corte binario) ponderado por el vector de pesos actual $\mathbf{w}$.
   * **Cálculo del Error Ponderado:**
     $$\epsilon_t = \sum_{n:\text{pred}_n \neq y_n} w_n$$
   * **Criterio de Parada:** Si $\epsilon_t \ge 0.5$, el modelo no supera al azar: se detiene el algoritmo sin incluirlo.
   * **Importancia del Modelo en la Votación ($\alpha_t$):**
     $$\alpha_t = \frac{1}{2} \ln\left( \frac{1 - \epsilon_t}{\epsilon_t} \right)$$
     *(Con un recorte numérico a $[10^{-10}, 1 - 10^{-10}]$ para evitar $\ln(0)$ o división por cero si $\epsilon_t = 0$)*.
   * **Actualización de los Pesos de los Datos:**
     $$w_n \leftarrow w_n \cdot \exp(-\alpha_t y_n \text{pred}_n)$$
     * Si el dato fue **bien clasificado**: $y_n \text{pred}_n = 1 \implies w_n \cdot e^{-\alpha_t}$ (el peso disminuye).
     * Si el dato fue **mal clasificado**: $y_n \text{pred}_n = -1 \implies w_n \cdot e^{+\alpha_t}$ (el peso aumenta).
   * **Renormalización:** Dividir por $\sum_{n=1}^N w_n$ para que la distribución de pesos vuelva a sumar $1$.

3. **Inferencia sobre una Muestra Nueva:**
   $$\text{predicción} = \text{signo}\left( \sum_{t=1}^T \alpha_t \cdot \text{tocón}_t(x) \right)$$
   Los tocones más precisos tienen un $\alpha_t$ grande y ejercen mayor influencia en la decisión; los tocones mediocres tienen un peso marginal.

### Aplicación Robótica en este Proyecto:
* **Clasificador Interpretable de Éxito/Fallo de Agarre:** Generar tocones sobre variables físicas continuas ($\delta_{\text{elastic}}$, velocidad en $Z$, orientación $q_w$) para obtener umbrales analíticos verificables y explicables.
* **Minería Activa de Demostraciones Difíciles (*Hard-Trajectory Mining*):** Utilizar la reponderación de AdaBoost para concentrar la recolección experta y el buffer de RL en regiones del espacio de trabajo donde la política actual comete fallos sistemáticos.
