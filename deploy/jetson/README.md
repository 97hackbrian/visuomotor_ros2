# Guía de Despliegue de Inferencia Visuomotora en NVIDIA Jetson Orin Nano (8 GB)

Este directorio contiene todo lo necesario para compilar y ejecutar el nodo de inferencia reactiva (`policy_node.py`) en una **NVIDIA Jetson Orin Nano (8 GB)** utilizando exclusivamente **Docker** y **Docker Compose**.

---

## 1. Arquitectura y Compatibilidad de Versiones

### Estado de tu Jetson (Detectado en `fastfetch`)
* **Host OS:** Ubuntu 20.04.6 LTS (`aarch64`)
* **Kernel:** `Linux 5.10.216-tegra` $\implies$ **JetPack 5.1.4 / 5.1.5 (L4T R35.6.x)**
* **Memoria:** 7.28 GiB LPDDR5 unificada

### Selección de Imagen Base Docker
En JetPack 5.x, el runtime de NVIDIA (`nvidia-container-cli`) monta los drivers CUDA 11.4 del host. Para garantizar compatibilidad binaria absoluta con la GPU Tegra sin errores de `libcuda.so` ni dependencias rotas de glibc:
* **JetPack 5.1.x (Por defecto):** Utiliza `dustynv/ros:humble-pytorch-l4t-r35.4.1` (Ubuntu 22.04 Jammy en contenedor, ROS 2 Humble, Python 3.10, PyTorch acelerado por hardware para Tegra).
* **JetPack 6.x (Si en el futuro actualizas la Jetson):** Puedes cambiar `BASE_IMAGE` a `dustynv/ros:jazzy-pytorch-l4t-r36.2.0` en `docker-compose.yml`.

> [!NOTE]
> **Compatibilidad ROS 2 sobre DDS:**
> El nodo de inferencia corriendo en ROS 2 Humble dentro del contenedor de la Jetson se comunica de forma transparente sobre la red DDS (`rmw_cyclonedds_cpp`, Domain ID 42) con cualquier nodo externo (cámaras, robots o computadoras corriendo ROS 2 Humble, Iron o Jazzy).

---

## 2. Preparación del Host (Se ejecuta una sola vez)

Antes de levantar Docker, optimiza la memoria y los relojes de la Jetson:

```bash
cd deploy/jetson/scripts
chmod +x setup_host.sh
./setup_host.sh
```

Este script automatiza:
1. Fijar el modo de energía a **15W Máximo Rendimiento** (`nvpmodel -m 0`).
2. Bloquear relojes de CPU, GPU y memoria a máxima frecuencia (`jetson_clocks`).
3. Activar una memoria **SWAP de 8 GB** para proteger la memoria unificada contra el *OOM Killer*.
4. Configurar `"default-runtime": "nvidia"` en `/etc/docker/daemon.json`.
5. Reducir la latencia de puertos USB serie a **1 ms** (`/sys/bus/usb-serial/devices/ttyUSB*/latency_timer`).

---

## 3. Despliegue en 3 Pasos

### Paso 1: Copiar los Pesos del Modelo
Copia los pesos y la configuración del modelo entrenado a la carpeta `deploy/jetson/model/`:

```bash
# Ejemplo usando el modelo maestro train_500_dataV7_fixed:
cp visuomotor_core/outputs/train_500_dataV7_fixed/model.safetensors deploy/jetson/model/
cp visuomotor_core/outputs/train_500_dataV7_fixed/config.json deploy/jetson/model/
```

### Paso 2: Ajustar Parámetros de la Cámara (Opcional)
Edita `deploy/jetson/config/jetson_policy_params.yaml` según los tópicos de tu cámara física:
```yaml
camera_topic: "/camera/color/image_raw"  # Tópico de tu cámara RealSense o USB
joint_states_topic: "/joint_states"
inference_hz: 20.0
num_inference_steps: 10
```

### Paso 3: Construir y Levantar el Contenedor
Desde la carpeta `deploy/jetson/`:

```bash
cd deploy/jetson
docker compose build
docker compose up -d
```

Para ver los logs de inferencia en tiempo real:
```bash
docker compose logs -f
```

Para detener el servicio:
```bash
docker compose down
```

---

## 4. Benchmark de Latencia y VRAM en Borde

Para verificar cuántos milisegundos toma la inferencia y cuánta memoria consume en la Jetson Orin Nano antes de conectar los motores físicos del robot:

```bash
# Abrir una terminal interactiva dentro del contenedor en ejecución
docker exec -it visuomotor_orin_inference bash

# Ejecutar el benchmark automatizado (evalúa FP32 vs FP16)
python3 /ros_ws/src/visuomotor_ros2/deploy/jetson/scripts/benchmark_jetson.py --model_path /ros_ws/model
```

### Resultados Esperados en Jetson Orin Nano (15W):
* **DDIM (10 pasos, FP16 con Tensor Cores):** $\approx 25 - 35\text{ ms}$ ($\mathbf{\approx 30\text{ Hz}}$, superando con holgura los 20 Hz requeridos).
* **Consumo VRAM pico:** $\approx 1.8\text{ GB}$ (dejando más de 5 GB libres en la Jetson).

---

## 5. Edición de Código en Vivo con VS Code (Modo Desarrollo Dual)

Si en algún momento deseas modificar `policy_node.py` o los módulos de `visuomotor_core` usando **VS Code** sin tener que reconstruir la imagen Docker (`docker compose build`), dispones de dos métodos directos:

### Método 1: Montaje en Vivo con Docker Compose Dev (Recomendado)
Levanta el contenedor utilizando el archivo de extensión `docker-compose.dev.yml`:

```bash
cd deploy/jetson
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

* **Cómo opera:** Monta en tiempo real las carpetas de código de tu host (`visuomotor_ros2/visuomotor_ros2` y `visuomotor_core/visuomotor_core`) directamente sobre el workspace del contenedor.
* **Flujo de trabajo:** Abres el repositorio en VS Code en la Jetson (o vía Remote SSH desde tu laptop), editas cualquier archivo `.py` y guardas (`Ctrl+S`). El cambio se ejecuta **de inmediato** en el robot sin recompilar.

### Método 2: VS Code "Attach to Running Container"
Con el contenedor estándar de Producción corriendo (`docker compose up -d`):
1. En VS Code, instala la extensión oficial **Dev Containers** (o **Docker**).
2. Presiona `F1` (o `Ctrl+Shift+P`) y selecciona:
   `Dev Containers: Attach to Running Container...` $\to$ elige `visuomotor_orin_inference`.
3. Abre la carpeta `/ros_ws/src/`.
4. Podrás editar código con resaltado de sintaxis, autocompletado y terminal interactiva con acceso a GPU y ROS 2 directamente dentro del contenedor.

