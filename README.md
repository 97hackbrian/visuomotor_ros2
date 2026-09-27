# Visuomotor Generative Framework (ROS 2)

Este meta-repositorio (`visuomotor_ros2`) consolida todo el ciclo de vida para el entrenamiento e inferencia de políticas visuomotoras (Diffusion Policies, Flow Matching) diseñadas para brazos robóticos en ROS 2. 

## Arquitectura

El sistema está estrictamente particionado para separar las dependencias de Machine Learning (Off-ROS) del runtime de tiempo real (ROS 2) y de los drivers de hardware.

1. **`visuomotor_core`**: (Python Puro) Contiene la arquitectura del modelo en PyTorch, esquemas DDIM/DDPM, bucles de entrenamiento y utilidades para Dataloaders (Zarr, HuggingFace).
2. **`visuomotor_msgs`**: (ament_cmake) Define las interfaces estándar (`ActionChunk.msg`, `ObservationData.msg`) independientes del hardware.
3. **`visuomotor_data_collection`**: (ament_python) Nodo de muestreo sincrónico que lee desde simuladores o hardware y empaqueta episodios en formato Zarr. **Incluye un Demostrador PID Autónomo**.
4. **`visuomotor_ros2`**: (ament_python) Nodo central de inferencia multihilo (`policy_node.py`).
5. **`visuomotor_bringup`**: (ament_cmake) Capa de abstracción del hardware. Aquí deben residir los launch files específicos (ej. Isaac Sim, xArm Lite6).

## Generación de Datos Autónoma (Simulación)

Para facilitar la generación de datos a gran escala, `sim_collector_node.py` integra un **Controlador PID Cartesiano Autónomo**. 
- Calcula el error entre el efector final y el objeto usando el árbol TF (`tf2_ros`).
- Genera comandos espaciales de corrección.
- **Lógica de Gripper**: Si la distancia al objeto es menor a 3 cm, asume un cierre de gripper (`0.0`), de lo contrario lo mantiene abierto (`1.0`). La acción registrada en el Zarr es de dimensión 8 (7DoF Pose + 1DoF Gripper).

## Flujo de Trabajo

1. **Colección:** Lanza el robot en simulación y ejecuta `data_collection.launch.py`. Usa el servicio `~/trigger_episode` con comandos `START` y `SAVE` para grabar.
2. **Entrenamiento:** Usa `visuomotor_core/train.py` pasándole la ruta del dataset Zarr.
3. **Inferencia:** Modifica el `policy_params.yaml` para apuntar a tu `.pth` y lanza `inference.launch.py`. La inferencia publicará automáticamente secuencias de acciones en `policy/action_chunk`.

## Consideraciones sobre el Gripper
El diseño de la red en `visuomotor_core/models/config.py` asume que el gripper es una variable **continua** entre `0.0` y `1.0` y no un booleano, ya que las redes neuronales operan sobre distribuciones de probabilidad (ruido Gaussiano) durante el proceso de *denoising*. Si `use_gripper: True`, las dimensiones de acción aumentan de 7 a 8 automáticamente.