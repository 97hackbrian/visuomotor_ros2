# Reporte de Evaluación SOTA: `demonstrations_real_v7_xarm5`

**Fecha:** 2026-11-22 00:04:56  
**Dataset:** `/ros/datasets/demonstrations_real_v7_xarm5.zarr`  
**Sensor:** Intel RealSense D435i (Estéreo Activo IR + IMU) + RTAB-Map VIO  
**Total Episodios:** 81  
**Total Fotogramas:** 26,868 (22.39 minutos de datos a 20 Hz)  

## 1. Tabla de Estadísticos Globales

| Dimensión / Métrica | Valor Medio (μ ± σ) | Mediana | Rango [Mín, Máx] |
| :--- | :--- | :--- | :--- |
| **Pasos por Episodio** | 331.7 ± 92.6 | 309 | [226, 728] |
| **Duración Temporal (s)** | 16.59s ± 4.63s | 15.45s | [11.30s, 36.40s] |
| **Longitud de Trayectoria (m)** | 3.240 m ± 1.100 m | - | [1.278 m, 8.197 m] |
| **Índice de Tortuosidad ($L/D$)** | 5.46 ± 1.80 | 5.36 | Directo y suave |
| **Velocidad Cartesiana Media** | 0.199 m/s ± 0.055 m/s | - | Pico máx: 1.810 m/s |
| **Suavidad (Log Dim. Jerk)** | 19.76 ± 0.52 | - | Fluidez humana experta |
| **Piso Señal Parada (done=1.0)** | 40.8 pas (2.04s) | 40 pas | Mín: 40 pas (2.00s) |
| **Tasa Cumplimiento Auto-Stop** | **100.0%** | 100% | ≥ 40 frames garantizados |

## 2. Volumen y Envolvente Espacial (Bounding Box 3D)

* **X:** [-0.194, 0.295] m (Amplitud: 0.489 m)
* **Y:** [-0.325, 0.308] m (Amplitud: 0.633 m)
* **Z:** [-0.720, 0.017] m (Amplitud: 0.736 m)

## 3. Panel Visual SOTA

![Panel SOTA Multicanal](demonstrations_real_v7_xarm5_sota_panel.png)
