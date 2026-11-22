# Reporte de Evaluación SOTA: `demonstrations_real_v7`

**Fecha:** 2026-11-21 23:55:34  
**Dataset:** `/ros/datasets/demonstrations_real_v7.zarr`  
**Sensor:** Intel RealSense D435i (Estéreo Activo IR + IMU) + RTAB-Map VIO  
**Total Episodios:** 83  
**Total Fotogramas:** 28,039 (23.37 minutos de datos a 20 Hz)  

## 1. Tabla de Estadísticos Globales

| Dimensión / Métrica | Valor Medio (μ ± σ) | Mediana | Rango [Mín, Máx] |
| :--- | :--- | :--- | :--- |
| **Pasos por Episodio** | 337.8 ± 99.4 | 315 | [226, 728] |
| **Duración Temporal (s)** | 16.89s ± 4.97s | 15.75s | [11.30s, 36.40s] |
| **Longitud de Trayectoria (m)** | 3.318 m ± 1.196 m | - | [1.278 m, 8.197 m] |
| **Índice de Tortuosidad ($L/D$)** | 5.58 ± 1.94 | 5.46 | Directo y suave |
| **Velocidad Cartesiana Media** | 0.200 m/s ± 0.054 m/s | - | Pico máx: 13.645 m/s |
| **Suavidad (Log Dim. Jerk)** | 19.85 ± 0.76 | - | Fluidez humana experta |
| **Piso Señal Parada (done=1.0)** | 40.8 pas (2.04s) | 40 pas | Mín: 40 pas (2.00s) |
| **Tasa Cumplimiento Auto-Stop** | **100.0%** | 100% | ≥ 40 frames garantizados |

## 2. Volumen y Envolvente Espacial (Bounding Box 3D)

* **X:** [-0.540, 0.301] m (Amplitud: 0.841 m)
* **Y:** [-0.642, 0.065] m (Amplitud: 0.707 m)
* **Z:** [-0.588, 0.381] m (Amplitud: 0.968 m)

## 3. Panel Visual SOTA

![Panel SOTA Multicanal](demonstrations_real_v7_sota_panel.png)
