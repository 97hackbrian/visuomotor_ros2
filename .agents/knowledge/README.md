# Base de Conocimiento Modular (Memoria Infinita del Proyecto)

Esta carpeta (`.agents/knowledge/`) conforma la **memoria a largo plazo persistente** del repositorio `visuomotor_ros2`. Su propósito es almacenar de manera modular, detallada y acumulativa todo el conocimiento técnico, decisiones arquitectónicas, modelos matemáticos y bitácoras de depuración para cualquier agente de IA (Antigravity, Cursor, Windsurf, Claude Code, etc.) y desarrollador humano.

---

## Índice de Módulos Temáticos

| Archivo | Tema Principal | Contenido Clave |
| :--- | :--- | :--- |
| **[`01_diffusion_policy_architecture.md`](./01_diffusion_policy_architecture.md)** | Arquitectura Neuronal | 1D U-Net, ResNet-18 GroupNorm, SpatialSoftmax, FiLM, DDIM Scheduler, Loss de Difusión. |
| **[`02_tensors_and_autostop.md`](./02_tensors_and_autostop.md)** | Tensores y Fin de Tarea | 8D Observación vs 9D Acción, prevención de Fuga Causal (*Causal Leakage*), Auto-Stop Neuronal vs Heurístico. |
| **[`03_data_collection_and_fsm.md`](./03_data_collection_and_fsm.md)** | Recolección de Datos | Generador experto FSM (`pickup_approximate.py`), PID cartesiano, Respawn Quasi-Monte Carlo (Halton), Freno Z y corrección de cuaterniones. |
| **[`04_dataloaders_and_datasets.md`](./04_dataloaders_and_datasets.md)** | Dataloaders y Zarr | Formato Zarr (`mode='a'`), Solución *Copy-Padding* para el bug de truncamiento del horizonte, parcheo seguro de datasets. |
| **[`05_troubleshooting_and_environment.md`](./05_troubleshooting_and_environment.md)** | Entorno y Troubleshooting | Configuración Docker (`-u hackbrian`), CycloneDDS vs FastDDS, parches de NumPy/cv_bridge, TFs y TF Frames. |
| **[`06_reinforcement_learning_dppo.md`](./06_reinforcement_learning_dppo.md)** | Fine-Tuning con RL | DPPO, MDP sobre espacio de difusión ($K_{\text{ft}}=5$), optimización PPO con clipping y pérdida BC. |
| **[`06_telemetry_metrics_and_rl_evaluation.md`](./06_telemetry_metrics_and_rl_evaluation.md)** | Telemetría y Métricas RL | Tópicos en tiempo real (`/rl/metrics/`), registro CSV y curvas headless con matplotlib. |
| **[`06_scientific_latex_reporting_and_tikz.md`](./06_scientific_latex_reporting_and_tikz.md)** | Redacción Científica y TikZ | Erradicación de sesgos de IA, diagramación TikZ nativa y justificación de test de Wilcoxon. |
| **[`07_scientific_benchmarking_and_metrics.md`](./07_scientific_benchmarking_and_metrics.md)** | Benchmarking Experimental | Protocolo de toma de datos, reproducibilidad y diseño factorial cruzado. |
| **[`07_sota_visuomotor_evaluation_and_uncertainty.md`](./07_sota_visuomotor_evaluation_and_uncertainty.md)** | Evaluación SOTA e Incertidumbre | Predicción Conforme (PlanCP al 95%), Diff-DAgger UQ, Verificadores Best-of-N y AdaBoost con tocones. |
| **[`08_3d_diffusion_policy_and_ml_rigor.md`](./08_3d_diffusion_policy_and_ml_rigor.md)** | Visión 3D (DP3) y Rigor ML | DP3Encoder sin BatchNorm, FPS a 512/1024 pts, cropping de mesa, partición estricta por episodios (Ley de Kolter). |
| **[`08_off_ros_logging_and_benchmarking.md`](./08_off_ros_logging_and_benchmarking.md)** | Logging Off-ROS y SQLite | Logger `coloredlogs`, benchmark nanosegundo GPU VRAM y base de datos relacional fail-safe. |
| **[`09_comparative_benchmarking_and_precision_metrics.md`](./09_comparative_benchmarking_and_precision_metrics.md)** | Benchmarking Pareado y Modelo TCP | Evaluación pareada Halton QMC (Solo DP vs DP+RL), modelo TCP $79.0\text{ mm}$ y panel gráfico 2x3. |
| **[`10_sim_to_real_finetuning_and_residual_policies.md`](./10_sim_to_real_finetuning_and_residual_policies.md)** | Transferencia Sim-to-Real | Few-Shot Fine-Tuning (Co-Fine-Tuning/LoRA) vs Residual Policy Learning (ResiP/TRANSIC) en control cartesiano. |
| **[`11_isaac_sim_headless_acceleration.md`](./11_isaac_sim_headless_acceleration.md)** | Aceleración Headless ($3\times\text{--}7\times$) | SimulationApp headless, reajuste en memoria a $256 \times 256$, rateLimit desanclado, orquestación en host y micro-daemon TCP inter-contenedor. |

---

## Reglas de Mantenimiento Aditivo

1. **Nunca sobreescribir ni degradar conocimiento histórico:** La información en esta carpeta se nutre de forma aditiva y acumulativa.
2. **Auditoría previa línea por línea:** Antes de modificar cualquier archivo de conocimiento, el agente debe leer los códigos fuente línea por línea para certificar que lo documentado refleja el estado funcional exacto del software.
3. **Revisión de Commits por Código:** Al analizar cambios recientes, el agente debe inspeccionar los diffs reales de código y no únicamente los mensajes descriptivos de los commits.
