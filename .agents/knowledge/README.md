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

---

## Reglas de Mantenimiento Aditivo

1. **Nunca sobreescribir ni degradar conocimiento histórico:** La información en esta carpeta se nutre de forma aditiva y acumulativa.
2. **Auditoría previa línea por línea:** Antes de modificar cualquier archivo de conocimiento, el agente debe leer los códigos fuente línea por línea para certificar que lo documentado refleja el estado funcional exacto del software.
3. **Revisión de Commits por Código:** Al analizar cambios recientes, el agente debe inspeccionar los diffs reales de código y no únicamente los mensajes descriptivos de los commits.
