---
name: update-knowledge
description: Sincroniza y actualiza automáticamente context.md, AGENTS.md, la memoria modular (.agents/knowledge/) y las skills (.agents/skills/) mediante el protocolo formal /learn, auditando el código fuente y herramientas modernas línea por línea.
---

# Skill: Update Knowledge (v2.2)

Esta habilidad automatiza la extracción y persistencia de conocimientos técnicos en la base de datos de memoria infinita del proyecto.

## Cuándo activar esta habilidad
- Cuando el usuario invoque `/update-knowledge`, use `/learn`, o solicite actualizar o persistir la base de conocimientos.
- Tras culminar hitos experimentales, calibraciones cinemáticas o la adición de nuevas herramientas de software.

## Procedimiento de Ejecución Obligatorio

1. **Auditoría Previa Línea por Línea del Código:**
   - Audita exhaustivamente tanto los programas fuente (`policy_node.py`, `pickup_approximate.py`, `zarr_dataset.py`, `train.py`, YAMLs) como las herramientas del sistema (`tools/sim/run_headless_system.sh`, `tools/verify_dataset_frequency.py`, `tools/visualize_dataset.py`, `tools/compare_models.py`, `tools/replot_benchmark.py`, `tools/experiment_tracker.py`, `tools/analyze_convergence.py`).
2. **Revisión de Commits por Código (Diffs Reales):**
   - Inspecciona los últimos commits analizando los cambios reales en el código (diffs) y no únicamente los mensajes del commit, para entender qué mejoras exactas se introdujeron.
3. **Control Aditivo de `.agents/knowledge/`:**
   - Nutre de forma acumulativa los módulos temáticos actuales o crea nuevos archivos temáticos:
     * `01_diffusion_policy_architecture.md`, `02_tensors_and_autostop.md`, `03_data_collection_and_fsm.md`
     * `04_dataloaders_and_datasets.md`, `05_troubleshooting_and_environment.md`
     * `06_reinforcement_learning_dppo.md`, `06_scientific_latex_reporting_and_tikz.md`, `06_telemetry_metrics_and_rl_evaluation.md`
     * `07_scientific_benchmarking_and_metrics.md`, `07_sota_visuomotor_evaluation_and_uncertainty.md`
     * `08_3d_diffusion_policy_and_ml_rigor.md`, `08_off_ros_logging_and_benchmarking.md`
     * `09_comparative_benchmarking_and_precision_metrics.md`
     * `10_sim_to_real_finetuning_and_residual_policies.md`, `11_isaac_sim_headless_acceleration.md`
     * `12_training_convergence_and_stopping_criteria.md`, `13_multicheckpoint_evaluation_and_pareto_convergence.md`
     * `14_historical_convergence_and_noise_floor.md`
   - Jamás destruyas ni sobrescribas conocimiento histórico válido; el control debe ser aditivo.
4. **Sincronización de `context.md` y `AGENTS.md`:**
   - Actualiza las lecciones aprendidas, roadmap y reglas operativas (ejecución Docker bajo `-u hackbrian`, Git en host local).
5. **Actualización de Skills Especializadas (`.agents/skills/`):**
   - Sincroniza las skills del repositorio en `.agents/skills/`: `scientific-benchmarking`, `visuomotor-benchmark-evaluator`, `headless-experimentation-workflow`, `diffusion-policy-expert`, `scientific-latex-reporting`, `update-knowledge`.
6. **Protocolo Interactivo de Aprendizaje (`/learn`):**
   - Toda propuesta de modificación de habilidades o reglas operativas debe plasmarse en un artefacto `learning_proposal.md` con metadatos interactivos `RequestFeedback: true`.
   - Clasificar rigurosamente: Reglas (`AGENTS.md`) vs Habilidades (`.agents/skills/`) vs Conocimiento (`.agents/knowledge/`).
   - Aplicar los cambios únicamente tras la aprobación explícita del usuario.
7. **Confirmación al Usuario:**
   - Presenta un resumen ejecutivo de los códigos auditados y los módulos de conocimiento actualizados.
