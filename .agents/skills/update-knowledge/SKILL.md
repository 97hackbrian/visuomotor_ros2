---
name: update-knowledge
description: Sincroniza y actualiza automáticamente context.md, AGENTS.md, la base de memoria infinita (.agents/knowledge/) y las skills de IA, auditando el código línea por línea y revisando los diffs de commits de forma acumulativa y aditiva.
---

# Skill: Update Knowledge (v2.0)

Esta habilidad automatiza la extracción y persistencia de conocimientos técnicos en la base de datos de memoria infinita del proyecto.

## Cuándo activar esta habilidad
- Cuando el usuario invoque `/update-knowledge`, use `/learn`, o solicite actualizar o persistir la base de conocimientos.

## Procedimiento de Ejecución Obligatorio

1. **Auditoría Previa Línea por Línea del Código:**
   - Antes de escribir o actualizar documentación, lee los programas fuente (`policy_node.py`, `pickup_approximate.py`, `zarr_dataset.py`, `train.py`, YAMLs) línea por línea para entender a fondo la implementación real y certificar que tal como está el código está bien.
2. **Revisión de Commits por Código (Diffs Reales):**
   - Inspecciona los últimos commits analizando los cambios reales en el código (diffs) y no únicamente los mensajes del commit, para entender qué mejoras exactas se introdujeron.
3. **Control Aditivo de `.agents/knowledge/`:**
   - Nutre de forma acumulativa los módulos temáticos en `.agents/knowledge/` (`01_diffusion_policy_architecture.md`, `02_tensors_and_autostop.md`, `03_data_collection_and_fsm.md`, `04_dataloaders_and_datasets.md`, `05_troubleshooting_and_environment.md`) o crea nuevos archivos temáticos.
   - Jamás destruyas ni sobrescribas conocimiento histórico válido; el control debe ser aditivo.
4. **Sincronización de `context.md` y `AGENTS.md`:**
   - Actualiza las lecciones aprendidas, roadmap y reglas operativas (ejecución Docker bajo `-u hackbrian`, Git en host local).
5. **Actualización de Skills Especializadas (`~/.gemini/skills/`):**
   - Sincroniza `diffusion-policy-expert` o skills de deep learning pertinentes.
6. **Confirmación al Usuario:**
   - Presenta un resumen ejecutivo de los códigos auditados y los módulos de conocimiento actualizados.
