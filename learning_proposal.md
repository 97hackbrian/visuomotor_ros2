---
RequestFeedback: true
date: 2026-11-20
protocol: UPDATE_KNOWLEDGE.md v2.2.0 (Paso 6)
status: aprobada y aplicada (2026-11-20)
---

# Propuesta de aprendizaje: demostraciones reales para el xArm5

**Clasificación:** habilidad (`.agents/skills/realsense-vio-data-collection/SKILL.md`, v1.0 → v1.1).
El conocimiento ya está en `.agents/knowledge/16_xarm5_real_hardware_bringup.md` (§5m–5o) y las reglas en `AGENTS.md`
(punto 18). Esta propuesta solo toca la skill.

## Cambio propuesto

### 1. Nueva sección "Restricciones para demostraciones destinadas al xArm5 (5 GDL)"
Se insertaría antes de "## 4. Controles Durante la Recolección Interactiva".

**Restricciones:**
- **Inicio:** empezar cada episodio con la cámara en la pose equivalente a HOME3, con la misma disposición de la escena. Ese primer frame se mapea al home de `deploy/jetson/config/jetson_policy_params.yaml`.
- **Orientación:** garra vertical, o inclinada solo en el plano vertical que contiene a la base del brazo, hasta ±85°. No inclinar lateralmente (roll): el xArm5 no puede ejecutarlo.
- **Volumen:** radio de 0.15 a 0.65 m. La altura alcanzable baja con la distancia: 0.75 m cerca de la base, 0.61 m a 0.50 m y 0.37 m a 0.65 m.
- **Velocidad:** ≤ 5 cm/s y ≤ 0.5 rad/s (límites actuales del puente), o subir esos límites antes de grabar y reconvertir.
- **Montaje:** el mismo soporte de cámara en la mano y en la muñeca del robot.

### 2. Nuevo paso "Paso 4: Conversión al espacio del xArm5 (obligatoria antes de entrenar)"
```bash
python3 tools/xarm5/convert_vio_dataset_to_xarm5.py --src datasets/demonstrations_real_vN.zarr --report-only
python3 tools/xarm5/convert_vio_dataset_to_xarm5.py --src datasets/demonstrations_real_vN.zarr --drop-flagged
```
- Se entrena con `demonstrations_real_vN_xarm5.zarr`; el original no se modifica.
- Requiere `extrinsic_calibrated: true` en `visuomotor_bringup/config/xarm5/xarm5_dataset_alignment.yaml`.
- Revisar `_conversion_report.json`, que señala orientación inalcanzable, salidas del volumen o del alcance, y exceso de velocidad.

### 3. Criterio de aceptación adicional en la sección 2
"0 episodios marcados en el informe de conversión, o episodios marcados excluidos con `--drop-flagged`".

## Para aplicar
Responder **Proceed** (o indicar cambios). Sin aprobación la skill no se modifica.

---
**Estado (2026-11-20): APROBADA por el usuario y APLICADA** en `.agents/skills/realsense-vio-data-collection/SKILL.md`
(v1.1), con el HOME3 redefinido (TCP (0.150, 0.005, 0.750) m, garra inclinada ~−24.6°).
