---
RequestFeedback: true
date: 2026-11-22
protocol: UPDATE_KNOWLEDGE.md v2.2.0 (Paso 6)
status: aprobada y aplicada (2026-11-22)
---

# Propuesta de aprendizaje (Fase 22): datos reales VIO y entrenamiento para el xArm5

**Clasificación:** habilidades. El conocimiento ya está en `.agents/knowledge/17_real_vio_dataset_curation_and_relative_actions.md`
y las reglas en `AGENTS.md` (punto 22). Esta propuesta solo toca dos skills.

## 1. `.agents/skills/realsense-vio-data-collection/SKILL.md` (v1.1 → v1.2)

Nueva sección "Protocolo de grabación para el xArm5" (antes de "Controles Durante la Recolección Interactiva"):
- **Reiniciar la odometría al inicio de cada episodio** (o registrar la IMU): hoy `odom` persiste en toda la sesión y su
  vertical deriva hasta 60°; solo el movimiento relativo dentro del episodio es fiable.
- **Sostener la cámara derecha** (arriba de la imagen = lejos de la base) y empezar con la inclinación de HOME3 (~25° de
  la vertical); en v7 la media fue 32° ± 6°.
- **No girar la muñeca más de ±140°** (el yaw del puente está acotado a ±2.6 rad): los giros de ~180° se descartan.
- **Velocidad real:** ~0.05 m/s es compatible con el puente; la cifra "0.20 m/s" del ruido no es la velocidad real.

Nuevo criterio de aceptación en la sección 2 (además del informe de conversión):
- Auditoría física con `tools/xarm5/analyze_vio_dataset_quality.py`: desalineación con la gravedad < 20° en la mayoría
  de episodios con la extrínseca configurada, 0 saltos de signo del cuaternión, ruido eliminado < 10 mm RMS.

## 2. `.agents/skills/diffusion-policy-expert/SKILL.md` (v1.1 → v1.2)

Nueva sección "Entrenamiento con demostraciones reales a mano":
- Usar `--action_representation relative_ee` (UMI) con datasets VIO; `absolute` no batió la línea base en el piloto.
- Siempre `--val_ratio 0.1` (split por episodio) y seleccionar checkpoints por métricas de muestreo completo (ADE/FDE,
  F1 done, progreso) frente a la línea base "quedarse quieto", nunca por la pérdida de validación.
- EMA activa (defecto), `--color_jitter`, `--save_every 10`, `--resume` y `--max_rss_gb 12`.
- Recursos: un entrenamiento pesado a la vez en este equipo (31 GB RAM); dos como máximo con `--num_workers 6
  --prefetch_factor 2`.
- Antes de desplegar: `tools/xarm5/replay_policy_offline.py` (calibrar umbral y paciencia del auto-stop) y
  `tools/xarm5/check_policy_closed_loop_sim.py`.

## Aplicación (2026-11-22)
Aprobada por el usuario y aplicada en ambas skills (v1.2). Correcciones adicionales hechas al aplicarla, porque las
versiones v1.1 contradecían el código y los datos de la Fase 22:
- `realsense-vio-data-collection` §3b.2: el umbral de desviación de orientación ya no es 0.15 rad fijo, sino
  `max_orientation_deviation_rad` de `xarm5_dataset_alignment.yaml` (0.70 rad).
- `realsense-vio-data-collection` §5: `--dst` con sufijo nuevo cuando el destino ya existe y el comando de la auditoría física.
- `diffusion-policy-expert` §4.B: la regla "inviolable" `qw >= 0` tiene una excepción (semiesfera `home` en el xArm5 real).
- `diffusion-policy-expert` §13.B: el rango E* de §12.A (datasets simulados) no se transfiere a datos reales.
