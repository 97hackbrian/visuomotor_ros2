---
RequestFeedback: true
date: 2026-11-22
protocol: UPDATE_KNOWLEDGE.md v2.2.0 (Paso 6)
status: aprobada y aplicada (2026-11-22)
---

# Propuesta de aprendizaje (Fase 22b): línea base cinemática y alcance real del xArm5

**Clasificación:** habilidades. El conocimiento ya está en los módulos 16 (§5s) y 17 (§9-10), en `AGENTS.md` (puntos 18
y 22, comandos 19-20) y en `context.md` (lección 42). Esta propuesta solo toca dos skills.

## 1. `.agents/skills/realsense-vio-data-collection/SKILL.md` (v1.2 → v1.3)

En §3b.3 "Volumen alcanzable", separar lo físico de lo configurado:
- Alcance del brazo (manual v2.7.1, XF1305): 700 mm de radio alrededor de la base a la altura del hombro (0.267 m). A
  la altura de aproximación de v7 (brida ~0.28 m, garra vertical), ±0.47 m laterales a la profundidad de los objetos.
- La caja del puente `|y| <= 0.30 m` es la caja de seguridad de la celda: respetarla o ampliarla antes de grabar,
  después de medir la celda. Con ±0.45 m ningún frame v7 quedaría fuera.

En §3c "Protocolo de grabación", agregar:
- **Desde HOME3, bajar antes de alejarse de la base.** Avanzar en alto saca la muñeca de alcance: 209 frames de
  5 episodios v7, hasta 51 mm.
- **Repartir los objetos a ambos lados y en varias profundidades,** con una grilla planificada. En v7 la media lateral
  quedó 10 cm a un lado, y desde el reposo ir siempre hacia la dirección media ya acertaba el 55 %.

En §2, agregar el criterio "0 frames fuera del alcance del puente", verificado con
`tools/xarm5/analyze_xarm5_workspace.py`.

## 2. `.agents/skills/diffusion-policy-expert/SKILL.md` (v1.2 → v1.3)

En §13.B "Selección de checkpoints":
- **Batir a "quedarse quieto" no basta.** Exigir además:
  - batir a la línea base "velocidad constante" (`cv_*` en `eval_metrics.csv`) al final del horizonte;
  - que el error suba claramente con imágenes cruzadas (`tools/diagnose_policy_baselines.py`).
- **Reportar el perfil de error por paso.** En v7 la política perdía con la velocidad constante en los pasos
  ejecutados (3.6 vs 1.3 mm) y ganaba desde ~0.53 s.

## Aplicación (2026-11-22)
Aprobada por el usuario ("Proceed") y aplicada: `realsense-vio-data-collection` v1.3 (§2 criterio de alcance, §3b.3
brazo vs. caja, §3c.6-7) y `diffusion-policy-expert` v1.3 (§13.B líneas base exigentes).
