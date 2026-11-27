---
name: update-knowledge
description: Guardar lo aprendido y el estado del trabajo en archivos del repo para que ningún agente dependa de su memoria de contexto. Primero el estado vivo (.agents/STATE.md); después context.md, AGENTS.md, .agents/knowledge/ y .agents/skills/. Usar con /update-knowledge o /learn, cuando el usuario dice "guarda todo", al cerrar un hito, antes de lanzar un proceso largo y siempre que el contexto esté por agotarse.
---

# Skill: Update Knowledge (v3.0, 2026-11-27)

**Objetivo:** que cualquier agente (Claude Code, Codex, Antigravity, Cursor…) retome el trabajo sin pérdidas aunque su
contexto se compacte o la sesión cambie. Lo que no está escrito en un archivo del repo se pierde: un resumen
automático del contexto omite comandos exactos, cifras y decisiones del usuario.

## Cuándo activarla

- **Pedido del usuario:** `/update-knowledge`, `/learn`, "guarda todo", "actualiza context y agents".
- **Al cerrar un hito:** un dataset armado o congelado, un entrenamiento lanzado o terminado, un ensayo en el robot,
  una decisión o corrección del usuario.
- **Antes de lanzar un proceso largo** (entrenamiento, recolección, DROID): primero el estado, después el proceso.
- **Cuando el contexto se acerca al límite:** conversación larga, muchas salidas de herramientas o un aviso de
  compactación. Hacer el paso 0 en ese momento, sin esperar al final de la tarea.

## Paso 0: estado vivo (≤ 5 min, siempre primero)

Reescribir `.agents/STATE.md`: como máximo ~150 líneas, reemplazando lo superado en vez de acumular historia. Debe
responder a esto:

1. **Qué corre ahora:**
   - el proceso y su carpeta;
   - cómo verlo (sesión de tmux);
   - cómo pararlo y reanudarlo;
   - la hora estimada de fin.
2. **Datasets y modelos:** el oficial de cada nivel (crudo, derivado y funcional, AGENTS §2.3) y para qué se usa cada uno.
3. **Pendientes en orden:** el próximo paso concreto de cada uno y quién lo aprueba.
4. **Reglas vigentes del usuario** que no están en AGENTS.md (correcciones y preferencias).
   - Copiar ahí lo durable de la memoria privada del agente, que los demás agentes no ven. En Claude Code está en
     `~/.claude/projects/<repo>/memory/`.
5. **Instrucciones del usuario aún sin cumplir**, casi textuales.

Después, en el host (nunca git en el contenedor):

```bash
git add .agents/STATE.md && git commit -m "docs(state): ..." && git push origin main
```

**Re-inyección automática (Claude Code):** el hook `SessionStart` de `.claude/settings.json` imprime
`.agents/STATE.md` en el contexto al iniciar, al reanudar, al limpiar y tras cada compactación. Si el archivo cambia
de ruta, actualizar el hook.

## Paso 1: auditar lo que cambió (el código, no los mensajes de commit)

- Leer `git log --stat` de la sesión y los diffs reales (`git show <commit>`), y las funciones tocadas.
- Correr en el contenedor las pruebas afectadas:
  ```bash
  docker exec -u 1000 -w /ros visuomotor_ros2 python3 -m pytest tests/<archivo> -q
  ```
- Si algo escrito antes resultó falso, corregirlo en todos los .md que lo repiten (buscarlo con `grep -rn`).

## Paso 2: conocimiento modular (`.agents/knowledge/`, aditivo)

- Sumar al módulo del tema, o crear `NN_tema.md` con el número siguiente. No borrar historia válida: lo superado se
  marca como superado, con la fecha.
- **Catálogo:**
  - 01–14: simulación, arquitectura de DP, RL, benchmarks, convergencia y ruido.
  - 15: RTAB-Map y RealSense.
  - 16: el xArm5 real.
  - 17: curación de v7 y acciones relativas.
  - 18: elección del estimador (DROID-SLAM).
  - 19: el pipeline de v8 a mano.
  - 20: el entrenamiento con v8.
- Cada cifra lleva su fuente (archivo JSON, CSV o log) y se da como distribución (mediana, p90 y máximo), nunca como
  el mejor caso.

## Paso 3: `context.md` y `AGENTS.md`

- **`context.md`:** una lección numerada por cada hallazgo, con síntoma, causa, arreglo y cómo evitarlo.
- **`AGENTS.md`:** solo reglas y estado estable (§2 reglas, §6 por fase, §7 comandos). Se mantiene conciso: el
  detalle va al módulo de conocimiento y el estado del momento a `STATE.md`.

## Paso 4: skills (`.agents/skills/`)

- Un procedimiento que funcionó y se va a repetir se convierte en una skill nueva o actualizada. Debe traer los
  comandos exactos, la checklist, los errores vistos con su forma de detectarlos, y las cifras de la última vez.
- **Skills actuales:**
  - `update-knowledge` (esta);
  - `v8-handheld-demo-collection` (demos a mano con la D435i, procedimiento de 2026-11-25/26);
  - `realsense-vio-data-collection`;
  - `droid-slam-offline-vio`;
  - `diffusion-policy-expert`;
  - `visuomotor-benchmark-evaluator`;
  - `scientific-benchmarking`;
  - `scientific-latex-reporting`;
  - `headless-experimentation-workflow`.

## Paso 5: memoria privada del agente (si la tiene)

- **Claude Code:** actualizar los archivos de `~/.claude/projects/<repo>/memory/` y su índice `MEMORY.md`.
- Todo lo que vaya ahí debe estar también en el repo: en `STATE.md` (paso 0), en AGENTS.md o en el conocimiento.

## Paso 6: cerrar

- Commit y push a `main`, por regla del usuario.
- Resumen al usuario en español: qué archivos cambiaron y qué quedó pendiente.

## Propuestas que el usuario no pidió (`/learn`)

Si el agente quiere cambiar reglas o skills sin que el usuario lo haya pedido, primero lo propone (en la conversación
o en `learning_proposal.md`) y lo aplica tras la aprobación. Cuando el usuario lo pide explícitamente, se aplica
directo.

## Errores vistos

- **Contexto compactado (2026-11-25 a 2026-11-27):** el de Claude Code se compactó al menos dos veces en sesiones
  largas de recolección y armado. El resumen conservó lo principal, pero no todos los comandos ni las cifras. Por eso
  el paso 0 va primero y el hook vuelve a cargar `STATE.md`.
- **Conclusiones corregidas después:** "los frames fuera del alcance son casi todos del comienzo" y "excluir la
  primera tanda" resultaron falsas al mirar los datos. Verificar con los datos antes de escribir una conclusión en
  los .md.
