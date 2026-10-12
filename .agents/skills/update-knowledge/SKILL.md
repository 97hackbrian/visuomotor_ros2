---
name: update-knowledge
description: Sincroniza y actualiza automáticamente context.md, AGENTS.md y las skills de IA del repositorio con los aprendizajes, soluciones de bugs y decisiones arquitectónicas de la sesión de trabajo.
---

# Skill: Update Knowledge

Esta habilidad automatiza la extracción de conocimientos adquiridos durante la sesión y los persiste en la documentación viva del proyecto.

## Cuándo activar esta habilidad
- Cuando el usuario diga: *"actualiza la base de conocimientos"*, *"actualiza context"*, *"guarda lo aprendido en esta sesión"*, o use `/update-knowledge`.

## Procedimiento
1. **Inspeccionar el historial de la conversación:** Identificar qué fallos se resolvieron (causa raíz y solución), qué decisiones de tensores/dimensiones se tomaron, y qué archivos de configuración cambiaron.
2. **Actualizar `context.md`:** Incorporar las lecciones aprendidas y actualizar el estado de los hitos del proyecto.
3. **Actualizar `AGENTS.md`:** Sincronizar el contrato operativo para que futuros agentes de cualquier plataforma sigan las reglas exactas.
4. **Actualizar skills especializadas (`~/.gemini/skills/`):** Si hubo avances en políticas de difusión, sincronizar `diffusion-policy-expert`.
5. **Confirmar al usuario:** Resumir los cambios puntuales aplicados.
