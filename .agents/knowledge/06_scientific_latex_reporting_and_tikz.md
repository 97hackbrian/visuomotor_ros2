# Módulo 06: Ingeniería Editorial Científica en LaTeX, Diagramación TikZ y Formulación Estadística en Robótica

Este documento consolida la base de conocimientos permanente para la redacción académica de alto impacto, la corrección de sesgos estilísticos de IA, la diagramación vectorial precisa en TikZ y la fundamentación estadística no paramétrica para robótica manipuladora y aprendizaje visuomotor, auditado a partir del desarrollo de `informe_avance_visuomotor_cd.tex`.

---

## 1. Erradicación de Sesgos Estilísticos de IA y Metalenguaje Académico

### A. Diagnóstico de Patrones de Texto Generados por IA
Al auditar borradores técnicos generados o modificados por modelos de lenguaje, se identificaron cuatro vicios recurrentes que comprometen la sobriedad académica:
1. **Metalenguaje de Rúbrica o Asignatura:** Inserción de frases que delatan la respuesta a una pauta de evaluación escolar (*"en concordancia con los requisitos de la rúbrica"*, *"lo que responde la evidencia"*, *"lo que queda por completar para la entrega final"*, inclusión de puntajes en títulos como `[30 PUNTOS]`).
2. **Hipérboles Grandilocuentes:** Uso indiscriminado de adjetivos épicos (*"paradigma de frontera"*, *"erradicar de raíz"*, *"integridad matemática absoluta"*, *"fuga causal catastrófica"*).
3. **Anclaje Bilingüe Mecánico:** Patrón repetitivo de traducir sustantivos técnicos al inglés entre paréntesis (*"amortiguamiento (damping)"*, *"recorte de acciones (action chunking)"*).
4. **Conectores Formulaicos al Inicio de Párrafo:** Repetición monótona de *"En primer término"*, *"En segundo lugar"*, *"Cabe destacar que"*, *"Como consecuencia directa de"*.

### B. Reglas de Corrección Editorial Humana
* **Traducción a Lenguaje Sobrio:** Reemplazar afirmaciones categóricas por descripciones funcionales y cuantitativas. En lugar de *"erradicar el colapso catastrófico"*, redactar *"corregir la inestabilidad en el entrenamiento mitigando el sobreajuste causal"*.
* **Uso Canónico de Términos en Inglés:** Los vocablos estandarizados por la comunidad internacional de robótica y aprendizaje automático se incorporan directamente en cursiva técnica (*\textit{dataloader}*, *\textit{rollout buffer}*, *\textit{clipping}*, *\textit{outliers}*, *\textit{flat minima}*), eliminando el paréntesis redundante.
* **Variabilidad Sintáctica (*Burstiness*):** Mezclar intencionalmente frases breves declarativas con oraciones compuestas subordinadas que expongan la causalidad físico-matemática.

---

## 2. Ingeniería de Diagramas TikZ Nativos de Precisión (Sin Sobrelapamientos)

### A. Coherencia Aritmética y Dimensional
En esquemas que ilustran flujos de tensores de redes neuronales, cualquier inconsistencia numérica resta credibilidad al manuscrito:
* **Caso Crítico Vector FiLM (Figura 2):** El nodo de condicionamiento global concatenaba las características visuales espaciales ($K=64$ keypoints latentes de dimensión 2D por fotograma, $64\text{D}$) y la propiocepción del brazo ($8\text{D}$ por fotograma). Con un horizonte de observación $T_{\text{obs}} = 2$:
  $$\mathbf{cond} = [v_t, v_{t-1}, s_t, s_{t-1}] \in \mathbb{R}^{2 \times 64 + 2 \times 8} = \mathbb{R}^{128 + 16} = \mathbb{R}^{144}$$
  La formulación anterior expresaba erróneamente `64D + 16D = 144D`. Se corrigió a `$2 \times 64\text{D} + 16\text{D} = 144\text{D}$`, preservando la consistencia dimensional exacta con el código fuente en `diffusion_policy.py`.

### B. Cuadrícula de Niveles y Envoltura de Texto (`text width`)
* **Cuadrícula Fija de Coordenadas:** Estructurar el canvas con coordenadas cartesianas explícitas para filas y columnas ($Y = 3.6, 1.0, -1.6$ y $X = 0.0, 6.2, 12.4$). Esto asegura flechas $100\%$ horizontales y perpendiculares sin desviaciones angulares indeseadas.
* **Control de Ancho y Saltos de Línea:** Asignar siempre `text width` acotado ($3.2$ a $3.6\text{ cm}$) a cada nodo con `align=center` para que los textos extensos se dividan en renglones balanceados mediante `\\`, previniendo que los bloques invadan el espacio de los nodos adyacentes.

### C. Resolución Geométrica de Colisiones en Aristas
* **Caso de la Flecha Vertical (Figura 6):** Al conectar dos bloques en TikZ mediante flechas verticales continuas, una etiqueta con texto centrado (`midway`) colisiona directamente con el trazo de la línea.
* **Solución Técnica:** Emplear posicionamientos laterales relativos explícitos:
  ```latex
  \path [line] (step3) -- node [midway, right=4pt, font=\scriptsize] {Telemetría} (step4);
  ```
* **Conexiones Ortogonales (`|-` y `-|`):** Ubicar las etiquetas en los tramos rectos mediante `pos=0.35` o `pos=0.65`, evitando situarlas en el vértice de giro de 90 grados.

### D. Enrutamiento Perimetral de Lazos de Retroalimentación (*Closed-Loop*)
* Para evitar que las flechas de retorno atraviesen bloques interiores, implementar rutas perimetrales exteriores:
  ```latex
  \path [line] (pid.west) -- ++(-0.8,0) 
        |- ([yshift=0.9cm]halton.north) 
        -- ([yshift=0.9cm]sim.north) 
        node [midway, above=2pt, font=\scriptsize] {Comandos de Velocidad (100 Hz)} 
        -| (sim.north);
  ```

### E. Conexión Multipuerto en Bloques Centralizadores
* En bloques colectores (ej. datasets Zarr o búfers de trayectorias), distribuir las llegadas de datos a través de distintas caras cardinales (`.west`, `.north`, `.east`), evitando que múltiples flechas converjan en el mismo punto geométrico.

### F. Nomenclatura Editorial y Control de Flotantes
* En el cuerpo del texto y epígrafes, nunca aludir al mecanismo de software (*"diagrama programado en código TikZ nativo"*); denominarlo simplemente *"diagrama"*, *"esquema"* o *"figura"*.
* Para diagramas anchos de pipeline metodológico que preceden secciones con listas densas, intercalar `\clearpage` antes del entorno `\begin{figure}[htbp]`. Esto previene que el motor tipográfico de LaTeX fragmente los ítems en páginas contiguas y asegura un empaquetado armónico donde la figura y la subsección siguiente coexisten en la misma página.

---

## 3. Reconciliación Empírica de Curvas de Aprendizaje por Refuerzo (RL)

### A. Auditoría Forense de Telemetría vs. Texto
Al contrastar las gráficas de `rl_training_curves.png` y los registros tabulares de `rl_training_metrics.csv` con el borrador del informe, se identificaron tres discrepancias fácticas:
1. **Divergencia KL en Difusión (DDIM):**
   - *Afirmación Errónea:* El texto sostenía que la divergencia aproximada de Kullback-Leibler se mantuvo estrictamente bajo $0{,}05$ con promedio de $0{,}024$.
   - *Realidad Empírica:* Debido a que la política de difusión ejecuta una cadena de 5 pasos estocásticos de eliminación de ruido (*denoising*), la razón de verosimilitud experimenta excursiones reales entre $0{,}10$ y $2{,}38$.
   - *Explicación Científica Rigurosa:* La estabilidad del entrenamiento no proviene de una KL plana, sino del mecanismo de recorte (*clipping*) de la razón PPO ($\epsilon = 0{,}2$) y del gradiente de imitación regularizador ($\lambda_{\text{BC}}$), los cuales absorben estas excursiones sin degradar el rendimiento cinemático.
2. **Retorno de Lote vs. Retornos Episódicos Individuales:**
   - En la gráfica de evolución temporal, la media por lote (*batch mean reward*) evoluciona desde $\approx 300$ hasta estabilizarse en $\approx 350-500$, con un pico transitorio de $775{,}60$ en la iteración 11.
   - Los valores superiores a $1600$ corresponden a episodios individuales ideales donde la pinza agarró el cubo en el primer intento y lo sostuvo en Home por el tiempo completo. Se debe diferenciar siempre la media estadística de los valores máximos aislados.
3. **Dinámica de la Pérdida del Crítico $V(s)$:**
   - La curva inicia en $81{,}73$, presenta un pico de ajuste transitorio en la iteración 11 ($\approx 464{,}68$) al acoplarse con la actualización de la política, y decae progresivamente hasta un régimen estable de $20$ a $40$ hacia la iteración 100 (no en $90{,}0$).

---

## 4. Fundamentación Estadística No Paramétrica para Robótica Manipuladora

### A. Invalidez de Supuestos Gaussianos en Manipulación
En la evaluación comparativa pareada (Solo DP vs. DP + RL):
* La proximidad física en el contacto ($d_{\text{contacto}}$) exhibe una masa concentrada en $0{,}88 - 7{,}14\text{ mm}$ para ensayos exitosos.
* Los ensayos fallidos o truncados por límite de tiempo generan valores atípicos (*outliers*) superiores a $100\text{ mm}$.
* Esto provoca distribuciones marcadamente asimétricas con colas pesadas, violando el supuesto de normalidad requerido por el test $t$ de Student pareado.

### B. La Prueba de Rangos con Signo de Wilcoxon
* **Definición:** Prueba de hipótesis no paramétrica para contrastar dos muestras emparejadas.
* **Hipótesis Nula ($H_0$):** La mediana de las diferencias pareadas ($D_i = X_{\text{DP+RL}, i} - X_{\text{SoloDP}, i}$) es igual a cero.
* **Mecanismo:** Ordena en rango los valores absolutos de las diferencias no nulas $|D_i|$ y asigna el signo original a cada rango, computando el estadístico $W = \min(W^+, W^-)$.
* **Ventaja Metodológica:** Al operar sobre rangos relativos en lugar de magnitudes continuas, es inmune a la distorsión inducida por fallos esporádicos extremos, permitiendo certificar si la mejora en precisión milimétrica inducida por RL es estadísticamente significativa ($p < 0{,}05$).

---

## 5. Alineación Bibliográfica: Difusión Discreta (DPPO) vs. Flujo Continuo (Flow Matching)

Al formular el aprendizaje por refuerzo sobre políticas generativas:
1. **DPPO sobre Cadenas DDIM:**
   - La política difusiva opera como un Proceso de Decisión de Markov de Eliminación de Ruido (*Denoising MDP*), optimizado a través de los pasos discretos de reversión temporal.
   - **Referencia Fundacional:** A. Z. Ren, J. Clark, A. Majumdar y B. Eysenbach, *"Diffusion Policy Policy Optimization"*, en *International Conference on Learning Representations (ICLR)*, 2025 (arXiv:2409.00588).
2. **Extensiones a Modelos de Flujo Continuo (Flow Matching):**
   - Métodos contemporáneos como *ReinFlow* (Zhang et al., NeurIPS 2025) y *Flow-GRPO* (Liu et al., NeurIPS 2025) extienden la optimización por refuerzo a campos vectoriales continuos mediante *Conditional Flow Matching* (CFM), constituyendo una línea ortogonal complementaria.

---

## 6. Control Editorial de Paginación y Cero Overfull Hbox

1. **Ruptura de Enlaces y Rutas Largas:**
   - Los nombres de archivos en bloque monoespaciado (`\texttt{}`) no poseen puntos de corte automático. Aplicar `\allowbreak` en guiones bajos:
     ```latex
     \texttt{benchmark\_}\allowbreak\texttt{comparison\_results.csv}
     ```
2. **Aislamiento de Bibliografía en Página Limpia:**
   - Una publicación rigurosa no debe permitir que el encabezado de referencias bibliográficas comparta página con párrafos residuales de trabajo futuro si esto desborda el límite de páginas.
   - Interpolar `\clearpage` antes de `\printbibliography` para que las referencias comiencen en la línea 1 de la página final asignada.
