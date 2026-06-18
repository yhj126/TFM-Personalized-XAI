# TFM Personalized XAI

Sistema experimental para generar explicaciones XAI personalizadas en clasificación
de imágenes. El proyecto combina visualizaciones explicables, una base de casos
con respuestas de usuarios, recuperación basada en casos (CBR) y un LLM local para
redactar explicaciones adaptadas al perfil de cada usuario.

Titulo: Generación de explicaciones personalizadas en clasificación de imágenes mediante LLMs y CBR
Autor: Haojie Yin  
Trabajo de Fin de Máster en Inteligencia Artificial  
Universidad Complutense de Madrid  
Curso 2025-2026

## Objetivo

El objetivo del proyecto es estudiar cómo recomendar y redactar explicaciones XAI
adaptadas a distintos perfiles de usuario. Para ello se parte de un conjunto de
imágenes de naturaleza/animales, se generan explicaciones visuales con varios
métodos XAI y se utiliza una base de casos construida a partir de un cuestionario
para decidir qué tipo de explicación conviene mostrar.

El sistema trabaja con dos capas principales:

- **Capa de problema**: imagen, descripción textual, clase predicha, dominio y
  perfil del usuario.
- **Capa de solución**: método XAI recomendado, formato, longitud, nivel técnico,
  componentes de la explicación y feedback observado en casos previos.

## Métodos XAI incluidos

Las opciones del cuestionario y del CBR se corresponden con estos métodos:

| Opción | Método | Descripción breve |
| --- | --- | --- |
| A | Anchor Image | Superpíxeles suficientes para mantener la predicción. |
| B | Grad-CAM | Mapa de calor sobre regiones relevantes de la red convolucional. |
| C | Integrated Gradients | Atribución por píxel respecto a una referencia. |
| D | LIME | Superpíxeles que apoyan localmente la clase predicha. |
| E | Saliency Maps | Sensibilidad del score ante cambios en los píxeles. |
| F | Ninguna | Sin explicación XAI visual. |

Todos los notebooks XAI usan ResNet50 preentrenado de `torchvision` como modelo
base y explican la clase top-1 predicha por el modelo.

## Estructura del proyecto

```text
.
├── base_de_casos/
│   ├── base_casos_doble_xai.ipynb
│   ├── notebook_similitud_casos_xai.ipynb
│   ├── respuestas_usuarios_anonimizadas.csv
│   ├── cbr_case_base_outputs/
│   └── cbr_similarity_outputs/
├── evaluacion_debug/
│   └── llm_cbr_pipeline.ipynb
├── evaluacion_offline/
│   ├── cbr_loo_eval.py
│   └── cbr_loo_validacion.ipynb
├── evaluacion_online/
│   ├── llm_cbr_pipeline.py
│   ├── llm_cbr_pipeline_demo_final_usuario.ipynb
│   ├── llm_cbr_pipeline_demo_final_3_imagenes.ipynb
│   └── resultados/
├── explicacion_XAI_imagenes/
│   ├── AnchorImage.ipynb
│   ├── GradCAM.ipynb
│   ├── IntegratedGradients.ipynb
│   ├── LIME.ipynb
│   └── SaliencyMaps.ipynb
├── generacion_descripcion_XAI/
│   ├── Descripciones_XAI_imagenes.ipynb
│   └── resultados_descripciones_xai/
├── imagenes/
│   └── original/
├── llm_zero_shot/
│   └── llm_cbr_zero_shot_pipeline_demo.ipynb
├── pyproject.toml
└── uv.lock
```

### Carpetas principales

- `imagenes/original/`: imágenes usadas como entrada. La base de casos principal
  trabaja con 11 imágenes de referencia.
- `explicacion_XAI_imagenes/`: notebooks para generar las visualizaciones XAI.
  Las carpetas `imagenes/output_*` son artefactos generados y están ignoradas por
  git.
- `generacion_descripcion_XAI/`: generación de descripciones textuales de las
  imágenes originales y de las visualizaciones XAI mediante Ollama.
- `base_de_casos/`: construcción de la base de casos a partir del cuestionario y
  experimentos de similitud CBR.
- `evaluacion_offline/`: validación Leave-One-Out del CBR.
- `evaluacion_debug/`: notebook de depuración para inspeccionar recuperación de
  vecinos, ranking CBR, prompt generado y variantes de explicación antes de la
  evaluación final.
- `evaluacion_online/`: pipeline CBR + LLM con feedback iterativo de usuario.
- `llm_zero_shot/`: condición comparativa sin recuperación CBR.

## Datos generados

La base de casos se construye a partir de respuestas anonimizadas del cuestionario.
El flujo principal genera:

| Archivo | Contenido |
| --- | --- |
| `user_profiles.csv` | Una fila por usuario. |
| `image_responses_long.csv` | Una fila por interacción usuario-imagen. |
| `case_base_double_partial.csv` | Perfil + respuesta por imagen. |
| `case_base_double_full.csv` | Base de casos completa con metadatos de imagen. |
| `image_metadata_template.csv` | Metadatos de las 11 imágenes. |

En el estado actual hay 27 usuarios y 11 imágenes, por lo que la base completa
contiene 297 casos usuario-imagen.

## Instalación

El proyecto requiere Python `>=3.12`. Se recomienda usar `uv`, porque el
repositorio incluye `uv.lock`, pero también puede instalarse con el flujo normal
de `venv` + `pip`.

### Opción A: instalación con `uv`

```bash
uv sync
```

Para abrir los notebooks:

```bash
uv run jupyter notebook
```

### Opción B: instalación sin `uv`

Crear y activar un entorno virtual:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Instalar las dependencias declaradas en `pyproject.toml`:

```bash
python -m pip install -e .
```

Para abrir los notebooks:

```bash
python -m notebook
```

Algunos pasos descargan pesos de `torchvision` y pueden tardar la primera vez.
Para las partes con LLM/visión local se usa Ollama, por defecto con
`qwen2.5vl:32b`.

```bash
ollama pull qwen2.5vl:32b
```

## Flujo reproducible

### 1. Generar explicaciones visuales XAI

Ejecutar los notebooks de `explicacion_XAI_imagenes/`:

- `AnchorImage.ipynb`
- `GradCAM.ipynb`
- `IntegratedGradients.ipynb`
- `LIME.ipynb`
- `SaliencyMaps.ipynb`

Cada notebook lee desde `imagenes/original/` y escribe en su carpeta
`imagenes/output_*` correspondiente.

### 2. Generar descripciones textuales

Ejecutar:

```text
generacion_descripcion_XAI/Descripciones_XAI_imagenes.ipynb
```

Salidas principales:

- `descripciones_por_imagen_xai.csv`
- `descripciones_por_imagen_xai.xlsx`
- `resumenes_por_caso_xai.csv`

Estas descripciones se usan después para construir prompts más informativos para
el LLM.

### 3. Construir la base de casos

Ejecutar:

```text
base_de_casos/base_casos_doble_xai.ipynb
```

El notebook transforma el cuestionario en perfiles, respuestas por imagen y base
de casos CBR.

### 4. Evaluar el CBR offline

El script principal de evaluación es:

```bash
uv run python evaluacion_offline/cbr_loo_eval.py --k 11
```


La validación separa:

- **Campos de problema**: texto de la imagen, perfil del usuario y metadatos de
  imagen.
- **Campos de solución**: opción elegida, formato, longitud, nivel técnico,
  impacto percibido, tipos de explicación preferidos y objetivos XAI.

La similitud se calcula por bloques:

| Bloque | Técnica | Peso |
| --- | --- | --- |
| `text` | TF-IDF + coseno sobre `problem_text` | 0.40 |
| `profile` | numéricas + categóricas del perfil | 0.40 |
| `image` | SSIM precalculado por `image_id` o metadatos | 0.20 |

La predicción de campos de solución se hace por voto ponderado según similitud.

### 5. Ejecutar el pipeline online CBR + LLM

Para generar el prompt sin llamar al LLM:

```bash
uv run python evaluacion_online/llm_cbr_pipeline.py \
  --dry-run \
  --description-case-id image03 \
  --output-dir /tmp/tfm_personalized_xai_dry_run \
  --query-json '{"image_id":3,"age_range":"25-34","education_level":"Master","occupation_raw":"Investigador/a (academico)","ai_knowledge_level":4,"domain_knowledge_level":3}'
```

Para generar una explicación real con Ollama:

```bash
uv run python evaluacion_online/llm_cbr_pipeline.py \
  --interactive \
  --model qwen2.5vl:32b \
  --description-case-id image03
```

El pipeline online:

1. Carga la query del usuario.
2. Enriquece la query con metadatos de imagen si están disponibles.
3. Recupera vecinos similares de la base de casos.
4. Recomienda una opción XAI combinando similitud y utilidad observada.
5. Recupera la descripción textual del método XAI recomendado.
6. Construye un prompt personalizado.
7. Genera la explicación con Ollama o devuelve el prompt en `--dry-run`.
8. Registra feedback en `evaluacion_online/resultados/llm_cbr_feedback_log.jsonl`.

Nota: incluso `--dry-run` guarda una línea de log. Para pruebas, conviene usar
`--output-dir` apuntando a una carpeta temporal.

### 6. Depurar prompts y flujo CBR + LLM

Antes de usar la interfaz final con usuarios, el notebook de depuración permite
ver el flujo completo con más detalle:

```text
evaluacion_debug/llm_cbr_pipeline.ipynb
```

Este notebook está pensado para revisar y ajustar el comportamiento del sistema:

- muestra el perfil de usuario usado como consulta;
- recupera los vecinos CBR y el ranking de soluciones recomendadas;
- enseña la imagen original y la visualización XAI seleccionada;
- recupera la descripción XAI asociada al método recomendado;
- construye y muestra el prompt completo enviado al LLM;
- permite ejecutar el flujo con `RUN_OLLAMA = False` para probar sin depender del
  modelo local;
- simula feedback positivo o negativo del usuario;
- prueba la regeneración de la explicación cuando el usuario no queda convencido.

Esta parte es útil para detectar problemas de depuración de prompts, información
faltante, recuperación de vecinos poco adecuada o explicaciones demasiado largas,
técnicas o genéricas.

### 7. Interfaz interactiva para evaluación online

La evaluación online con usuarios se realiza desde notebooks con interfaz
interactiva basada en widgets:

```text
evaluacion_online/llm_cbr_pipeline_demo_final_usuario.ipynb
evaluacion_online/llm_cbr_pipeline_demo_final_3_imagenes.ipynb
```

Estas versiones están orientadas al experimento final. En lugar de mostrar todos
los detalles internos del CBR, presentan una interfaz más limpia para el usuario:

- selección de imagen;
- introducción o selección del perfil del usuario;
- generación de la explicación final;
- visualización de la imagen y de la explicación XAI recomendada;
- recogida de feedback sobre si la explicación convence;
- registro de la interacción en los logs de `evaluacion_online/resultados/`.

El notebook `llm_cbr_pipeline_demo_final_3_imagenes.ipynb` limita la prueba a las
tres imágenes seleccionadas para la evaluación final, mientras que
`llm_cbr_pipeline_demo_final_usuario.ipynb` mantiene una versión más general de
la interfaz.

## Resultados offline actuales

Los resultados guardados en `base_de_casos/cbr_similarity_outputs/` corresponden
a una validación Leave-One-Out por imagen con `k=11`, excluyendo casos del mismo
usuario y usando SSIM como similitud visual.

Resumen actual:

| Métrica | CBR | Baseline |
| --- | ---: | ---: |
| Accuracy media | 0.3771 | 0.4300 |
| F1-macro medio | 0.1573 | 0.1530 |

Lectura de estos resultados:

- La baseline de clase mayoritaria sigue siendo fuerte para varios campos
  categóricos.
- El CBR no supera la accuracy media global en esta configuración.
- En campos multivalor, la coincidencia parcial mejora frente a baseline:
  `preferred_explanation_types_raw` tiene Jaccard 0.4650 frente a 0.4389, y
  `main_goals_raw` 0.2572 frente a 0.1918.
- El análisis de sensibilidad muestra que `k=11` mejora la accuracy media frente
  a `k=7` manteniendo un F1-macro medio similar.

## Zero-shot

El notebook:

```text
llm_zero_shot/llm_cbr_zero_shot_pipeline_demo.ipynb
```

implementa una condición comparativa sin CBR. En este modo el LLM recibe la
imagen, la descripción XAI seleccionada y el perfil básico del usuario, pero no
recibe vecinos recuperados, rankings, preferencias inferidas ni ejemplos
similares.


## Licencia

Este proyecto se distribuye bajo la licencia Creative Commons Attribution 4.0
International (CC BY 4.0).

Puedes compartir, copiar, redistribuir, adaptar y reutilizar el material del
proyecto, siempre que se cite adecuadamente al autor.

Las imágenes, datasets, modelos o recursos externos utilizados mantienen sus
licencias originales.
