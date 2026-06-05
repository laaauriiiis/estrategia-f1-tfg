# Calculadora de estrategias de neumáticos en Fórmula 1 mediante Inteligencia Artificial

Este repositorio contiene el código desarrollado para el Trabajo de Fin de Grado **"Calculadora de estrategias en Fórmula 1 mediante Inteligencia Artificial"**.

El objetivo del proyecto es recomendar estrategias iniciales de neumáticos en Fórmula 1 a partir de información disponible antes del inicio de una carrera. El problema se formula como una decisión discreta entre estrategias válidas de compuestos slicks (`SOFT`, `MEDIUM`, `HARD`) y se comparan dos enfoques:

- Un baseline de **aprendizaje supervisado**, que intenta imitar las estrategias observadas históricamente.
- Un enfoque **basado en valor**, relacionado con aprendizaje por refuerzo offline, que estima el valor de distintas estrategias mediante un simulador simplificado de tiempo de carrera.

## Estructura del proyecto

```text
estrategia_f1/
├── datasets/
│   ├── raw/
│   │   └── circuitos.csv
│   ├── processed/
│   │   ├── dataset_experimental.csv
│   │   ├── dataset_ML.csv
│   │   ├── dataset_RL.csv
│   │   └── dataset_simulador.csv
│   └── datos_memoria/
│       ├── evaluacion_ml_*.csv
│       ├── evaluacion_rl_*.csv
│       ├── evaluacion_simulador_*.csv
│       └── resumen_acciones_108.csv
│
├── estrategia_f1/
│   ├── data/
│   │   ├── dataset_builder.py
│   │   ├── dataset_features.py
│   │   ├── dataset_categorias.py
│   │   ├── dataset_utils.py
│   │   ├── filtro_outliers.py
│   │   └── openf1_client.py
│   │
│   ├── sim/
│   │   ├── simulador.py
│   │   └── evaluacion_sim.py
│   │
│   ├── ml/
│   │   ├── entrenamiento_ml.py
│   │   └── evaluacion_ml.py
│   │
│   ├── rl/
│   │   ├── entrenamiento_rl.py
│   │   ├── evaluacion_rl.py
│   │   └── evaluacion_rl_real.py
│   │
│   ├── acciones.py
│   ├── cache_utils.py
│   ├── config.py
│   └── features.py
│
├── scripts/
│   ├── script_crear_datasets.py
│   ├── script_evaluar_sim.py
│   ├── script_evaluar_ml.py
│   ├── script_evaluar_rl.py
│   ├── script_resumen_dataset.py
│   └── script_ranking_sensibilidad.py
│
└── runs/
    ├── ml/
    └── rl/
```
## Descripción general

El sistema trabaja con datos históricos de Fórmula 1 obtenidos principalmente mediante la API pública de OpenF1. A partir de estos datos se construye un dataset experimental formado por observaciones piloto-carrera.

Cada observación contiene:

- Información del circuito.
- Condiciones ambientales previas a la carrera.
- Variables históricas relacionadas con neumáticos.
- Estrategia real utilizada.
- Tiempo final de carrera observado.

El espacio de acciones está formado por 108 estrategias válidas, compuestas por secuencias de entre 2 y 4 stints utilizando los compuestos `SOFT`, `MEDIUM` y `HARD`.

## Requisitos

El proyecto está desarrollado en Python. Las principales librerías utilizadas son:

- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`
- `requests`
- `tqdm`

## Datos necesarios

El proyecto necesita el fichero auxiliar:

```text
datasets/raw/circuitos.csv
```

Este fichero contiene información estructural de los circuitos, como el identificador del circuito, el nombre y la longitud del trazado.

En el caso de que no se encontrase dicho fichero en el repositorio, el usuario deberá descargarlo aquí: https://drive.google.com/file/d/1W4NnwgFzeU3jwSMeMB0TddRah5FkRKDf/view?usp=drive_link y situarlo en la sigueinte ruta:

```text
datasets/raw/circuitos.csv
```

## Ejecución del proyecto

Todos los scripts principales se encuentran en la carpeta `scripts/`.

### 1. Crear el dataset experimental

```bash
python scripts/script_crear_datasets.py
```

Este script construye el dataset experimental a partir de los datos históricos y lo guarda en:

```text
datasets/processed/dataset_experimental.csv
```

### 2. Generar resumen del dataset

```bash
python scripts/script_resumen_dataset.py
```

Este script genera tablas descriptivas del dataset y de la distribución de acciones, utilizadas para la memoria.

### 3. Evaluar el simulador

```bash
python scripts/script_evaluar_sim.py
```

Este script valida empíricamente el simulador comparando los tiempos simulados con los tiempos reales de carrera.

Genera resultados en:

```text
datasets/datos_memoria/evaluacion_simulador_detalle.csv
datasets/datos_memoria/evaluacion_simulador_por_gp.csv
```

### 4. Entrenar y evaluar modelos supervisados

```bash
python scripts/script_evaluar_ml.py
```

Este script entrena y evalúa varios modelos supervisados:

- Regresión Logística
- Random Forest
- HistGradientBoosting
- MLP

Los modelos se evalúan tanto como clasificadores como políticas estratégicas mediante simulación.

### 5. Entrenar y evaluar modelos basados en valor

```bash
python scripts/script_evaluar_rl.py
```

Este script entrena modelos de regresión para aproximar una función de valor `Q(s,a)`, evaluando distintas estrategias candidatas mediante el simulador.

Los modelos utilizados son:

- Ridge
- Random Forest
- HistGradientBoosting
- MLP

### 6. Análisis de sensibilidad del ranking

```bash
python scripts/script_ranking_sensibilidad.py
```

Este script permite analizar cómo cambia el ranking de estrategias del simulador ante modificaciones en algunos parámetros.

## Resultados generados

Los resultados principales se guardan automáticamente en:

```text
datasets/datos_memoria/
```

Entre ellos:

```text
evaluacion_ml_detalle.csv
evaluacion_ml_modelos.csv
evaluacion_ml_resumen.csv
evaluacion_rl_detalle.csv
evaluacion_rl_modelos.csv
evaluacion_rl_resumen.csv
evaluacion_rl_real_detalle.csv
evaluacion_rl_real_resumen.csv
evaluacion_simulador_detalle.csv
evaluacion_simulador_por_gp.csv
resumen_acciones_108.csv
```

Los modelos entrenados y artefactos de ejecución se guardan en:

```text
runs/
```

## Reproducibilidad

La configuración general del proyecto se encuentra centralizada en:

```text
estrategia_f1/config.py
```

Ahí se definen:

- Compuestos disponibles.
- Número mínimo y máximo de stints.
- Número de acciones válidas.
- Parámetros del simulador.
- Columnas del dataset.
- Modelos utilizados.
- Semilla aleatoria.
- Rutas de entrada y salida.

La semilla utilizada en los experimentos es:

```python
SEED = 42
```

## Nota sobre el simulador

El simulador no pretende reproducir de forma exacta una carrera real de Fórmula 1. Su objetivo es proporcionar un entorno simplificado y reproducible para comparar estrategias bajo las mismas condiciones.

No se modelizan explícitamente factores como:

- Tráfico en pista.
- Diferencias entre pilotos o monoplazas.
- Safety Car o Virtual Safety Car dinámicos.
- Cambios meteorológicos durante la carrera.
- Errores en paradas.
- Adelantamientos.
- Decisiones estratégicas vuelta a vuelta.

Por tanto, las recomendaciones deben interpretarse como una herramienta de apoyo y comparación dentro del entorno simulado, no como una predicción exacta del resultado real de una carrera.

## Autoría

Proyecto desarrollado por **Laura Rodríguez López** como Trabajo de Fin de Grado del Grado en Ingeniería Informática.
