"""
config.py

Definición centralizada de constantes, parámetros y rutas utilizadas
en el sistema.

Este módulo no contiene lógica.

Centralizar esta configuración garantiza consistencia, reproducibilidad
y facilidad de mantenimiento a lo largo del pipeline experimental.
"""

# IMPORTS
from pathlib import Path
from typing import Any

# PARÁMETROS DEL DOMINIO DE LA FÓRMULA 1 -------------------------------------------------------------------------------
COMPUESTOS = ["SOFT", "MEDIUM", "HARD"]

# Diccionario String -> Número
# e.g. i = COMPUESTOS_A_INDICE["MEDIUM"]
COMPUESTOS_A_INDICE = {c: i for i, c in enumerate(COMPUESTOS)}

# Diccionario inverso Número -> String
# e.g. c = INDICE_A_COMPUESTOS[2]
INDICE_A_COMPUESTOS = {i: c for c, i in COMPUESTOS_A_INDICE.items()}

NUM_COMPUESTOS = len(COMPUESTOS)

# Mínimo 1 parada (>=2 stints) y máximo 3 paradas (<=4 stints)
MIN_STINTS = 2
MAX_STINTS = 4

# Parámetros generales de stints (simulador / degradación)
DEG_MIN = 0.01
DEG_MAX = 2.0
TEMP_REF = 30.0
TEMP_SLOPE = 0.003
TEMP_CLIP = (0.94, 1.08)

# Nota: asumimos categorías en minúscula ("baja","media","alta")
WEAR_MAP = {"baja": 1.0, "media": 1.10, "alta": 1.20}
TEMP_MAP = {"baja": 0.95, "media": 1.0, "alta": 1.05}
RAIN_THRESHOLDS = {"baja": 0.10, "media": 0.30}
WEAR_THRESHOLDS = {"baja": 0.04, "media": 0.08,}

CLAVES_NEUMATICOS = [
    "life_soft", "life_medium", "life_hard",
    "pace_soft", "pace_medium", "pace_hard",
    "deg_soft", "deg_medium", "deg_hard",
]

# ESPACIO DE ACCIONES --------------------------------------------------------------------------------------------------
N_ACCIONES = 108
ENCODING_ACCIONES = "sequence"

# PARÁMETROS DEL SIMULADOR ---------------------------------------------------------------------------------------------
DEFAULT_PIT_LOSS = 22.0
PENALIZACION_STINT = 15.0
PENALIZACION_VIDA_UTIL = 4.0
EXTRA_PARADA_MULTIPLE = 2.0

BASELINE_PRIORIDAD = [
    ["MEDIUM", "HARD"],
    ["SOFT", "MEDIUM"],
    ["MEDIUM", "SOFT"],
    ["SOFT", "HARD"],
    ["HARD", "MEDIUM"],
]

# FEATURES DEL ESTADO / DATASET ----------------------------------------------------------------------------------------
ID_COLS = ["season", "race_id", "race_date", "circuit_key"]

ESTADO_COLS = [
    "track_length_km", "n_laps", "wear_index", "pit_loss_s",
    "track_temp_cat", "weather_condition", "rain_prob_cat", "sc_prob",
    "life_soft", "life_medium", "life_hard",
    "pace_soft", "pace_medium", "pace_hard",
    "deg_soft", "deg_medium", "deg_hard",
]

ACCION_COLS = ["action_id", "strategy_compounds", "n_stints"]
TIEMPO_COL = ["finish_time_s"]
FILTER_COLS = ["dnf", "dns", "dsq"]

# PARÁMETROS DE ENTRENAMIENTO ------------------------------------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.2

# Para el TopK acciones por estado en evaluación y RL
K_ACCIONES_MUESTREO = 30

TOPK = (3, 5)
MODELOS_RL: dict[str, dict[str, Any]] = {
    "hist_gb": {
        "learning_rate": 0.06,
        "max_iter": 300,
    },
    "random_forest": {
        "n_estimators": 400,
        "max_depth": None,
        "n_jobs": -1,
    },
    "ridge": {
        "alpha": 1.0
    },
    "mlp": {
        "hidden_layer_sizes": (256, 128),
        "alpha": 1e-4,
        "learning_rate_init": 1e-3,
        "max_iter": 300,
    },
}

MODELOS_ML: dict[str, dict[str, Any]] = {
    "hist_gb": {
        "learning_rate": 0.06,
        "max_iter": 300,
    },
    "random_forest": {
        "n_estimators": 400,
        "max_depth": None,
        "n_jobs": -1,
    },
    "logreg": {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 500,
    },
    "mlp": {
        "hidden_layer_sizes": (256, 128),
        "alpha": 1e-4,
        "learning_rate_init": 1e-3,
        "max_iter": 300,
    },
}

# ESTRUCTURA DEL PROYECTO ----------------------------------------------------------------------------------------------
BASE_API = "https://api.openf1.org/v1"

# Raíz
BASE_DIR = Path(__file__).resolve().parent.parent

# Carpetas comunes
DATASETS_RAW_DIR = BASE_DIR / "datasets" / "raw"
DATASETS_PROCESSED_DIR = BASE_DIR / "datasets" / "processed"
SCRIPTS_DIR = BASE_DIR / "scripts"
SRC_DIR = BASE_DIR / "estrategia_f1"

# Datasets
CIRCUITOS_CSV = DATASETS_RAW_DIR / "circuitos.csv"
DATASET_ML_CSV = DATASETS_PROCESSED_DIR / "dataset_ML.csv"
DATASET_RL_CSV = DATASETS_PROCESSED_DIR / "dataset_RL.csv"
DATASET_SIM_CSV = DATASETS_PROCESSED_DIR / "dataset_simulador.csv"

# Outputs
RUNS_DIR = BASE_DIR / "runs"
RL_RUNS_DIR = RUNS_DIR / "rl"
ML_RUNS_DIR = RUNS_DIR / "ml"
RL_FILTRADO_RUNS_DIR = RL_RUNS_DIR / "filtrado"
ML_FILTRADO_RUNS_DIR = ML_RUNS_DIR / "filtrado"
RL_RAW_RUNS_DIR = RL_RUNS_DIR / "raw"
ML_RAW_RUNS_DIR = ML_RUNS_DIR / "raw"