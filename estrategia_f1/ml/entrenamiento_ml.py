"""
entrenamiento_ml.py
TODO
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, top_k_accuracy_score

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.base import ClassifierMixin

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from estrategia_f1.config import ESTADO_COLS

from estrategia_f1.acciones import (
    construir_mapa_acciones,
    construir_grupos,
    construir_estado_df,
)

from estrategia_f1.features import (
    precomputar_features_acciones,
)

# TODO
# Balancear las clases!!
# Dos approaches: 1 solo modelo que prediga directamente, y 2 modelos: 1 para predecir el número de stints,
# y 1 para predecir la secuencia dada n_stints

# Ajustes del entrenamiento---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ConfiguracionEntrenamientoML:
    seed: int
    test_size: float
    # Modelo "ridge" | "random_forest" | "hist_gb" | "mlp"
    modelo: str = "hist_gb"
    modelo_params: dict[str, Any] | None = None

@dataclass(frozen=True)
class DireccionesML:
    ruta_modelo: Path
    ruta_meta: Path

# Modelo----------------------------------------------------------------------------------------------------------------
def construir_modelo(*, nombre: str, seed: int, params: dict[str, Any] | None = None) -> ClassifierMixin:
    """
    Construye el clasificador para ML (estado -> action_id).
    """
    nombre = str(nombre).strip().lower()
    params = dict(params or {})

    if nombre == "hist_gb":
        defaults = dict(
            loss="log_loss",
            learning_rate=0.06,
            max_iter=300,
            random_state=seed,
        )
        defaults.update(params)
        return HistGradientBoostingClassifier(**defaults)

    if nombre == "logreg":
        defaults = dict(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=500,
            random_state=seed,
        )
        defaults.update(params)
        return LogisticRegression(**defaults)

    if nombre == "random_forest":
        defaults = dict(
            n_estimators=500,
            random_state=seed,
            n_jobs=-1,
        )
        defaults.update(params)
        return RandomForestClassifier(**defaults)

    if nombre == "mlp":
        defaults = dict(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=200,
            random_state=seed,
            early_stopping=True,
        )
        defaults.update(params)
        return MLPClassifier(**defaults)

    raise ValueError("El modelo usado no es reconocido. Usa: 'hist_gb', 'logreg', 'random_forest', 'mlp'.")

def entrenar_modelo(X: np.ndarray, y: np.ndarray, *, configuracionML: ConfiguracionEntrenamientoML) -> ClassifierMixin:
    modelo_base = construir_modelo(
        nombre=configuracionML.modelo,
        seed=configuracionML.seed,
        params=configuracionML.modelo_params
    )

    modelos_que_escalan = {"mlp", "logreg"}

    if configuracionML.modelo in modelos_que_escalan:
        modelo = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", modelo_base),
        ])
    else:
        modelo = modelo_base

    modelo.fit(X, y)
    return modelo

def evaluar_clasificador(modelo, X: np.ndarray, y: np.ndarray, *, topk: tuple[int, ...] = (1, 3, 5)) -> dict:
    pred = modelo.predict(X)
    resultados = {
        "accuracy": float(accuracy_score(y, pred)),
    }

    if not hasattr(modelo, "predict_proba"):
        return resultados

    proba = modelo.predict_proba(X)
    labels = np.asarray(getattr(modelo, "classes_", []), dtype=int)

    # ---- FILTRO: quedarnos solo con y que estén en labels ----
    labels_set = set(labels.tolist())
    mask = np.array([int(v) in labels_set for v in y], dtype=bool)

    resultados["pct_test_labels_vistas_en_train"] = float(mask.mean() * 100.0)

    if mask.sum() == 0:
        # No hay ninguna muestra evaluable para top-k
        for k in topk:
            resultados[f"top{k}_accuracy"] = np.nan
        return resultados

    y_ok = y[mask]
    proba_ok = proba[mask]

    for k in topk:
        resultados[f"top{k}_accuracy"] = float(
            top_k_accuracy_score(y_ok, proba_ok, k=k, labels=labels)
        )

    return resultados

# Entrenamiento---------------------------------------------------------------------------------------------------------
def entrenar_ml_v1(df: pd.DataFrame, *, configuracionML: ConfiguracionEntrenamientoML, paths: DireccionesML) -> dict:
    """
    Entrena (o carga) pares/modelo y devuelve un dict con:
      - modelo
      - meta
      - stats
      - metricas_clasificador
      - columnas_estado
      - mapa_acciones, representacion_accion, ids_acciones
      - df_test, X_test_estado
    """
    paths.ruta_meta.parent.mkdir(parents=True, exist_ok=True)
    paths.ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
    paths.ruta_meta.parent.mkdir(parents=True, exist_ok=True)
    paths.ruta_modelo.parent.mkdir(parents=True, exist_ok=True)

    # Mapa de acciones + representación numérica de la acción
    mapa_acciones = construir_mapa_acciones()
    ids_acciones = np.array(sorted(mapa_acciones.keys()), dtype=int)
    representacion_accion = precomputar_features_acciones(mapa_acciones)

    if "action_id" not in df.columns:
        raise KeyError("Falta columna 'action_id' en df para entrenar ML.")

    y_raw = pd.to_numeric(df["action_id"], errors="coerce")
    mask = np.isfinite(y_raw.to_numpy())
    df_ok = df.loc[mask].copy().reset_index(drop=True)
    y = y_raw.loc[mask].astype(int).to_numpy()

    # Estado
    X_estado = construir_estado_df(df_ok, columnas=ESTADO_COLS, columnas_excluir=[], imputar_numericas=True)

    # Grupos para split
    grupos = construir_grupos(df_ok)

    # Split (cache)
    if paths.ruta_meta.exists():
        meta = joblib.load(paths.ruta_meta)
        idx_train = meta["idx_train"]
        idx_test = meta["idx_test"]

        cols_guardadas = meta.get("estado_cols_raw")
        if cols_guardadas is not None and list(X_estado.columns) != list(cols_guardadas):
            print("Han cambiado las columnas de estado respecto a meta.")
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=configuracionML.test_size, random_state=configuracionML.seed)
        idx_train, idx_test = next(gss.split(df_ok, groups=grupos))

        meta = {
            "idx_train": idx_train,
            "idx_test": idx_test,
            "estado_cols_raw": list(X_estado.columns),
            "onehot_estado": True,
            "seed": configuracionML.seed,
            "test_size": configuracionML.test_size,
            "modelo": configuracionML.modelo,
            "modelo_params": configuracionML.modelo_params,
            "grupo_columna": "race_id",
        }
        joblib.dump(meta, paths.ruta_meta)

    # Split train/test
    df_test = df_ok.iloc[idx_test].reset_index(drop=True)

    X_train_estado_df = X_estado.iloc[idx_train].reset_index(drop=True)
    X_test_estado_df = X_estado.iloc[idx_test].reset_index(drop=True)

    y_train = y[idx_train]
    y_test = y[idx_test]

    # One-hot del estado
    num_cols = [c for c in X_train_estado_df.columns if pd.api.types.is_numeric_dtype(X_train_estado_df[c])]
    cat_cols = [c for c in X_train_estado_df.columns if c not in num_cols]

    pre_estado = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ],
        remainder="drop",
    )

    X_train_estado = pre_estado.fit_transform(X_train_estado_df)
    X_test_estado = pre_estado.transform(X_test_estado_df)

    if hasattr(X_train_estado, "toarray"):
        X_train_estado = X_train_estado.toarray()
        X_test_estado = X_test_estado.toarray()

    X_train_estado = X_train_estado.astype(np.float32, copy=False)
    X_test_estado = X_test_estado.astype(np.float32, copy=False)

    # Modelo (cache)
    if paths.ruta_modelo.exists():
        modelo = joblib.load(paths.ruta_modelo)
    else:
        modelo = entrenar_modelo(X_train_estado, y_train, configuracionML=configuracionML)
        joblib.dump(modelo, paths.ruta_modelo)

    metricas_clasificador = evaluar_clasificador(modelo, X_test_estado, y_test)

    return {
        "modelo": modelo,
        "meta": meta,
        "metricas_clasificador": metricas_clasificador,
        "columnas_estado": list(X_estado.columns),
        "mapa_acciones": mapa_acciones,
        "representacion_accion": representacion_accion,
        "ids_acciones": ids_acciones,
        "df_test": df_test,
        "X_test_estado": X_test_estado,
        "y_test": y_test,
    }