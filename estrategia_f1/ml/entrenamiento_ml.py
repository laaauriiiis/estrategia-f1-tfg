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

from estrategia_f1.data.filtro_outliers import filtrar_dataset

from estrategia_f1.cache_utils import (
    calcular_hash_dataset,
    firma_entrenamiento_ml,
    invalidar_archivos,
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
    # Modelo "ridge" | "logreg" | "random_forest" | "hist_gb" | "mlp"
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
    NOTA: No incluye class_weight aquí, se maneja en entrenar_modelo()
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
            n_estimators=400,  # Cambio de 500 a 400 para consistencia con config
            max_depth=None,    # Agregar max_depth=None por defecto
            n_jobs=-1,         # Agregar paralelización
            random_state=seed,
        )
        defaults.update(params)
        return RandomForestClassifier(**defaults)

    if nombre == "mlp":
        defaults = dict(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=300,      # Cambio de 200 a 300 para consistencia
            random_state=seed,
            early_stopping=True,
        )
        defaults.update(params)
        return MLPClassifier(**defaults)

    raise ValueError("El modelo usado no es reconocido. Usa: 'hist_gb', 'logreg', 'random_forest', 'mlp'.")


def entrenar_modelo(X: np.ndarray, y: np.ndarray, *, configuracionML: ConfiguracionEntrenamientoML) -> ClassifierMixin:
    """
    Construye y entrena el modelo sin balanceo de clases.
    """
    params_originales = configuracionML.modelo_params or {}

    modelo_base = construir_modelo(
        nombre=configuracionML.modelo,
        seed=configuracionML.seed,
        params=params_originales,
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

# Entrenamiento---------------------------------------------------------------------------------------------------------
def entrenar_ml_v1(
    df: pd.DataFrame,
    *,
    configuracionML: ConfiguracionEntrenamientoML,
    paths: DireccionesML,
    aplicar_filtros: bool = True,
) -> dict:
    """
    Entrena (o carga) modelo ML y devuelve un dict con resultados.
    """
    # Aplicar filtros específicos para ML al inicio si está habilitado
    if aplicar_filtros:
        print("Aplicando filtros específicos para ML...")
        df_filtrado, stats_filtros = filtrar_dataset(df, tipo_pipeline="ml")

        if len(df_filtrado) == 0:
            raise ValueError("El dataset quedó vacío después del filtrado ML")

        print(f"Filtrado ML completado: {len(df_filtrado):,} filas restantes\n")
        df = df_filtrado
    else:
        stats_filtros = {
            "aplicado": False,
            "n_inicial": int(len(df)),
            "n_final": int(len(df)),
            "tipo_pipeline": "ninguno",
        }

    paths.ruta_meta.parent.mkdir(parents=True, exist_ok=True)
    paths.ruta_modelo.parent.mkdir(parents=True, exist_ok=True)

    # Mapa de acciones + representación numérica de la acción
    mapa_acciones = construir_mapa_acciones()
    ids_acciones = np.array(sorted(mapa_acciones.keys()), dtype=int)
    representacion_accion = precomputar_features_acciones(mapa_acciones)

    if "action_id" not in df.columns:
        raise KeyError("Falta columna 'action_id' en df para entrenar ML.")

    # Dataset final real de entrenamiento/split
    y_raw = pd.to_numeric(df["action_id"], errors="coerce")
    mask = np.isfinite(y_raw.to_numpy())
    df_ok = df.loc[mask].copy().reset_index(drop=True)
    y = y_raw.loc[mask].astype(int).to_numpy()

    if len(df_ok) == 0:
        raise ValueError("El dataset quedó vacío tras validar action_id.")

    # Hash robusto del dataset final
    hash_dataset = calcular_hash_dataset(df_ok)
    stats_filtros["hash_dataset"] = hash_dataset

    # Estado
    X_estado = construir_estado_df(
        df_ok,
        columnas=ESTADO_COLS,
        columnas_excluir=[],
        imputar_numericas=True,
    )

    # Firma del experimento actual
    firma_actual = firma_entrenamiento_ml(
        df_hash=hash_dataset,
        columnas_estado=list(X_estado.columns),
        configuracionML=configuracionML,
        stats_filtros=stats_filtros,
    )

    # Grupos para split
    grupos = construir_grupos(df_ok)

    # Validación de meta por firma
    meta_reutilizable = False
    meta = None

    if paths.ruta_meta.exists():
        meta = joblib.load(paths.ruta_meta)
        firma_anterior = meta.get("signature")

        if firma_anterior != firma_actual:
            print("Detectados cambios en datos/columnas/configuración ML.")
            print(f"Firma anterior: {firma_anterior}")
            print(f"Firma actual  : {firma_actual}")
            print("Invalidando caché de meta y modelo...")
            invalidar_archivos(paths.ruta_meta, paths.ruta_modelo)
            meta = None
        else:
            meta_reutilizable = True
            print("Usando cache existente de train/test split")

    # Split (cache)
    if meta_reutilizable and meta is not None:
        idx_train = meta["idx_train"]
        idx_test = meta["idx_test"]
    else:
        print("Generando nuevo train/test split...")
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=configuracionML.test_size,
            random_state=configuracionML.seed,
        )
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
            "stats_filtros": stats_filtros,
            "signature": firma_actual,
            "timestamp": pd.Timestamp.now().isoformat(),
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
    if paths.ruta_modelo.exists() and meta_reutilizable:
        print("Cargando modelo desde cache...")
        modelo = joblib.load(paths.ruta_modelo)
    else:
        print("Entrenando nuevo modelo...")
        modelo = entrenar_modelo(X_train_estado, y_train, configuracionML=configuracionML)
        joblib.dump(modelo, paths.ruta_modelo)

    return {
        "modelo": modelo,
        "meta": meta,
        "stats_filtros": stats_filtros,
        "columnas_estado": list(X_estado.columns),
        "mapa_acciones": mapa_acciones,
        "representacion_accion": representacion_accion,
        "ids_acciones": ids_acciones,
        "df_test": df_test,
        "X_test_estado": X_test_estado,
        "y_test": y_test,
    }