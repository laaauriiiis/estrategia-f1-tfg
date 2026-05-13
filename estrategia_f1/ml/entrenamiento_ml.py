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
    Los hiperparámetros vienen de config.py (params).
    """
    nombre = str(nombre).strip().lower()
    params = dict(params or {})

    if nombre == "hist_gb":
        base = dict(
            loss="log_loss",
            random_state=seed,
        )
        base.update(params)
        return HistGradientBoostingClassifier(**base)

    if nombre == "logreg":
        base = dict(
            random_state=seed,
        )
        base.update(params)
        return LogisticRegression(**base)

    if nombre == "random_forest":
        base = dict(
            random_state=seed,
        )
        base.update(params)
        return RandomForestClassifier(**base)

    if nombre == "mlp":
        base = dict(
            random_state=seed,
            early_stopping=True,
        )
        base.update(params)
        return MLPClassifier(**base)

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

    Importante:
    - El split train/test se construye siempre sobre un dataset base común.
    - Si aplicar_filtros=True, el filtrado se aplica SOLO al train.
    - El test se mantiene igual entre variantes raw y filtrado.
    """
    paths.ruta_meta.parent.mkdir(parents=True, exist_ok=True)
    paths.ruta_modelo.parent.mkdir(parents=True, exist_ok=True)

    # Mapa de acciones + representación numérica de la acción
    mapa_acciones = construir_mapa_acciones()
    ids_acciones = np.array(sorted(mapa_acciones.keys()), dtype=int)
    representacion_accion = precomputar_features_acciones(mapa_acciones)

    if "action_id" not in df.columns:
        raise KeyError("Falta columna 'action_id' en df para entrenar ML.")

    # -------------------------------------------------------------------------
    # 1) DATASET BASE COMÚN (sin filtrar), para que el split sea comparable
    # -------------------------------------------------------------------------
    y_raw = pd.to_numeric(df["action_id"], errors="coerce")
    mask = np.isfinite(y_raw.to_numpy())

    df_base = df.loc[mask].copy().reset_index(drop=True)
    y_base = y_raw.loc[mask].astype(int).to_numpy()

    if len(df_base) == 0:
        raise ValueError("El dataset base quedó vacío tras validar action_id.")

    # Estado base (solo para definir columnas y split común)
    X_estado_base = construir_estado_df(
        df_base,
        columnas=ESTADO_COLS,
        columnas_excluir=[],
        imputar_numericas=True,
    )

    grupos_base = construir_grupos(df_base)

    # Hash del dataset base común
    hash_dataset_base = calcular_hash_dataset(df_base)

    # Firma SOLO del split base común
    stats_split = {
        "aplicado": False,
        "tipo_pipeline": "split_base",
        "n_inicial": int(len(df_base)),
        "n_final": int(len(df_base)),
        "hash_dataset": hash_dataset_base,
    }

    firma_split = firma_entrenamiento_ml(
        df_hash=hash_dataset_base,
        columnas_estado=list(X_estado_base.columns),
        configuracionML=ConfiguracionEntrenamientoML(
            seed=configuracionML.seed,
            test_size=configuracionML.test_size,
            modelo="split_base",
            modelo_params=None,
        ),
        stats_filtros=stats_split,
    )

    # -------------------------------------------------------------------------
    # 2) CACHE DEL SPLIT BASE
    # -------------------------------------------------------------------------
    meta_split_reutilizable = False
    meta = None

    if paths.ruta_meta.exists():
        meta = joblib.load(paths.ruta_meta)
        firma_anterior = meta.get("signature_split")

        if firma_anterior != firma_split:
            print("Detectados cambios en dataset base/columnas base del split.")
            print(f"Firma split anterior: {firma_anterior}")
            print(f"Firma split actual  : {firma_split}")
            print("Invalidando caché de meta y modelo...")
            invalidar_archivos(paths.ruta_meta, paths.ruta_modelo)
            meta = None
        else:
            meta_split_reutilizable = True
            print("Usando cache existente de train/test split")

    if meta_split_reutilizable and meta is not None:
        idx_train = meta["idx_train"]
        idx_test = meta["idx_test"]
    else:
        print("Generando nuevo train/test split...")
        carreras = (
            df_base[["race_id", "race_date"]]
            .drop_duplicates()
            .sort_values("race_date")
        )

        n_test = int(len(carreras) * configuracionML.test_size)

        races_train = carreras.iloc[:-n_test]["race_id"]
        races_test = carreras.iloc[-n_test:]["race_id"]

        idx_train = df_base.index[df_base["race_id"].isin(races_train)].to_numpy()
        idx_test = df_base.index[df_base["race_id"].isin(races_test)].to_numpy()
        meta = {
            "idx_train": idx_train,
            "idx_test": idx_test,
            "estado_cols_raw_base": list(X_estado_base.columns),
            "seed": configuracionML.seed,
            "test_size": configuracionML.test_size,
            "grupo_columna": "race_id",
            "signature_split": firma_split,
            "timestamp_split": pd.Timestamp.now().isoformat(),
        }
        joblib.dump(meta, paths.ruta_meta)

    # -------------------------------------------------------------------------
    # 3) TRAIN BASE / TEST FIJO
    # -------------------------------------------------------------------------
    df_train_base = df_base.iloc[idx_train].reset_index(drop=True)
    df_test = df_base.iloc[idx_test].reset_index(drop=True)
    y_test = y_base[idx_test]

    # -------------------------------------------------------------------------
    # 4) FILTRADO SOLO EN TRAIN
    # -------------------------------------------------------------------------
    if aplicar_filtros:
        print("Aplicando filtros específicos para ML SOLO sobre train...")
        df_train, stats_filtros = filtrar_dataset(df_train_base, tipo_pipeline="ml")

        if len(df_train) == 0:
            raise ValueError("El train quedó vacío después del filtrado ML")

        print(f"Filtrado ML en train completado: {len(df_train):,} filas restantes\n")
    else:
        df_train = df_train_base.copy()
        stats_filtros = {
            "aplicado": False,
            "n_inicial": int(len(df_train_base)),
            "n_final": int(len(df_train_base)),
            "tipo_pipeline": "ninguno",
        }

    # Hash del train final usado para entrenar
    hash_dataset_train = calcular_hash_dataset(df_train)
    stats_filtros["hash_dataset"] = hash_dataset_train

    # -------------------------------------------------------------------------
    # 5) ESTADOS DEFINITIVOS TRAIN / TEST
    # -------------------------------------------------------------------------
    X_train_estado_df = construir_estado_df(
        df_train,
        columnas=ESTADO_COLS,
        columnas_excluir=[],
        imputar_numericas=True,
    )

    X_test_estado_df = construir_estado_df(
        df_test,
        columnas=ESTADO_COLS,
        columnas_excluir=[],
        imputar_numericas=True,
    )

    # Firma del entrenamiento (depende del train final)
    firma_entrenamiento = firma_entrenamiento_ml(
        df_hash=hash_dataset_train,
        columnas_estado=list(X_train_estado_df.columns),
        configuracionML=configuracionML,
        stats_filtros=stats_filtros,
    )

    # Si cambia la firma de entrenamiento, invalidar solo el modelo
    firma_modelo_anterior = meta.get("signature_modelo")
    modelo_reutilizable = False

    if firma_modelo_anterior == firma_entrenamiento and paths.ruta_modelo.exists():
        modelo_reutilizable = True
    else:
        if paths.ruta_modelo.exists():
            print("Detectados cambios en train/configuración ML. Reentrenando modelo...")
            invalidar_archivos(paths.ruta_modelo)
        modelo_reutilizable = False

    # y_train debe reconstruirse desde df_train
    y_train_raw = pd.to_numeric(df_train["action_id"], errors="coerce")
    y_train = y_train_raw.astype(int).to_numpy()

    # -------------------------------------------------------------------------
    # 6) PREPROCESADO
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 7) MODELO
    # -------------------------------------------------------------------------
    if modelo_reutilizable:
        print("Cargando modelo desde cache...")
        modelo = joblib.load(paths.ruta_modelo)
    else:
        print("Entrenando nuevo modelo...")
        modelo = entrenar_modelo(X_train_estado, y_train, configuracionML=configuracionML)
        joblib.dump(modelo, paths.ruta_modelo)

        # actualizar meta con info del entrenamiento actual
        meta["signature_modelo"] = firma_entrenamiento
        meta["modelo"] = configuracionML.modelo
        meta["modelo_params"] = configuracionML.modelo_params
        meta["stats_filtros_train"] = stats_filtros
        meta["timestamp_modelo"] = pd.Timestamp.now().isoformat()
        joblib.dump(meta, paths.ruta_meta)

    return {
        "modelo": modelo,
        "meta": meta,
        "stats_filtros": stats_filtros,
        "columnas_estado": list(X_train_estado_df.columns),
        "mapa_acciones": mapa_acciones,
        "representacion_accion": representacion_accion,
        "ids_acciones": ids_acciones,
        "df_test": df_test,
        "X_test_estado": X_test_estado,
        "y_test": y_test,
    }