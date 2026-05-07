"""
entrenamiento_rl.py
TODO
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.base import RegressorMixin
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from estrategia_f1.sim.simulador import simular_tiempo_carrera

from estrategia_f1.config import ESTADO_COLS

from estrategia_f1.acciones import (
    construir_mapa_acciones,
    compuestos_disponibles,
    estrategia_desde_accion_id,
    elegir_estrategia_baseline,
    construir_grupos,
    construir_estado_df,
)

from estrategia_f1.features import (
    precomputar_features_acciones,
)

from estrategia_f1.data.filtro_outliers import filtrar_dataset

from estrategia_f1.cache_utils import (
    calcular_hash_dataset,
    firma_entrenamiento_rl,
    invalidar_archivos,
)

# Ajustes del entrenamiento---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ConfiguracionEntrenamientoRL:
    seed: int
    test_size: float
    k_acciones_muestreo: int
    # Aproximador de Q(s,a) "ridge" | "random_forest" | "hist_gb" | "mlp"
    modelo_q: str = "hist_gb"
    modelo_q_params: dict[str, Any] | None = None


@dataclass(frozen=True)
class DireccionesRL:
    ruta_pares: Path
    ruta_modelo: Path
    ruta_meta: Path


# Dataset de pares (s,a) -> recompensa----------------------------------------------------------------------------------
def construir_dataset_pares(df_sub: pd.DataFrame, matriz_estados: np.ndarray, mapa_acciones: dict[int, list[str]],
    representacion_accion: dict[int, np.ndarray], ids_acciones: np.ndarray, *,
    configuracionRL: ConfiguracionEntrenamientoRL) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Devuelve:
      - X_sa (np.ndarray): concat(estado, representacion_accion)
      - y (np.ndarray): recompensa
      - stats (dict): contadores de omitidos
    """
    gen_aleatorio = np.random.default_rng(configuracionRL.seed)

    X: list[np.ndarray] = []
    y: list[float] = []
    filas_omitidas = 0
    pares_omitidos = 0

    for i in tqdm(range(len(df_sub)), desc="Construyendo pares (s,a)"):
        fila = df_sub.iloc[i]
        estado = matriz_estados[i]

        baseline = elegir_estrategia_baseline(fila)
        if baseline is None:
            filas_omitidas += 1
            continue

        tiempo_carrera_baseline = simular_tiempo_carrera(fila, baseline)
        if not np.isfinite(tiempo_carrera_baseline):
            filas_omitidas += 1
            continue

        # Muestreo de acciones
        if configuracionRL.k_acciones_muestreo >= len(ids_acciones):
            acciones_muestreadas = ids_acciones
        else:
            acciones_muestreadas = gen_aleatorio.choice(
                ids_acciones,
                size=configuracionRL.k_acciones_muestreo,
                replace=False,
            )

        disponibles = compuestos_disponibles(fila)

        for accion_id in acciones_muestreadas:
            estrategia = estrategia_desde_accion_id(int(accion_id), mapa_acciones)

            if not set(estrategia).issubset(disponibles):
                pares_omitidos += 1
                continue

            tiempo_carrera_accion = simular_tiempo_carrera(fila, estrategia)
            if not np.isfinite(tiempo_carrera_accion):
                pares_omitidos += 1
                continue

            recompensa = -(tiempo_carrera_accion - tiempo_carrera_baseline)  # >0 mejora vs baseline
            x = np.concatenate([estado, representacion_accion[int(accion_id)]], axis=0)

            X.append(x.astype(np.float32, copy=False))
            y.append(float(recompensa))

    stats = {"filas_omitidas": filas_omitidas, "pares_omitidos": pares_omitidos}

    if len(X) == 0:
        dim_accion = len(next(iter(representacion_accion.values())))
        return (
            np.zeros((0, matriz_estados.shape[1] + dim_accion), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            stats,
        )

    return np.vstack(X).astype(np.float32), np.array(y, dtype=np.float32), stats


# Aproximador Q(s,a)----------------------------------------------------------------------------------------------------
def construir_modelo_q(*, nombre: str, seed: int, params: dict[str, Any] | None = None) -> RegressorMixin:
    """
    Construye el aproximador de Q(s,a) según 'nombre'.
    Los hiperparámetros vienen de config.py (params).
    """
    nombre = str(nombre).strip().lower()
    params = dict(params or {})

    if nombre == "hist_gb":
        base = dict(
            loss="squared_error",
            random_state=seed,
        )
        base.update(params)
        return HistGradientBoostingRegressor(**base)

    if nombre == "ridge":
        base = dict()
        base.update(params)
        return Ridge(**base)

    if nombre == "random_forest":
        base = dict(
            random_state=seed,
        )
        base.update(params)
        return RandomForestRegressor(**base)

    if nombre == "mlp":
        base = dict(
            random_state=seed,
            early_stopping=True,  # esto sí es estructural
        )
        base.update(params)
        return MLPRegressor(**base)

    raise ValueError("El modelo usado no es reconocido. Usa: 'hist_gb', 'ridge', 'random_forest', 'mlp'.")


def entrenar_modelo_q(X: np.ndarray, y: np.ndarray, *, configuracionRL: ConfiguracionEntrenamientoRL) -> RegressorMixin:
    modelo_base = construir_modelo_q(
        nombre=configuracionRL.modelo_q,
        seed=configuracionRL.seed,
        params=configuracionRL.modelo_q_params,
    )

    modelos_que_escalan = {"mlp", "ridge"}

    if configuracionRL.modelo_q in modelos_que_escalan:
        modelo = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", modelo_base),
        ])
    else:
        modelo = modelo_base

    modelo.fit(X, y)
    return modelo


def evaluar_regresor(modelo: RegressorMixin, X: np.ndarray, y: np.ndarray) -> dict:
    pred = modelo.predict(X)
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }


def _n_features_modelo(modelo) -> int | None:
    """
    Devuelve n_features esperado por el modelo (si sklearn lo expone).
    Soporta Pipeline y estimadores directos.
    """
    try:
        if hasattr(modelo, "named_steps") and "reg" in modelo.named_steps:
            est = modelo.named_steps["reg"]
        else:
            est = modelo

        return int(getattr(est, "n_features_in_", None)) if getattr(est, "n_features_in_", None) is not None else None
    except Exception:
        return None


# Entrenamiento---------------------------------------------------------------------------------------------------------
def entrenar_rl_offline(
    df: pd.DataFrame,
    *,
    configuracionRL: ConfiguracionEntrenamientoRL,
    paths: DireccionesRL,
    aplicar_filtros: bool = True,
) -> dict:
    """
    Entrena (o carga) pares/modelo y devuelve un dict con:
      - modelo
      - meta
      - stats
      - stats_filtros
      - metricas_regresor
      - columnas_estado
      - mapa_acciones, representacion_accion, ids_acciones
      - df_test, X_test_estado

    Importante:
    - El split train/test se construye siempre sobre un dataset base común.
    - Si aplicar_filtros=True, el filtrado se aplica SOLO al train.
    - El test se mantiene igual entre variantes raw y filtrado.
    """
    paths.ruta_meta.parent.mkdir(parents=True, exist_ok=True)
    paths.ruta_pares.parent.mkdir(parents=True, exist_ok=True)
    paths.ruta_modelo.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1) DATASET BASE COMÚN (sin filtrar), para que el split sea comparable
    # -------------------------------------------------------------------------
    df_base = df.copy().reset_index(drop=True)

    if len(df_base) == 0:
        raise ValueError("El dataset base quedó vacío en RL.")

    # Mapa de acciones + representación numérica de la acción
    mapa_acciones = construir_mapa_acciones()
    ids_acciones = np.array(sorted(mapa_acciones.keys()), dtype=int)
    representacion_accion = precomputar_features_acciones(mapa_acciones)

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
        "tipo_pipeline": "split_base_rl",
        "n_inicial": int(len(df_base)),
        "n_final": int(len(df_base)),
        "hash_dataset": hash_dataset_base,
    }

    firma_split = firma_entrenamiento_rl(
        df_hash=hash_dataset_base,
        columnas_estado=list(X_estado_base.columns),
        configuracionRL=ConfiguracionEntrenamientoRL(
            seed=configuracionRL.seed,
            test_size=configuracionRL.test_size,
            k_acciones_muestreo=configuracionRL.k_acciones_muestreo,
            modelo_q="split_base_rl",
            modelo_q_params=None,
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
            print("Detectados cambios en dataset base/columnas base del split RL.")
            print(f"Firma split anterior: {firma_anterior}")
            print(f"Firma split actual  : {firma_split}")
            print("Invalidando caché de meta, pares y modelo...")
            invalidar_archivos(paths.ruta_meta, paths.ruta_pares, paths.ruta_modelo)
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

        n_test = int(len(carreras) * configuracionRL.test_size)

        races_train = carreras.iloc[:-n_test]["race_id"]
        races_test = carreras.iloc[-n_test:]["race_id"]

        idx_train = df_base.index[df_base["race_id"].isin(races_train)].to_numpy()
        idx_test = df_base.index[df_base["race_id"].isin(races_test)].to_numpy()

        meta = {
            "idx_train": idx_train,
            "idx_test": idx_test,
            "estado_cols_raw_base": list(X_estado_base.columns),
            "seed": configuracionRL.seed,
            "test_size": configuracionRL.test_size,
            "k_acciones_muestreo": configuracionRL.k_acciones_muestreo,
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

    # -------------------------------------------------------------------------
    # 4) FILTRADO SOLO EN TRAIN
    # -------------------------------------------------------------------------
    if aplicar_filtros:
        print("Aplicando filtros RL SOLO sobre train...")
        df_train, stats_filtros = filtrar_dataset(df_train_base, tipo_pipeline="rl")

        if len(df_train) == 0:
            raise ValueError("El train quedó vacío después del filtrado RL")

        print(f"Filtrado RL en train completado: {len(df_train):,} filas restantes\n")
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

    # Firma del entrenamiento RL (depende del train final)
    firma_entrenamiento = firma_entrenamiento_rl(
        df_hash=hash_dataset_train,
        columnas_estado=list(X_train_estado_df.columns),
        configuracionRL=configuracionRL,
        stats_filtros=stats_filtros,
    )

    # Si cambia la firma de entrenamiento, invalidar pares y modelo
    firma_train_anterior = meta.get("signature_train_rl")
    cache_entrenamiento_reutilizable = False

    if firma_train_anterior == firma_entrenamiento:
        cache_entrenamiento_reutilizable = True
    else:
        if paths.ruta_pares.exists() or paths.ruta_modelo.exists():
            print("Detectados cambios en train/configuración RL. Regenerando pares y modelo...")
            invalidar_archivos(paths.ruta_pares, paths.ruta_modelo)
        cache_entrenamiento_reutilizable = False

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
    # 7) PARES (cache)
    # -------------------------------------------------------------------------
    if paths.ruta_pares.exists() and cache_entrenamiento_reutilizable:
        print("Cargando pares desde cache...")
        pares = np.load(paths.ruta_pares, allow_pickle=True)
        X_train_pares = pares["X_train_pares"]
        y_train = pares["y_train"]
        X_test_pares = pares["X_test_pares"]
        y_test = pares["y_test"]
        stats = {"cache": True}
    else:
        print("Construyendo nuevos pares (s,a)...")
        X_train_pares, y_train, stats_train = construir_dataset_pares(
            df_train,
            X_train_estado,
            mapa_acciones,
            representacion_accion,
            ids_acciones,
            configuracionRL=configuracionRL,
        )
        X_test_pares, y_test, stats_test = construir_dataset_pares(
            df_test,
            X_test_estado,
            mapa_acciones,
            representacion_accion,
            ids_acciones,
            configuracionRL=configuracionRL,
        )
        stats = {"cache": False, "train": stats_train, "test": stats_test}

        np.savez(
            paths.ruta_pares,
            X_train_pares=X_train_pares,
            y_train=y_train,
            X_test_pares=X_test_pares,
            y_test=y_test,
        )

    # -------------------------------------------------------------------------
    # 8) MODELO
    # -------------------------------------------------------------------------
    if paths.ruta_modelo.exists() and cache_entrenamiento_reutilizable:
        print("Cargando modelo RL desde cache...")
        modelo = joblib.load(paths.ruta_modelo)
        n_expected = _n_features_modelo(modelo)
        n_actual = int(X_train_pares.shape[1])

        if n_expected is not None and n_expected != n_actual:
            print(f"[CACHE INVÁLIDO] Modelo espera {n_expected} features, pero ahora hay {n_actual}. Reentrenando...")
            invalidar_archivos(paths.ruta_modelo)
            modelo = entrenar_modelo_q(X_train_pares, y_train, configuracionRL=configuracionRL)
            joblib.dump(modelo, paths.ruta_modelo)
    else:
        print("Entrenando nuevo modelo RL...")
        modelo = entrenar_modelo_q(X_train_pares, y_train, configuracionRL=configuracionRL)
        joblib.dump(modelo, paths.ruta_modelo)

    # actualizar meta con info del entrenamiento actual
    meta["signature_train_rl"] = firma_entrenamiento
    meta["modelo_q"] = configuracionRL.modelo_q
    meta["modelo_q_params"] = configuracionRL.modelo_q_params
    meta["stats_filtros_train"] = stats_filtros
    meta["timestamp_modelo"] = pd.Timestamp.now().isoformat()
    joblib.dump(meta, paths.ruta_meta)

    metricas_regresor = evaluar_regresor(modelo, X_test_pares, y_test)

    return {
        "modelo": modelo,
        "meta": meta,
        "stats": stats,
        "stats_filtros": stats_filtros,
        "metricas_regresor": metricas_regresor,
        "columnas_estado": list(X_train_estado_df.columns),
        "mapa_acciones": mapa_acciones,
        "representacion_accion": representacion_accion,
        "ids_acciones": ids_acciones,
        "df_test": df_test,
        "X_test_estado": X_test_estado,
    }