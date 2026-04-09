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
    """
    nombre = str(nombre).strip().lower()
    params = dict(params or {})

    if nombre == "hist_gb":
        defaults = dict(
            loss="squared_error",
            learning_rate=0.06,
            max_iter=300,
            random_state=seed,
        )
        defaults.update(params)
        return HistGradientBoostingRegressor(**defaults)

    if nombre == "ridge":
        defaults = dict(alpha=1.0)
        defaults.update(params)
        return Ridge(**defaults)

    if nombre == "random_forest":
        defaults = dict(
            n_estimators=500,
            random_state=seed,
            n_jobs=-1,
        )
        defaults.update(params)
        return RandomForestRegressor(**defaults)

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
        return MLPRegressor(**defaults)

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
def entrenar_rl_offline(df: pd.DataFrame, *, configuracionRL: ConfiguracionEntrenamientoRL, paths: DireccionesRL,
    aplicar_filtros: bool = True) -> dict:
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
    """
    # Aplicar filtros al inicio si está habilitado
    if aplicar_filtros:
        print("Aplicando filtros al dataset...")
        df_filtrado, stats_filtros = filtrar_dataset(df, tipo_pipeline="rl")

        if len(df_filtrado) == 0:
            raise ValueError("El dataset quedó vacío después del filtrado")

        print(f"Filtrado completado: {len(df_filtrado):,} filas restantes\n")
        df = df_filtrado
    else:
        stats_filtros = {"aplicado": False, "n_inicial": int(len(df)), "n_final": int(len(df))}

    paths.ruta_meta.parent.mkdir(parents=True, exist_ok=True)
    paths.ruta_pares.parent.mkdir(parents=True, exist_ok=True)
    paths.ruta_modelo.parent.mkdir(parents=True, exist_ok=True)

    # Mapa de acciones + representación numérica de la acción
    mapa_acciones = construir_mapa_acciones()
    ids_acciones = np.array(sorted(mapa_acciones.keys()), dtype=int)
    representacion_accion = precomputar_features_acciones(mapa_acciones)

    # Estado
    X_estado = construir_estado_df(df, columnas=ESTADO_COLS, columnas_excluir=[], imputar_numericas=True)
    grupos = construir_grupos(df)

    # Si existe meta, comprobar compatibilidad también con filtros
    if paths.ruta_meta.exists():
        meta_existente = joblib.load(paths.ruta_meta)

        filtros_anteriores = meta_existente.get("stats_filtros", {})
        n_final_anterior = filtros_anteriores.get("n_final")
        n_final_actual = stats_filtros.get("n_final")

        cols_guardadas = meta_existente.get("estado_cols_raw")
        columnas_estado_cambiaron = cols_guardadas is not None and list(X_estado.columns) != list(cols_guardadas)

        if columnas_estado_cambiaron or n_final_anterior != n_final_actual:
            print("Han cambiado las columnas de estado o el filtrado. Regenerando meta/pares/modelo...")
            paths.ruta_meta.unlink(missing_ok=True)
            paths.ruta_pares.unlink(missing_ok=True)
            paths.ruta_modelo.unlink(missing_ok=True)

    # Split (cache)
    if paths.ruta_meta.exists():
        meta = joblib.load(paths.ruta_meta)
        idx_train = meta["idx_train"]
        idx_test = meta["idx_test"]

        cols_guardadas = meta.get("estado_cols_raw")
        if cols_guardadas is not None and list(X_estado.columns) != list(cols_guardadas):
            print("Han cambiado las columnas de estado respecto a meta. Regenerar meta/pares/modelo.")
    else:
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=configuracionRL.test_size,
            random_state=configuracionRL.seed,
        )
        idx_train, idx_test = next(gss.split(df, groups=grupos))

        meta = {
            "idx_train": idx_train,
            "idx_test": idx_test,
            "estado_cols_raw": list(X_estado.columns),
            "onehot_estado": True,
            "seed": configuracionRL.seed,
            "k_acciones_muestreo": configuracionRL.k_acciones_muestreo,
            "test_size": configuracionRL.test_size,
            "modelo_q": configuracionRL.modelo_q,
            "modelo_q_params": configuracionRL.modelo_q_params,
            "grupo_columna": "race_id",
            "stats_filtros": stats_filtros,
        }
        joblib.dump(meta, paths.ruta_meta)

    # Split train/test
    df_train = df.iloc[idx_train].reset_index(drop=True)
    df_test = df.iloc[idx_test].reset_index(drop=True)

    X_train_estado_df = X_estado.iloc[idx_train].reset_index(drop=True)
    X_test_estado_df = X_estado.iloc[idx_test].reset_index(drop=True)

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

    # Si el cache de pares viene de una versión anterior, lo invalidamos
    print("ruta_pares:", paths.ruta_pares)
    print("exists:", paths.ruta_pares.exists())
    print("exists npz:", paths.ruta_pares.with_suffix(".npz").exists())

    if paths.ruta_pares.exists():
        meta_cache_ok = bool(meta.get("onehot_estado", False))
        if not meta_cache_ok:
            paths.ruta_pares.unlink(missing_ok=True)

    # Pares (cache)
    if paths.ruta_pares.exists():
        pares = np.load(paths.ruta_pares, allow_pickle=True)
        X_train_pares = pares["X_train_pares"]
        y_train = pares["y_train"]
        X_test_pares = pares["X_test_pares"]
        y_test = pares["y_test"]
        stats = {"cache": True}
    else:
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
        print("[GUARDADO] ruta_pares:", paths.ruta_pares)
        print("[GUARDADO] exists:", paths.ruta_pares.exists())
        print("[GUARDADO] size:", paths.ruta_pares.stat().st_size if paths.ruta_pares.exists() else None)

    # Modelo (cache)
    if paths.ruta_modelo.exists():
        modelo = joblib.load(paths.ruta_modelo)
        n_expected = _n_features_modelo(modelo)
        n_actual = int(X_train_pares.shape[1])

        if n_expected is not None and n_expected != n_actual:
            print(f"[CACHE INVALIDO] Modelo espera {n_expected} features, pero ahora hay {n_actual}. Reentrenando...")
            paths.ruta_modelo.unlink(missing_ok=True)
            modelo = entrenar_modelo_q(X_train_pares, y_train, configuracionRL=configuracionRL)
            joblib.dump(modelo, paths.ruta_modelo)
    else:
        modelo = entrenar_modelo_q(X_train_pares, y_train, configuracionRL=configuracionRL)
        joblib.dump(modelo, paths.ruta_modelo)

    metricas_regresor = evaluar_regresor(modelo, X_test_pares, y_test)

    return {
        "modelo": modelo,
        "meta": meta,
        "stats": stats,
        "stats_filtros": stats_filtros,
        "metricas_regresor": metricas_regresor,
        "columnas_estado": list(X_estado.columns),
        "mapa_acciones": mapa_acciones,
        "representacion_accion": representacion_accion,
        "ids_acciones": ids_acciones,
        "df_test": df_test,
        "X_test_estado": X_test_estado,
    }