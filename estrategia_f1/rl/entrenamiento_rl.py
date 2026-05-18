"""
entrenamiento_rl.py

Entrenamiento del aproximador offline de valor Q(s,a)
para la recomendación de estrategias de neumáticos.

Incluye:
- La configuración del entrenamiento y de las rutas de caché.
- La construcción del dataset de pares estado-acción-recompensa.
- La construcción del aproximador supervisado de Q(s,a).
- La separación temporal por carreras entre entrenamiento y test.
- La aplicación opcional de filtros solo sobre el conjunto de entrenamiento.
- La gestión de caché del split, del dataset de pares y del modelo entrenado.
"""

# IMPORTS
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.base import RegressorMixin
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

# CONFIGURACIÓN Y PERSISTENCIA DEL ENTRENAMIENTO -----------------------------------------------------------------------
@dataclass(frozen=True)
class ConfiguracionEntrenamientoRL:
    """
    Configuración utilizada durante el entrenamiento
    del aproximador offline de valor Q(s,a).

    Atributos
    ----------
    seed : int
        Semilla aleatoria utilizada para garantizar
        reproducibilidad experimental.
    test_size : float
        Proporción del conjunto reservada para evaluación.
    k_acciones_muestreo : int
        Número de acciones muestreadas por observación
        durante la construcción del dataset de pares (s,a).
    modelo_q : str
        Nombre del aproximador de Q(s,a) a entrenar
        ("hist_gb", "ridge", "random_forest" o "mlp").
    modelo_q_params : dict[str, Any] | None
        Hiperparámetros específicos del aproximador.
    """
    seed: int
    test_size: float
    k_acciones_muestreo: int
    modelo_q: str = "hist_gb"
    modelo_q_params: dict[str, Any] | None = None

@dataclass(frozen=True)
class DireccionesRL:
    """
    Rutas utilizadas para guardar y recuperar los
    artefactos del entrenamiento offline.

    Atributos
    ----------
    ruta_pares : Path
        Ruta donde se almacena el dataset de pares
        estado-acción-recompensa utilizado para
        entrenar el aproximador Q(s,a).
    ruta_modelo : Path
        Ruta donde se almacena el aproximador
        de valor entrenado.
    ruta_meta : Path
        Ruta donde se almacenan metadatos del
        entrenamiento, firmas de caché y estadísticas.
    """
    ruta_pares: Path
    ruta_modelo: Path
    ruta_meta: Path

# CONSTRUCCIÓN DE PARES ------------------------------------------------------------------------------------------------
def construir_dataset_pares(df_sub: pd.DataFrame, matriz_estados: np.ndarray, mapa_acciones: dict[int, list[str]],
    representacion_accion: dict[int, np.ndarray], ids_acciones: np.ndarray, *,
    configuracionRL: ConfiguracionEntrenamientoRL) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Construye el dataset de pares estado-acción-recompensa.

    Para cada observación piloto-carrera, se muestrea un subconjunto
    de acciones válidas, se simula el tiempo asociado a cada una y
    se calcula una recompensa relativa respecto a la estrategia
    baseline de la misma observación.

    Parámetros
    ----------
    df_sub : pd.DataFrame
        Subconjunto del dataset experimental utilizado para construir
        los pares.
    matriz_estados : np.ndarray
        Matriz numérica con la representación del estado de cada
        observación.
    mapa_acciones : dict[int, list[str]]
        Diccionario que relaciona cada action_id con su estrategia
        de compuestos.
    representacion_accion : dict[int, np.ndarray]
        Diccionario con la representación numérica de cada acción.
    ids_acciones : np.ndarray
        Identificadores de acciones candidatas del espacio discreto.
    configuracionRL : ConfiguracionEntrenamientoRL
        Configuración del entrenamiento basado en valor.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict]
        Tupla formada por:
        - matriz X con la concatenación del estado y la acción.
        - vector y con la recompensa simulada.
        - diccionario de estadísticas con filas y pares omitidos.
    """
    # Generador reproducible para muestrear acciones candidatas
    gen_aleatorio = np.random.default_rng(configuracionRL.seed)

    X: list[np.ndarray] = []
    y: list[float] = []

    filas_omitidas = 0
    pares_omitidos = 0

    for i in tqdm(range(len(df_sub)), desc="Construyendo pares (s,a)"):
        fila = df_sub.iloc[i]
        estado = matriz_estados[i]

        # La estrategia baseline define la referencia común para calcular recompensas relativas
        baseline = elegir_estrategia_baseline(fila)
        if baseline is None:
            filas_omitidas += 1
            continue

        # Si la referencia no puede simularse, la observación completa no es útil para generar pares comparables
        tiempo_carrera_baseline = simular_tiempo_carrera(fila, baseline)
        if not np.isfinite(tiempo_carrera_baseline):
            filas_omitidas += 1
            continue

        # Se muestrea un subconjunto de acciones para limitar el coste computacional de simular el espacio
        if configuracionRL.k_acciones_muestreo >= len(ids_acciones):
            acciones_muestreadas = ids_acciones
        else:
            acciones_muestreadas = gen_aleatorio.choice(
                ids_acciones,
                size=configuracionRL.k_acciones_muestreo,
                replace=False,
            )

        # Solo se evalúan estrategias compatibles con los compuestos disponibles en la observación
        disponibles = compuestos_disponibles(fila)

        for accion_id in acciones_muestreadas:
            estrategia = estrategia_desde_accion_id(int(accion_id), mapa_acciones)

            # Las acciones no simulables se registran, pero no se incorporan al dataset de entrenamiento
            if not set(estrategia).issubset(disponibles):
                pares_omitidos += 1
                continue

            # Una recompensa positiva implica mejora frente al baseline
            tiempo_carrera_accion = simular_tiempo_carrera(fila, estrategia)
            if not np.isfinite(tiempo_carrera_accion):
                pares_omitidos += 1
                continue

            # Recompensa positiva implica mejora frente al baseline
            recompensa = -(tiempo_carrera_accion - tiempo_carrera_baseline)

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


# MODELO ---------------------------------------------------------------------------------------------------------------
def construir_modelo_q(*, nombre: str, seed: int, params: dict[str, Any] | None = None) -> RegressorMixin:
    """
    Construye el aproximador de valor Q(s,a).

    Selecciona e inicializa el modelo de regresión encargado
    de aproximar la recompensa esperada asociada a cada par
    estado-acción.

    Parámetros
    ----------
    nombre : str
        Identificador del modelo a construir
        ("hist_gb", "ridge", "random_forest" o "mlp").
    seed : int
        Semilla aleatoria utilizada para garantizar
        reproducibilidad experimental.
    params : dict[str, Any] | None
        Hiperparámetros adicionales específicos del modelo.
        Si es None, se utilizan únicamente los parámetros
        por defecto.

    Returns
    -------
    RegressorMixin
        Instancia del modelo de regresión configurada
        y lista para entrenamiento.
    """
    nombre = str(nombre).strip().lower()
    params = dict(params or {})

    # Construcción del aproximador según el modelo seleccionado
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
            early_stopping=True,
        )
        base.update(params)
        return MLPRegressor(**base)

    raise ValueError("El modelo usado no es reconocido. Usa: 'hist_gb', 'ridge', 'random_forest', 'mlp'.")


def entrenar_modelo_q(X: np.ndarray, y: np.ndarray, *, configuracionRL: ConfiguracionEntrenamientoRL) -> RegressorMixin:
    """
        Construye, prepara y entrena el aproximador de valor Q(s,a).

        Parámetros
        ----------
        X : np.ndarray
            Matriz de entrada formada por pares estado-acción.
        y : np.ndarray
            Recompensas objetivo asociadas a cada par.
        configuracionRL : ConfiguracionEntrenamientoRL
            Configuración del entrenamiento basado en valor.

        Returns
        -------
        RegressorMixin
            Aproximador Q(s,a) entrenado y listo para inferencia.
        """

    # Se construye el aproximador configurado externamente
    modelo_base = construir_modelo_q(
        nombre=configuracionRL.modelo_q,
        seed=configuracionRL.seed,
        params=configuracionRL.modelo_q_params,
    )

    # Algunos aproximadores lineales o neuronales son sensibles a la escala de las variables de entrada
    modelos_que_escalan = {"mlp", "ridge"}

    if configuracionRL.modelo_q in modelos_que_escalan:
        modelo = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", modelo_base),
        ])
    else:
        # Los modelos basados en árboles no requieren normalización
        modelo = modelo_base

    # Entrenamiento del aproximador sobre pares estado-acción
    modelo.fit(X, y)

    return modelo

def evaluar_regresor(modelo: RegressorMixin, X: np.ndarray, y: np.ndarray) -> dict:
    """
    Evalúa el rendimiento predictivo del aproximador Q(s,a).

    Parámetros
    ----------
    modelo : RegressorMixin
        Aproximador de valor previamente entrenado.
    X : np.ndarray
        Matriz de entrada con pares estado-acción.
    y : np.ndarray
        Recompensas reales utilizadas como referencia.

    Returns
    -------
    dict
        Diccionario con métricas de regresión:
        - mae: error absoluto medio.
        - r2: coeficiente de determinación.
    """
    pred = modelo.predict(X)
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }

def _n_features_modelo(modelo) -> int | None:
    """
     Recupera el número de variables esperado por un modelo.

     Compatible tanto con estimadores directos de scikit-learn
     como con pipelines que encapsulan el regresor final.

     Parámetros
     ----------
     modelo : Any
         Modelo entrenado o pipeline de scikit-learn.

     Returns
     -------
     int | None
         Número de variables de entrada esperadas por el modelo.
         Devuelve None si esta información no está disponible.
     """
    try:
        if hasattr(modelo, "named_steps") and "reg" in modelo.named_steps:
            est = modelo.named_steps["reg"]
        else:
            est = modelo

        return int(getattr(est, "n_features_in_", None)) if getattr(est, "n_features_in_", None) is not None else None
    except Exception:
        return None


# ENTRENAMIENTO --------------------------------------------------------------------------------------------------------
def entrenar_rl_offline(df: pd.DataFrame, *, configuracionRL: ConfiguracionEntrenamientoRL, paths: DireccionesRL,
    aplicar_filtros: bool = True) -> dict:
    """
    Entrena (o carga) el aproximador offline de valor Q(s,a).

    El pipeline construye un split temporal común por carreras,
    aplica opcionalmente filtros solo sobre el conjunto de
    entrenamiento, genera el dataset de pares estado-acción-
    recompensa y entrena el aproximador de valor utilizando
    mecanismos de caché para reutilizar artefactos ya generados.

    Parámetros
    ----------
    df : pd.DataFrame
        Dataset experimental completo.
    configuracionRL : ConfiguracionEntrenamientoRL
        Configuración del entrenamiento basado en valor.
    paths : DireccionesRL
        Rutas utilizadas para persistir caché, pares y modelo.
    aplicar_filtros : bool, optional
        Si es True, aplica filtrado únicamente sobre el
        conjunto de entrenamiento.

    Returns
    -------
    dict
        Diccionario con:
        - modelo entrenado.
        - metadatos de caché.
        - estadísticas de construcción de pares.
        - estadísticas de filtrado.
        - métricas del regresor.
        - representación de estados y acciones.
        - conjunto de test preprocesado.
    """
    paths.ruta_meta.parent.mkdir(parents=True, exist_ok=True)
    paths.ruta_pares.parent.mkdir(parents=True, exist_ok=True)
    paths.ruta_modelo.parent.mkdir(parents=True, exist_ok=True)

    # El split temporal se construye siempre sobre un dataset común para permitir comparaciones justas entre variantes
    df_base = df.copy().reset_index(drop=True)

    if len(df_base) == 0:
        raise ValueError("El dataset base quedó vacío en RL.")

    # Mapa de acciones y representación numérica de la acción
    mapa_acciones = construir_mapa_acciones()
    ids_acciones = np.array(sorted(mapa_acciones.keys()), dtype=int)
    representacion_accion = precomputar_features_acciones(mapa_acciones)

    # El estado base solo se utiliza para fijar la estructura de columnas y calcular una firma estable del split temporal
    X_estado_base = construir_estado_df(
        df_base,
        columnas=ESTADO_COLS,
        columnas_excluir=[],
        imputar_numericas=True,
    )

    hash_dataset_base = calcular_hash_dataset(df_base)

    stats_split = {
        "aplicado": False,
        "tipo_pipeline": "split_base_rl",
        "n_inicial": int(len(df_base)),
        "n_final": int(len(df_base)),
        "hash_dataset": hash_dataset_base,
    }

    # La firma del split depende únicamente del dataset base y de la definición del estado
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

    meta_split_reutilizable = False
    meta = None

    # Si existe una partición previa compatible, se reutiliza
    if paths.ruta_meta.exists():
        meta = joblib.load(paths.ruta_meta)
        firma_anterior = meta.get("signature_split")

        if firma_anterior != firma_split:
            print("Se han detectado cambios en el dataset base/columnas base del split.")
            print(f"Firma split anterior: {firma_anterior}")
            print(f"Firma split actual  : {firma_split}")
            print("Invalidando caché de meta, pares y modelo...")
            invalidar_archivos(paths.ruta_meta, paths.ruta_pares, paths.ruta_modelo)
            meta = None
        else:
            meta_split_reutilizable = True
            print("Se ha encontrado un split anterior, usando cache existente de train/test.")

    if meta_split_reutilizable and meta is not None:
        idx_train = meta["idx_train"]
        idx_test = meta["idx_test"]
    else:
        print("Generando nuevo train/test split...")
        # El split se realiza cronológicamente por carrera para evitar fuga temporal entre GPs de train y test
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

    # El conjunto de test permanece siempre fijo y sin filtrar, independientemente de la variante
    df_train_base = df_base.iloc[idx_train].reset_index(drop=True)
    df_test = df_base.iloc[idx_test].reset_index(drop=True)

    # Los filtros solo se aplican sobre train para mejorar la robustez del aprendizaje sin alterar la evaluación final
    if aplicar_filtros:
        print("Aplicando filtros...")
        df_train, stats_filtros = filtrar_dataset(df_train_base, tipo_pipeline="rl")

        if len(df_train) == 0:
            raise ValueError("El train quedó vacío después del filtrado RL.")

        print(f"Filtrado RL en train completado: {len(df_train):,} filas restantes\n")
    else:
        df_train = df_train_base.copy()
        stats_filtros = {
            "aplicado": False,
            "n_inicial": int(len(df_train_base)),
            "n_final": int(len(df_train_base)),
            "tipo_pipeline": "ninguno",
        }

    hash_dataset_train = calcular_hash_dataset(df_train)
    stats_filtros["hash_dataset"] = hash_dataset_train

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

    # La firma del modelo refleja el train, incluyendo filtros, columnas y configuración del clasificador
    firma_entrenamiento = firma_entrenamiento_rl(
        df_hash=hash_dataset_train,
        columnas_estado=list(X_train_estado_df.columns),
        configuracionRL=configuracionRL,
        stats_filtros=stats_filtros,
    )

    firma_train_anterior = meta.get("signature_train_rl")

    if firma_train_anterior == firma_entrenamiento:
        cache_entrenamiento_reutilizable = True
    else:
        if paths.ruta_pares.exists() or paths.ruta_modelo.exists():
            print("Detectados cambios en train/configuración RL. Regenerando pares y modelo...")
            invalidar_archivos(paths.ruta_pares, paths.ruta_modelo)
        cache_entrenamiento_reutilizable = False

    # Se separan automáticamente variables numéricas y categóricas para aplicar el preprocesado adecuado a cada tipo
    num_cols = [c for c in X_train_estado_df.columns if pd.api.types.is_numeric_dtype(X_train_estado_df[c])]
    cat_cols = [c for c in X_train_estado_df.columns if c not in num_cols]

    # La imputación y codificación se ajustan únicamente con train y luego se aplican sobre test
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

    # Algunos clasificadores de scikit-learn requieren matrices densas para completar correctamente el entrenamiento
    if hasattr(X_train_estado, "toarray"):
        X_train_estado = X_train_estado.toarray()
        X_test_estado = X_test_estado.toarray()

    X_train_estado = X_train_estado.astype(np.float32, copy=False)
    X_test_estado = X_test_estado.astype(np.float32, copy=False)

    # Los pares solo se reutilizan si el train y la configuración coinciden exactamente con la firma previa
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

    # El modelo solo se reutiliza si el train y la configuración coinciden exactamente con la firma previa
    if paths.ruta_modelo.exists() and cache_entrenamiento_reutilizable:
        print("Cargando modelo desde cache...")
        modelo = joblib.load(paths.ruta_modelo)
        n_expected = _n_features_modelo(modelo)
        n_actual = int(X_train_pares.shape[1])

        if n_expected is not None and n_expected != n_actual:
            print(f"[CACHE INVÁLIDO] Modelo espera {n_expected} features, pero ahora hay {n_actual}. Reentrenando...")
            invalidar_archivos(paths.ruta_modelo)
            modelo = entrenar_modelo_q(X_train_pares, y_train, configuracionRL=configuracionRL)
            joblib.dump(modelo, paths.ruta_modelo)
    else:
        print("Entrenando nuevo modelo...")
        modelo = entrenar_modelo_q(X_train_pares, y_train, configuracionRL=configuracionRL)
        joblib.dump(modelo, paths.ruta_modelo)

        # Los metadatos del entrenamiento se actualizan únicamente cuando se genera un modelo nuevo
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