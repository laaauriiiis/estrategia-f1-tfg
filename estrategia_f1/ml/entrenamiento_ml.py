"""
entrenamiento_ml.py

Entrenamiento del modelo supervisado para la recomendación de estrategias.

Incluye:
- La configuración del entrenamiento y de las rutas de caché.
- La construcción del clasificador seleccionado.
- El preprocesado de variables numéricas y categóricas.
- La separación temporal por carreras entre entrenamiento y test.
- La aplicación opcional de filtros solo sobre el conjunto de entrenamiento.
- La gestión de caché del split y del modelo entrenado.
"""

# IMPORTS
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
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

# CONFIGURACIÓN Y PERSISTENCIA DEL ENTRENAMIENTO -----------------------------------------------------------------------
@dataclass(frozen=True)
class ConfiguracionEntrenamientoML:
    """
      Configuración utilizada durante el entrenamiento del modelo supervisado.

      Atributos
      ----------
      seed : int
          Semilla aleatoria utilizada para garantizar reproducibilidad.
      test_size : float
          Proporción del conjunto reservada para evaluación.
      modelo : str
          Nombre del modelo de clasificación a entrenar
          ("hist_gb", "logreg", "random_forest" o "mlp").
      modelo_params : dict[str, Any] | None
          Hiperparámetros específicos del modelo.
      """
    seed: int
    test_size: float
    modelo: str = "hist_gb"
    modelo_params: dict[str, Any] | None = None

@dataclass(frozen=True)
class DireccionesML:
    """
    Rutas utilizadas para guardar y recuperar artefactos del entrenamiento.

    Atributos
    ----------
    ruta_modelo : Path
        Ruta donde se almacena el modelo entrenado.
    ruta_meta : Path
        Ruta donde se almacenan metadatos del entrenamiento,
        firmas de caché y estadísticas.
    """
    ruta_modelo: Path
    ruta_meta: Path

# MODELO ---------------------------------------------------------------------------------------------------------------
def construir_modelo(*, nombre: str, seed: int, params: dict[str, Any] | None = None) -> ClassifierMixin:
    """
    Construye una instancia del clasificador supervisado seleccionado.

    Parámetros
    ----------
    nombre : str
        Identificador del modelo a construir
        ("hist_gb", "logreg", "random_forest" o "mlp").
    seed : int
        Semilla aleatoria utilizada para garantizar reproducibilidad.
    params : dict[str, Any] | None
        Hiperparámetros adicionales específicos del modelo.
        Si es None, se utilizan únicamente los parámetros por defecto.

    Returns
    -------
    ClassifierMixin
        Instancia del clasificador de scikit-learn configurada
        y lista para entrenamiento.
    """
    nombre = str(nombre).strip().lower()
    params = dict(params or {})

    # Construcción de cada modelo según el nombre inserido
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
    Construye, prepara y entrena el clasificador supervisado.

    Parámetros
    ----------
    X : np.ndarray
        Matriz de características de entrada.
    y : np.ndarray
        Etiquetas objetivo correspondientes al action_id.
    configuracionML : ConfiguracionEntrenamientoML
        Configuración del modelo y de sus hiperparámetros.

    Returns
    -------
    ClassifierMixin
        Modelo entrenado y listo para inferencia.
    """
    # Se recuperan los hiperparámetros definidos externamente en config.py
    params_originales = configuracionML.modelo_params or {}

    modelo_base = construir_modelo(
        nombre=configuracionML.modelo,
        seed=configuracionML.seed,
        params=params_originales,
    )

    # Algunos modelos son sensibles a la escala de las variables
    modelos_que_escalan = {"mlp", "logreg"}

    if configuracionML.modelo in modelos_que_escalan:
        modelo = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", modelo_base),
        ])
    else:
        # Los modelos basados en árboles no requieren normalización
        modelo = modelo_base

    # Entrenamiento del modelo sobre los datos de entrenamiento
    modelo.fit(X, y)
    return modelo

# ENTRENAMIENTO --------------------------------------------------------------------------------------------------------
def entrenar_ml(df: pd.DataFrame, *, configuracionML: ConfiguracionEntrenamientoML, paths: DireccionesML,
                aplicar_filtros: bool = True) -> dict:
    """
    Entrena o reutiliza el modelo supervisado de recomendación de estrategias.

    Esta función gestiona el flujo completo del entrenamiento supervisado:
    valida el dataset, genera o recupera el split temporal por carrera,
    aplica filtros únicamente sobre el conjunto de entrenamiento, construye
    los estados definitivos, preprocesa las variables y entrena o carga el
    modelo desde caché.

    Parámetros
    ----------
    df : pd.DataFrame
        Dataset de entrada con las observaciones piloto-carrera y la columna
        objetivo action_id.
    configuracionML : ConfiguracionEntrenamientoML
        Configuración del entrenamiento, incluyendo semilla, tamaño del test,
        modelo seleccionado e hiperparámetros.
    paths : DireccionesML
        Rutas donde se almacenan o recuperan el modelo entrenado y los
        metadatos del entrenamiento.
    aplicar_filtros : bool, default=True
        Indica si se deben aplicar filtros específicos de ML sobre el conjunto
        de entrenamiento. El conjunto de test se mantiene siempre sin filtrar
        para permitir comparaciones consistentes.

    Returns
    -------
    dict
        Diccionario con el modelo entrenado o cargado, metadatos del proceso,
        estadísticas de filtrado, columnas del estado, espacio de acciones,
        representación de acciones y conjunto de test preparado.
    """
    paths.ruta_meta.parent.mkdir(parents=True, exist_ok=True)
    paths.ruta_modelo.parent.mkdir(parents=True, exist_ok=True)

    # Mapa de acciones y representación numérica de la acción
    mapa_acciones = construir_mapa_acciones()
    ids_acciones = np.array(sorted(mapa_acciones.keys()), dtype=int)
    representacion_accion = precomputar_features_acciones(mapa_acciones)

    if "action_id" not in df.columns:
        raise KeyError("Falta columna 'action_id' en df para entrenar ML.")

    # Se conservan únicamente observaciones con (mínimo) una acción válida que pueda utilizarse como variable objetivo
    y_raw = pd.to_numeric(df["action_id"], errors="coerce")
    mask = np.isfinite(y_raw.to_numpy())

    # Dataset de referencia común para que todas las variantes compartan el mismo split
    df_base = df.loc[mask].copy().reset_index(drop=True)
    y_base = y_raw.loc[mask].astype(int).to_numpy()

    if len(df_base) == 0:
        raise ValueError("El dataset base se ha quedado vacío tras validar action_id.")

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
        "tipo_pipeline": "split_base",
        "n_inicial": int(len(df_base)),
        "n_final": int(len(df_base)),
        "hash_dataset": hash_dataset_base,
    }

    # La firma del split depende únicamente del dataset base y de la definición del estado
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
            print("Invalidando caché de meta y modelo...")
            invalidar_archivos(paths.ruta_meta, paths.ruta_modelo)
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

    # El conjunto de test permanece siempre fijo y sin filtrar, independientemente de la variante
    df_train_base = df_base.iloc[idx_train].reset_index(drop=True)
    df_test = df_base.iloc[idx_test].reset_index(drop=True)
    y_test = y_base[idx_test]

    # Los filtros solo se aplican sobre train para mejorar la robustez del aprendizaje sin alterar la evaluación final
    if aplicar_filtros:
        print("Aplicando filtros...")
        df_train, stats_filtros = filtrar_dataset(df_train_base, tipo_pipeline="ml")

        if len(df_train) == 0:
            raise ValueError("El train quedó vacío después del filtrado ML.")

        print(f"Filtrado completado: {len(df_train):,} filas restantes\n")
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
    firma_entrenamiento = firma_entrenamiento_ml(
        df_hash=hash_dataset_train,
        columnas_estado=list(X_train_estado_df.columns),
        configuracionML=configuracionML,
        stats_filtros=stats_filtros,
    )

    firma_modelo_anterior = meta.get("signature_modelo")

    if firma_modelo_anterior == firma_entrenamiento and paths.ruta_modelo.exists():
        modelo_reutilizable = True
    else:
        if paths.ruta_modelo.exists():
            print("Detectados cambios en train/configuración ML. Reentrenando modelo...")
            invalidar_archivos(paths.ruta_modelo)
        modelo_reutilizable = False

    # La variable objetivo se reconstruye desde el train final, por si se han eliminado observaciones
    y_train_raw = pd.to_numeric(df_train["action_id"], errors="coerce")
    y_train = y_train_raw.astype(int).to_numpy()

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

    # El modelo solo se reutiliza si el train y la configuración coinciden exactamente con la firma previa
    if modelo_reutilizable:
        print("Cargando modelo desde caché...")
        modelo = joblib.load(paths.ruta_modelo)
    else:
        print("Entrenando nuevo modelo...")
        modelo = entrenar_modelo(X_train_estado, y_train, configuracionML=configuracionML)
        joblib.dump(modelo, paths.ruta_modelo)

        # Los metadatos del entrenamiento se actualizan únicamente cuando se genera un modelo nuevo
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