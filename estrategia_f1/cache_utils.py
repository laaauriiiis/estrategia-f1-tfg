"""
cache_utils.py

Gestión de firmas experimentales y validación de cachés de entrenamiento.

Este módulo contiene la lógica necesaria para:
- Calcular identificadores hash estables a partir del dataset utilizado en cada experimento.
- Generar firmas únicas de configuración para los pipelines supervisado y basado en valor.
- Invalidar archivos de caché cuando cambia la configuración del entrenamiento o los datos de entrada.
"""

# IMPORTS
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any

# HASH DEL DATASET -----------------------------------------------------------------------------------------------------
def calcular_hash_dataset(df) -> str:
    """
    Calcula una firma hash estable a partir del dataset utilizado
    en entrenamiento y evaluación.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con las observaciones del experimento.

    Returns
    -------
    str
        Cadena hexadecimal que identifica de forma única el
        contenido estructural del dataset utilizado.

        El hash se calcula a partir de columnas identificadoras
        estables de cada observación, independientes del orden
        original de las filas.
    """

    # Se prueban distintas combinaciones de identificadores según la disponibilidad del dataset procesado
    candidatos = [
        ["season", "race_id", "driver_number"],
        ["season", "race_id", "driver_id"],
        ["season", "race_id", "constructor_id", "driver_id"],
        ["race_id"],
    ]

    # Se selecciona la primera combinación de columnas que permite identificar de forma estable cada observación
    id_cols = None
    for cols in candidatos:
        if all(col in df.columns for col in cols):
            id_cols = cols
            break

    if id_cols is None:
        raise KeyError(
            "No se encontraron columnas identificadoras estables para calcular el hash del dataset."
        )

    # Se ordenan las observaciones antes del hash para garantizar reproducibilidad independientemente del orden
    df_ids = df[id_cols].copy().sort_values(id_cols).reset_index(drop=True)
    contenido = df_ids.to_csv(index=False)
    return hashlib.md5(contenido.encode("utf-8")).hexdigest()

# SERIALIZACIÓN ESTABLE ------------------------------------------------------------------------------------------------
def _normalizar_para_json(obj: Any) -> Any:
    """
    Convierte objetos complejos a una representación estable
    y serializable en formato JSON.

    Parámetros
    ----------
    obj : Any
        Objeto que se desea convertir a un formato compatible
        con serialización JSON.

    Returns
    -------
    Any
        Objeto transformado a una representación determinista
        y serializable, adecuada para generar firmas hash
        reproducibles.
    """
    # Los diccionarios se ordenan por clave para garantizar una serialización independiente del orden de inserción
    if isinstance(obj, dict):
        return {str(k): _normalizar_para_json(v) for k, v in sorted(obj.items(), key=lambda x: str(x[0]))}
    # Las secuencias mantienen su orden original
    if isinstance(obj, (list, tuple)):
        return [_normalizar_para_json(v) for v in obj]
    # Los conjuntos se ordenan explícitamente para obtener una representación determinista
    if isinstance(obj, set):
        return sorted(_normalizar_para_json(v) for v in obj)
    # Las rutas se convierten a texto para permitir su serialización directa
    if isinstance(obj, Path):
        return str(obj)
    return obj

def _hash_payload(payload: dict[str, Any]) -> str:
    """
    Genera una firma hash a partir de una configuración
    serializable.

    Parámetros
    ----------
    payload : dict[str, Any]
        Diccionario con la información que define un
        experimento o configuración concreta.

    Returns
    -------
    str
        Cadena hexadecimal generada mediante MD5 a partir
        de una serialización JSON estable del contenido.
    """

    # La serialización se realiza de forma determinista para que configuraciones equivalentes generen siempre
    # la misma firma
    texto = json.dumps(
        _normalizar_para_json(payload),
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.md5(texto.encode("utf-8")).hexdigest()

# FIRMAS DE ENTRENAMIENTO ----------------------------------------------------------------------------------------------
def firma_entrenamiento_ml(*, df_hash: str, columnas_estado: list[str], configuracionML, stats_filtros: dict[str, Any]) -> str:
    """
    Genera una firma única para un experimento de aprendizaje
    supervisado.

    Parámetros
    ----------
    df_hash : str
        Identificador hash del dataset utilizado.
    columnas_estado : list[str]
        Variables del estado utilizadas como entrada
        del modelo.
    configuracionML : Any
        Objeto con la configuración del experimento,
        incluyendo semilla, partición y parámetros
        del modelo supervisado.
    stats_filtros : dict[str, Any]
        Información sobre los filtros aplicados durante
        la preparación del dataset.

    Returns
    -------
    str
        Cadena hexadecimal que identifica de forma única
        la configuración completa del experimento.
    """
    payload = {
        "pipeline": "ml",
        "df_hash": df_hash,
        "columnas_estado": list(columnas_estado),
        "seed": configuracionML.seed,
        "test_size": configuracionML.test_size,
        "modelo": configuracionML.modelo,
        "modelo_params": configuracionML.modelo_params,
        "filtros_aplicados": stats_filtros.get("aplicado", True),
        "tipo_pipeline": stats_filtros.get("tipo_pipeline", "ml"),
    }
    return _hash_payload(payload)

def firma_entrenamiento_rl(*, df_hash: str, columnas_estado: list[str], configuracionRL, stats_filtros: dict[str, Any]) -> str:
    """
    Genera una firma única para un experimento de aprendizaje
    basado en valor.

    Parámetros
    ----------
    df_hash : str
        Identificador hash del dataset utilizado.
    columnas_estado : list[str]
        Variables del estado utilizadas como entrada
        del modelo.
    configuracionRL : Any
        Objeto con la configuración del experimento,
        incluyendo semilla, partición y parámetros
        del modelo basado en valor.
    stats_filtros : dict[str, Any]
        Información sobre los filtros aplicados durante
        la preparación del dataset.

    Returns
    -------
    str
        Cadena hexadecimal que identifica de forma única
        la configuración completa del experimento.
    """
    payload = {
        "pipeline": "rl",
        "df_hash": df_hash,
        "columnas_estado": list(columnas_estado),
        "seed": configuracionRL.seed,
        "test_size": configuracionRL.test_size,
        "k_acciones_muestreo": configuracionRL.k_acciones_muestreo,
        "modelo_q": configuracionRL.modelo_q,
        "modelo_q_params": configuracionRL.modelo_q_params,
        "filtros_aplicados": stats_filtros.get("aplicado", True),
        "tipo_pipeline": stats_filtros.get("tipo_pipeline", "rl"),
    }
    return _hash_payload(payload)

# INVALIDACIÓN DE CACHÉS -----------------------------------------------------------------------------------------------
def invalidar_archivos(*paths: Path) -> None:
    """
    Elimina archivos de caché asociados a experimentos previos.

    Parámetros
    ----------
    *paths : Path
        Rutas de los archivos de caché que deben invalidarse.

    Returns
    -------
    None
        Esta función no devuelve ningún valor. Su objetivo
        es eliminar archivos obsoletos antes de regenerar
        nuevos artefactos de entrenamiento.
    """
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass