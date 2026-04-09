"""
cache_utils.py
Utilidades comunes para validar e invalidar cachés de entrenamiento.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def calcular_hash_dataset(df) -> str:
    """
    Calcula un hash estable del dataset final usado para el split/entrenamiento.

    Se intenta usar columnas identificadoras lógicas y estables. Si no existen,
    se lanza error para evitar hashes ambiguos o dependientes del orden de columnas.
    """
    candidatos = [
        ["season", "race_id", "driver_number"],
        ["season", "race_id", "driver_id"],
        ["season", "race_id", "constructor_id", "driver_id"],
        ["race_id"],
    ]

    id_cols = None
    for cols in candidatos:
        if all(col in df.columns for col in cols):
            id_cols = cols
            break

    if id_cols is None:
        raise KeyError(
            "No se encontraron columnas identificadoras estables para calcular "
            "el hash del dataset."
        )

    df_ids = df[id_cols].copy().sort_values(id_cols).reset_index(drop=True)
    contenido = df_ids.to_csv(index=False)
    return hashlib.md5(contenido.encode("utf-8")).hexdigest()


def _normalizar_para_json(obj: Any) -> Any:
    """
    Convierte objetos no serializables a una forma estable para json.dumps.
    """
    if isinstance(obj, dict):
        return {str(k): _normalizar_para_json(v) for k, v in sorted(obj.items(), key=lambda x: str(x[0]))}
    if isinstance(obj, (list, tuple)):
        return [_normalizar_para_json(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_normalizar_para_json(v) for v in obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _hash_payload(payload: dict[str, Any]) -> str:
    """
    Serializa un payload de forma estable y devuelve un MD5.
    """
    texto = json.dumps(
        _normalizar_para_json(payload),
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.md5(texto.encode("utf-8")).hexdigest()


def firma_entrenamiento_ml(
    *,
    df_hash: str,
    columnas_estado: list[str],
    configuracionML,
    stats_filtros: dict[str, Any],
) -> str:
    """
    Genera una firma única del experimento ML.
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


def firma_entrenamiento_rl(
    *,
    df_hash: str,
    columnas_estado: list[str],
    configuracionRL,
    stats_filtros: dict[str, Any],
) -> str:
    """
    Genera una firma única del experimento RL.
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


def invalidar_archivos(*paths: Path) -> None:
    """
    Borra archivos de caché si existen.
    """
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            # No interrumpimos el flujo por un caché ya ausente o bloqueado;
            # el código de entrenamiento podrá regenerarlo después si hace falta.
            pass