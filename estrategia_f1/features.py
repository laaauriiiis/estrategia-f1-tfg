"""
features.py
TODO + separar/juntar
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from estrategia_f1.acciones import (
    limpiar_compuestos,
    estrategia_valida,
    acciones_validas_para_fila,
)

from estrategia_f1.config import (
    MAX_STINTS,
    NUM_COMPUESTOS,
    COMPUESTOS_A_INDICE,
)

# Features de acciones--------------------------------------------------------------------------------------------------
def accion_a_features(estrategia: list[str], max_stints: int = MAX_STINTS) -> np.ndarray:
    """
    Codifica una estrategia como:
      - One-hot por stint y compuesto: (max_stints x NUM_COMPUESTOS)
      - num_stints
      - num_paradas
    E.g. ["SOFT", "HARD"] -> [1,0,0, 0,0,1, 0,0,0, 0,0,0, 2,1]
    """
    estrategia = limpiar_compuestos(estrategia)

    if not estrategia_valida(estrategia):
        return np.full((max_stints * NUM_COMPUESTOS + 2,), np.nan, dtype=np.float32)

    num_stints = len(estrategia)
    num_paradas = max(0, num_stints - 1)

    matriz = np.zeros((max_stints, NUM_COMPUESTOS), dtype=np.float32)
    for i in range(min(num_stints, max_stints)):
        idx = COMPUESTOS_A_INDICE.get(estrategia[i])
        if idx is not None:
            matriz[i, idx] = 1.0

    feat = matriz.flatten()
    return np.concatenate([feat, np.array([num_stints, num_paradas], dtype=np.float32)])


def precomputar_features_acciones(mapa_acciones: dict[int, list[str]]) -> dict[int, np.ndarray]:
    """
    Devuelve {accion_id: feature_vector}.
    """
    return {aid: accion_a_features(seq) for aid, seq in mapa_acciones.items()}


def acciones_validas_y_features_para_fila(fila: pd.Series, mapa_acciones: dict[int, list[str]],
        features_acciones: dict[int, np.ndarray] | None = None) -> list[tuple[int, np.ndarray]]:
    """
    Devuelve [(accion_id, feat_accion), ...] solo para acciones compatibles.
    """
    ids = acciones_validas_para_fila(fila, mapa_acciones)

    out: list[tuple[int, np.ndarray]] = []
    for accion_id in ids:
        if features_acciones is not None:
            feat = features_acciones[accion_id]
        else:
            feat= accion_a_features(mapa_acciones[accion_id])
        out.append((accion_id, feat))

    return out

