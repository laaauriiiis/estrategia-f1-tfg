"""
features.py

Construcción de representaciones numéricas de acciones estratégicas.

Este módulo contiene la lógica necesaria para:
- Transformar estrategias de neumáticos en vectores numéricos aptos para modelos de aprendizaje.
- Codificar cada acción mediante una representación estructurada basada en la secuencia de compuestos.
- Precalcular las características de todas las acciones del espacio discreto para reutilizarlas durante entrenamiento y
    evaluación.

A diferencia del mapa de acciones, que relaciona cada identificador con su representación simbólica:

    action_id = 17 -> ["SOFT", "HARD", "MEDIUM"]

este módulo transforma esa misma acción en una representación numérica (feature vector) que pueda ser utilizada por los
modelos:

    ["SOFT", "HARD", "MEDIUM"] -> [1,0,0, 0,0,1, 0,1,0, 0,0,0, 3,2]

donde la codificación incluye la secuencia de compuestos mediante one-hot por stint, el número total de stints y el
número de paradas.

Estas representaciones permiten que los modelos basados en valor aprendan la relación entre un estado de carrera y una
acción candidata, aproximando la función Q(s, a).
"""

# IMPORTS
from __future__ import annotations
import numpy as np
from estrategia_f1.acciones import (
    limpiar_compuestos,
    estrategia_valida,
)
from estrategia_f1.config import (
    MAX_STINTS,
    NUM_COMPUESTOS,
    COMPUESTOS_A_INDICE,
)

# FEATURES DE ACCIONES -------------------------------------------------------------------------------------------------
def accion_a_features(estrategia: list[str], max_stints: int = MAX_STINTS) -> np.ndarray:
    """
    Convierte una estrategia de neumáticos en una representación
    numérica utilizada por los modelos de aprendizaje.

    Parámetros
    ----------
    estrategia : list[str]
        Secuencia ordenada de compuestos que representa
        una acción estratégica.
    max_stints : int, optional
        Número máximo de stints considerados en la codificación.

    Returns
    -------
    np.ndarray
        Vector numérico que describe la acción mediante:
        - Codificación one-hot por stint y compuesto.
        - Número total de stints.
        - Número total de paradas.

        Ejemplo:
            ["SOFT", "HARD"]
        se transforma en:
            [1,0,0, 0,0,1, 0,0,0, 0,0,0, 2,1]
        donde:
        - [1,0,0] representa SOFT en el primer stint.
        - [0,0,1] representa HARD en el segundo stint.
        - Los stints no utilizados permanecen a cero.
        - Los dos últimos valores representan número de stints (2) y número de paradas (1).

        Si la estrategia no es válida, devuelve un vector relleno con valores np.nan.
    """
    estrategia = limpiar_compuestos(estrategia)

    if not estrategia_valida(estrategia):
        return np.full((max_stints * NUM_COMPUESTOS + 2,), np.nan, dtype=np.float32)

    num_stints = len(estrategia)
    num_paradas = max(0, num_stints - 1)

    # Cada fila representa un stint y cada columna un compuesto
    matriz = np.zeros((max_stints, NUM_COMPUESTOS), dtype=np.float32)
    for i in range(min(num_stints, max_stints)):
        idx = COMPUESTOS_A_INDICE.get(estrategia[i])
        if idx is not None:
            matriz[i, idx] = 1.0

    # La matriz se aplana para obtener un único vector compatible con los modelos de aprendizaje
    feat = matriz.flatten()
    return np.concatenate([feat, np.array([num_stints, num_paradas], dtype=np.float32)])


def precomputar_features_acciones(mapa_acciones: dict[int, list[str]]) -> dict[int, np.ndarray]:
    """
    Precalcula la representación numérica de todas las acciones
    del espacio discreto.

    Parameters
    ----------
    mapa_acciones : dict[int, list[str]]
        Diccionario que asocia cada identificador de acción
        (action_id) con su estrategia de neumáticos
        correspondiente.

    Returns
    -------
    dict[int, np.ndarray]
        Diccionario que asocia cada action_id con su vector
        de características numéricas.

        Esta estructura permite reutilizar la codificación
        de acciones durante entrenamiento y evaluación,
        evitando recalcularla repetidamente.
    """
    return {aid: accion_a_features(seq) for aid, seq in mapa_acciones.items()}