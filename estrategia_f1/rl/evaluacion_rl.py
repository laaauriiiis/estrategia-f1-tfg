"""
evaluacion_rl.py

Evaluación de políticas basadas en aproximación de valor Q(s,a).

Incluye:
- La selección greedy de acciones mediante el aproximador Q.
- La evaluación de estrategias sobre el conjunto de test.
- La comparación frente a una baseline y un oracle simulado.
- El cálculo de métricas top-k y regret.
- La evaluación de políticas considerando únicamente
  acciones válidas según las restricciones de carrera.
"""

# IMPORTS
from __future__ import annotations
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.base import RegressorMixin
from estrategia_f1.sim.simulador import simular_tiempo_carrera
from estrategia_f1.acciones import (
    estrategia_desde_accion_id,
    acciones_validas_para_fila,
    elegir_estrategia_baseline,
)

# POLÍTICA -------------------------------------------------------------------------------------------------------------
def elegir_accion_modelo(modelo: RegressorMixin, x_estado: np.ndarray, ids_acciones: np.ndarray,
        representacion_accion: dict[int, np.ndarray]) -> tuple[int, np.ndarray]:
    """
    Selecciona la acción greedy según el aproximador Q(s,a).

    Evalúa todas las acciones candidatas para el estado actual
    y devuelve aquella con mayor valor Q predicho.

    Parámetros
    ----------
    modelo : RegressorMixin
        Aproximador de valor previamente entrenado.
    x_estado : np.ndarray
        Representación numérica del estado de carrera.
    ids_acciones : np.ndarray
        Identificadores de acciones candidatas a evaluar.
    representacion_accion : dict[int, np.ndarray]
        Representación numérica asociada a cada acción.

    Returns
    -------
    tuple[int, np.ndarray]
        Tupla formada por:
        - Identificador de la acción greedy seleccionada.
        - Valores Q predichos en el mismo orden que ids_acciones.
    """
    if len(ids_acciones) == 0:
        raise ValueError("ids_acciones está vacío")

    x_estado = np.asarray(x_estado, dtype=np.float32)

    # El mismo estado se replica para evaluar simultáneamente todas las acciones candidatas disponibles
    matriz_estado = np.repeat(x_estado[None, :], repeats=len(ids_acciones), axis=0).astype(np.float32, copy=False)
    matriz_acciones = np.stack([representacion_accion[int(a)] for a in ids_acciones], axis=0).astype(np.float32, copy=False)

    # Cada fila representa un par estado-acción: X = [estado || acción]
    X = np.concatenate([matriz_estado, matriz_acciones], axis=1)

    # El aproximador predice un valor Q por acción candidata
    q_values = np.asarray(modelo.predict(X)).reshape(-1)

    # Política greedy: seleccionar la acción con mayor Q
    idx_mejor = int(np.argmax(q_values))

    return int(ids_acciones[idx_mejor]), q_values

# EVALUACIÓN -----------------------------------------------------------------------------------------------------------
def evaluar_politica_rl(df: pd.DataFrame, X: pd.DataFrame, modelo: RegressorMixin, mapa_acciones: dict[int, list[str]],
        representacion_accion: dict[int, np.ndarray], *, topk: tuple[int, ...] = (3, 5), nombre_modelo: str | None = None) -> pd.DataFrame:
    """
    Evalúa una política basada en aproximación de valor Q(s,a).

    Para cada observación del conjunto de test, el modelo
    selecciona la acción greedy con mayor valor Q predicho
    entre las estrategias válidas disponibles. Posteriormente,
    la estrategia seleccionada se compara frente a:

    - una estrategia baseline.
    - un oracle obtenido mediante búsqueda exhaustiva.
    - métricas top-k basadas en el ranking de valores Q.

    Parámetros
    ----------
    df : pd.DataFrame
        Observaciones piloto-carrera del conjunto de evaluación.
    X : pd.DataFrame | np.ndarray
        Representación numérica de los estados evaluados.
    modelo : RegressorMixin
        Aproximador de valor previamente entrenado.
    mapa_acciones : dict[int, list[str]]
        Diccionario que relaciona cada action_id con su estrategia.
    representacion_accion : dict[int, np.ndarray]
        Representación numérica asociada a cada acción.
    topk : tuple[int, ...], optional
        Valores de k utilizados para las métricas top-k.
    nombre_modelo : str | None, optional
        Nombre descriptivo del modelo evaluado.

    Returns
    -------
    pd.DataFrame
        DataFrame con métricas de evaluación por observación,
        incluyendo baseline, policy, oracle y métricas top-k.
    """
    resultados: list[dict] = []

    for i in tqdm(range(len(df)), desc="Evaluando política"):
        fila = df.iloc[i]

        # Se recupera la representación numérica del estado
        if isinstance(X, np.ndarray):
            x_estado = X[i].astype(np.float32, copy=False)
        else:
            x_estado = X.iloc[i].to_numpy(dtype=np.float32, copy=False)

        # Caché local para evitar simular varias veces la misma acción dentro de la observación actual
        tiempos_cache: dict[int, float] = {}

        def tiempo_accion(accion: int) -> float:
            """
            Devuelve el tiempo simulado de esa acción usando caché.
            """
            accion = int(accion)
            if accion not in tiempos_cache:
                estrategia = estrategia_desde_accion_id(accion, mapa_acciones)
                tiempo = simular_tiempo_carrera(fila, estrategia)

                if np.isfinite(tiempo):
                    tiempos_cache[accion] = tiempo
                else:
                    tiempos_cache[accion] = np.nan
            return tiempos_cache[accion]

        # La baseline actúa como referencia común de comparación
        baseline = elegir_estrategia_baseline(fila)
        if baseline is None:
            continue

        tiempo_carrera_baseline = simular_tiempo_carrera(fila, baseline)
        if not np.isfinite(tiempo_carrera_baseline):
            continue

        # Solo se consideran acciones compatibles con los compuestos disponibles en carrera
        ids_validas = np.array(acciones_validas_para_fila(fila, mapa_acciones), dtype=int)
        if len(ids_validas) == 0:
            continue

        # Política greedy basada en el valor Q predicho
        try:
            accion_id_pi, q_values = elegir_accion_modelo(
                modelo,
                x_estado,
                ids_validas,
                representacion_accion,
            )
        except Exception:
            continue

        estrategia_pi = estrategia_desde_accion_id(int(accion_id_pi), mapa_acciones)

        # Tiempo asociado a la estrategia seleccionada
        tiempo_carrera_pi = tiempo_accion(int(accion_id_pi))
        if not np.isfinite(tiempo_carrera_pi):
            continue

        # Ranking descendente según valor Q
        order_desc = np.argsort(-q_values)
        ranking_ids = ids_validas[order_desc]

        # Oracle: mejor estrategia simulada entre todas las acciones válidas disponibles
        mejor_tiempo = np.inf
        accion_id_oracle: int | None = None

        for accion_id in ids_validas:
            t_accion_id = tiempo_accion(int(accion_id))
            if np.isfinite(t_accion_id) and t_accion_id < mejor_tiempo:
                mejor_tiempo = float(t_accion_id)
                accion_id_oracle = int(accion_id)

        if accion_id_oracle is None or not np.isfinite(mejor_tiempo):
            continue

        tiempo_oracle = float(mejor_tiempo)
        estrategia_oracle = estrategia_desde_accion_id(int(accion_id_oracle), mapa_acciones)

        # Métricas principales de policy, baseline y oracle
        row = {
            "season": fila.get("season"),
            "race_id": fila.get("race_id"),
            "circuit_key": fila.get("circuit_key"),
            "accion_real_id": int(fila["action_id"]) if pd.notna(fila.get("action_id")) else None,
            "estrategia_real": str(fila.get("strategy_compounds")),
            "finish_time_real": float(fila["finish_time_s"]) if pd.notna(fila.get("finish_time_s")) else None,

            "tiempo_baseline": float(tiempo_carrera_baseline),
            "tiempo_policy": float(tiempo_carrera_pi),
            "delta_policy_vs_baseline": float(tiempo_carrera_pi - tiempo_carrera_baseline),

            "baseline": str(baseline),

            "accion_policy_id": int(accion_id_pi),
            "accion_policy": str(estrategia_pi),
            "q_policy": float(np.max(q_values)),

            "oracle_id": int(accion_id_oracle),
            "oracle": str(estrategia_oracle),
            "tiempo_oracle": float(tiempo_oracle),

            "regret_policy": float(tiempo_carrera_pi - tiempo_oracle),
            "n_acciones_validas": int(len(ids_validas)),
        }

        if nombre_modelo is not None:
            row["modelo_q"] = str(nombre_modelo)

        # Evaluación top-k basada en el ranking Q
        for k in topk:
            top_ids = ranking_ids[:k]

            mejor_tiempo_topk = np.inf
            mejor_accion_id_topk: int | None = None

            for accion_id in top_ids:
                t_accion_id = tiempo_accion(accion_id)
                if np.isfinite(t_accion_id) and t_accion_id < mejor_tiempo_topk:
                    mejor_tiempo_topk = float(t_accion_id)
                    mejor_accion_id_topk = int(accion_id)

            if mejor_accion_id_topk is None or not np.isfinite(mejor_tiempo_topk):
                row[f"best_time@{k}"] = np.nan
                row[f"mejor_accion_id@{k}"] = -1
                row[f"regret@{k}"] = np.nan
                row[f"hit@{k}"] = 0
            else:
                row[f"best_time@{k}"] = float(mejor_tiempo_topk)
                row[f"mejor_accion_id@{k}"] = int(mejor_accion_id_topk)
                row[f"regret@{k}"] = float(mejor_tiempo_topk - tiempo_oracle)
                row[f"hit@{k}"] = int(int(accion_id_oracle) in top_ids)

        resultados.append(row)

    return pd.DataFrame(resultados)