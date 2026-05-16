"""
evaluacion_ml.py

Evaluación experimental del enfoque supervisado.

Este módulo contiene la lógica necesaria para:
- Evaluar la política supervisada dentro del simulador frente a un baseline y frente al oracle.
- Calcular métricas de clasificación filtradas, incluyendo accuracy, macro-F1, balanced accuracy y top-k accuracy.
- Analizar la calidad del ranking estratégico generado por el modelo mediante regret y hit@k.

Estas funciones permiten comparar el comportamiento del enfoque
supervisado tanto desde el punto de vista de clasificación como
desde su rendimiento operativo dentro del entorno de simulación.
"""

# IMPORTS
from __future__ import annotations
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.base import ClassifierMixin
from estrategia_f1.sim.simulador import simular_tiempo_carrera
from estrategia_f1.acciones import (
    estrategia_desde_accion_id,
    acciones_validas_para_fila,
    elegir_estrategia_baseline,
)

# SELECCIÓN DE ACCIÓN --------------------------------------------------------------------------------------------------
def elegir_accion_modelo_ml(modelo: ClassifierMixin, x_estado: np.ndarray, ids_validas: np.ndarray) -> tuple[int, np.ndarray]:
    """
    Selecciona la acción recomendada por el modelo supervisado.

    Parámetros
    ----------
    modelo : ClassifierMixin
        Clasificador supervisado previamente entrenado.
    x_estado : np.ndarray
        Vector de estado de la carrera.
    ids_validas : np.ndarray
        Acciones compatibles con la carrera evaluada.

    Returns
    -------
    tuple[int, np.ndarray]
        Tupla formada por:
        - action_id seleccionado mediante política greedy.
        - Probabilidades asociadas a las acciones válidas,
          conservando el orden de ids_validas.
    """
    if len(ids_validas) == 0:
        raise ValueError("ids_validas está vacío")

    # Se normaliza el estado para garantizar compatibilidad con los clasificadores de scikit-learn durante inferencia
    x_estado = np.asarray(x_estado, dtype=np.float32)

    if not hasattr(modelo, "predict_proba"):
        raise TypeError("El modelo no tiene predict_proba().")

    # Se obtienen probabilidades únicamente para las clases realmente observadas durante entrenamiento
    proba = np.asarray(modelo.predict_proba(x_estado[None, :]))[0]

    # La posición de cada clase dentro de predict_proba() depende de modelo.classes_
    classes = np.asarray(modelo.classes_, dtype=int)
    class_to_col = {}

    for i, c in enumerate(classes):
        class_to_col[int(c)] = i

    # Se reconstruyen las probabilidades únicamente para las acciones válidas en la carrera actual
    scores = []

    for accion_id in ids_validas:
        accion_id = int(accion_id)

        # Si una acción válida nunca apareció en train, se le asigna probabilidad nula explícitamente
        if accion_id in class_to_col:
            idx = class_to_col[accion_id]
            prob = proba[idx]
        else:
            prob = 0.0

        scores.append(prob)
    scores = np.array(scores, dtype=float)

    # Política greedy sobre el subconjunto de acciones válidas
    idx_mejor = int(np.argmax(scores))
    return int(ids_validas[idx_mejor]), scores

# EVALUACIONES ---------------------------------------------------------------------------------------------------------
def evaluar_clasificacion_ml(df: pd.DataFrame, X, y_true, modelo: ClassifierMixin, mapa_acciones: dict[int, list[str]],
    *, topk: tuple[int, ...] = (3, 5), nombre_modelo: str | None = None) -> dict:
    """
    Evalúa el modelo supervisado como clasificador multiclase.

    La evaluación se realiza filtrando previamente las acciones
    compatibles con cada observación, de modo que las métricas
    reflejan la capacidad del modelo para recuperar la estrategia
    real dentro del espacio de acciones válido de cada carrera.

    Parámetros
    ----------
    df : pd.DataFrame
        Dataset de test con las observaciones piloto-carrera.
    X : np.ndarray | pd.DataFrame
        Matriz de estados utilizada como entrada del modelo.
    y_true : array-like
        Etiquetas reales correspondientes a action_id.
    modelo : ClassifierMixin
        Clasificador supervisado entrenado.
    mapa_acciones : dict[int, list[str]]
        Diccionario que relaciona cada action_id con su estrategia.
    topk : tuple[int, ...], optional
        Valores de k utilizados para calcular top-k accuracy.
    nombre_modelo : str | None, optional
        Nombre del modelo evaluado, utilizado para trazabilidad.

    Returns
    -------
    dict
        Diccionario con métricas de clasificación filtradas,
        incluyendo accuracy, macro-F1, balanced accuracy y top-k.
    """
    y_true = np.asarray(y_true, dtype=int)
    n = 0
    n_top1 = 0
    topk_hits = {k: 0 for k in topk}

    y_pred_filtrada = []
    y_true_filtrada = []

    for i in range(len(df)):
        fila = df.iloc[i]

        # Se obtiene el vector de estado de la observación, aceptando tanto matrices NumPy como DataFrames
        if isinstance(X, np.ndarray):
            x_estado = X[i].astype(np.float32, copy=False)
        else:
            x_estado = X.iloc[i].to_numpy(dtype=np.float32, copy=False)

        true_label = int(y_true[i])

        # La clasificación se restringe al subconjunto de acciones estratégicamente válidas para la carrera evaluada
        ids_validas = np.array(acciones_validas_para_fila(fila, mapa_acciones), dtype=int)
        if len(ids_validas) == 0:
            continue

        # Si la estrategia real no pertenece al espacio válido actual, la observación no es comparable y se descarta
        if true_label not in ids_validas:
            continue

        try:
            accion_pred, scores_validas = elegir_accion_modelo_ml(modelo, x_estado, ids_validas)
        except Exception:
            continue

        n += 1
        y_pred_filtrada.append(int(accion_pred))
        y_true_filtrada.append(true_label)

        if int(accion_pred) == true_label:
            n_top1 += 1

        # Ranking de acciones válidas por probabilidad descendente
        order_desc = np.argsort(-scores_validas)
        ranking_ids = ids_validas[order_desc]

        # Top-k mide si la estrategia real aparece entre las k acciones más probables propuestas por el modelo
        for k in topk:
            if true_label in ranking_ids[:k]:
                topk_hits[k] += 1

    resultados = {
        "n_muestras_validas_eval": int(n),
        "accuracy_filtrada": float(n_top1 / n) if n > 0 else np.nan,
    }

    if n > 0:
        # Macro-F1 pondera por igual todas las clases observadas
        resultados["macro_f1_filtrada"] = float(
            f1_score(y_true_filtrada, y_pred_filtrada, average="macro", zero_division=0)
        )
        # Balanced accuracy reduce el sesgo hacia estrategias mayoritarias
        resultados["balanced_accuracy_filtrada"] = float(
            balanced_accuracy_score(y_true_filtrada, y_pred_filtrada)
        )
    else:
        resultados["macro_f1_filtrada"] = np.nan
        resultados["balanced_accuracy_filtrada"] = np.nan

    for k in topk:
        resultados[f"top{k}_accuracy_filtrada"] = float(topk_hits[k] / n) if n > 0 else np.nan

    if nombre_modelo is not None:
        resultados["modelo"] = str(nombre_modelo)

    return resultados

def evaluar_politica_ml(df: pd.DataFrame, X: pd.DataFrame, modelo: ClassifierMixin, mapa_acciones: dict[int, list[str]],
        *, topk: tuple[int, ...] = (3, 5), nombre_modelo: str | None = None) -> pd.DataFrame:
    """
    Evalúa la política supervisada dentro del simulador.

    Para cada observación del conjunto de test, el modelo propone
    una estrategia entre las acciones válidas disponibles. Después,
    dicha estrategia se evalúa mediante el simulador y se compara
    frente a una estrategia baseline y frente al oracle simulado.

    Parámetros
    ----------
    df : pd.DataFrame
        Dataset de test con las observaciones piloto-carrera.
    X : pd.DataFrame
        Matriz de estados ya preprocesada.
    modelo : ClassifierMixin
        Clasificador supervisado entrenado.
    mapa_acciones : dict[int, list[str]]
        Diccionario que relaciona cada action_id con su estrategia
        de compuestos.
    topk : tuple[int, ...], optional
        Valores de k utilizados para evaluar ranking estratégico.
    nombre_modelo : str | None, optional
        Nombre del modelo evaluado, utilizado para trazabilidad.

    Returns
    -------
    pd.DataFrame
        DataFrame con los resultados de evaluación por observación,
        incluyendo tiempos simulados, delta frente al baseline,
        regret respecto al oracle y métricas top-k.
    """
    resultados: list[dict] = []

    for i in tqdm(range(len(df)), desc="Evaluando política"):
        fila = df.iloc[i]
        if isinstance(X, np.ndarray):
            x_estado = X[i].astype(np.float32, copy=False)
        else:
            x_estado = X.iloc[i].to_numpy(dtype=np.float32, copy=False)

        # Caché local para no simular varias veces la misma acción dentro de una misma observación piloto-carrera
        tiempos_cache: dict[int, float] = {}

        def tiempo_accion(accion_id: int) -> float:
            accion_id = int(accion_id)
            if accion_id not in tiempos_cache:
                estrategia = estrategia_desde_accion_id(accion_id, mapa_acciones)
                tiempo = simular_tiempo_carrera(fila, estrategia)
                tiempos_cache[accion_id] = float(tiempo) if np.isfinite(tiempo) else np.nan
            return tiempos_cache[accion_id]

        # La estrategia baseline actúa como referencia común para calcular la mejora relativa de la política supervisada
        baseline = elegir_estrategia_baseline(fila)
        if baseline is None:
            continue

        tiempo_carrera_baseline = simular_tiempo_carrera(fila, baseline)
        if not np.isfinite(tiempo_carrera_baseline):
            continue

        # Se restringe el espacio de decisión a estrategias compatibles con los compuestos y condiciones disponibles en la carrera
        ids_validas = np.array(acciones_validas_para_fila(fila, mapa_acciones), dtype=int)
        if len(ids_validas) == 0:
            continue

        # La política supervisada selecciona la acción válida con mayor probabilidad predicha por el clasificador
        try:
            accion_id_pi, p_values = elegir_accion_modelo_ml(modelo, x_estado, ids_validas)
        except Exception:
            continue

        estrategia_pi = estrategia_desde_accion_id(int(accion_id_pi), mapa_acciones)

        tiempo_carrera_pi = tiempo_accion(int(accion_id_pi))
        if not np.isfinite(tiempo_carrera_pi):
            continue

        # El ranking mantiene las acciones válidas ordenadas por probabilidad descendente para calcular métricas top-k
        order_desc = np.argsort(-p_values)
        ranking_ids = ids_validas[order_desc]

        # El oracle se obtiene simulando todas las acciones válidas y escogiendo la de menor tiempo estimado
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


        # Registro de métricas principales de la política para la observación evaluada
        row = {
            "tiempo_baseline": float(tiempo_carrera_baseline),
            "tiempo_policy": float(tiempo_carrera_pi),
            "delta_policy_vs_baseline": float(tiempo_carrera_pi - tiempo_carrera_baseline),

            "baseline": str(baseline),

            "accion_policy_id": int(accion_id_pi),
            "accion_policy": str(estrategia_pi),
            "p_policy": float(np.max(p_values)),

            "oracle_id": int(accion_id_oracle),
            "oracle": str(estrategia_oracle),
            "tiempo_oracle": float(tiempo_oracle),

            "regret_policy": float(tiempo_carrera_pi - tiempo_oracle),
            "n_acciones_validas": int(len(ids_validas)),
        }

        if nombre_modelo is not None:
            row["modelo"] = str(nombre_modelo)

        # Para cada top-k se evalúa la mejor estrategia simulada dentro de las k acciones más probables según el modelo
        for k in topk:
            top_ids = ranking_ids[:k]

            mejor_tiempo_topk = np.inf
            mejor_accion_id_topk: int | None = None

            for accion_id in top_ids:
                t_accion_id = tiempo_accion(int(accion_id))
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