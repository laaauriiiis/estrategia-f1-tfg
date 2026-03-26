"""
evaluacion_ml.py
TODO
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

from estrategia_f1.sim.simulador import simular_tiempo_carrera
from estrategia_f1.acciones import (
    estrategia_desde_accion_id,
    acciones_validas_para_fila,
    elegir_estrategia_baseline,
)


# Escogemos acción por probabilidad predecida---------------------------------------------------------------------------
def elegir_accion_modelo_ml(modelo, x_estado, ids_validas):
    probs = modelo.predict_proba(x_estado[None, :])[0]

    # Filtramos solo acciones válidas
    probs_validas = probs[ids_validas]

    idx_mejor_local = np.argmax(probs_validas)
    accion_id_mejor = ids_validas[idx_mejor_local]

    return accion_id_mejor, probs

from sklearn.base import ClassifierMixin

def elegir_accion_modelo_ml(modelo: ClassifierMixin, x_estado: np.ndarray, ids_validas: np.ndarray) -> tuple[int, np.ndarray]:
    """
    Devuelve:
      - accion_id_mejor (greedy)
      - scores_validas: probas en el mismo orden que ids_validas
    """
    if len(ids_validas) == 0:
        raise ValueError("ids_validas está vacío")

    x_estado = np.asarray(x_estado, dtype=np.float32)

    if not hasattr(modelo, "predict_proba"):
        raise TypeError("El modelo no tiene predict_proba().")

    proba = np.asarray(modelo.predict_proba(x_estado[None, :]))[0]

    # Score por acción válida (si la clase no existe en train -> prob=0)
    # Diccionario clase -> índice en proba
    classes = np.asarray(modelo.classes_, dtype=int)
    class_to_col = {}

    for i, c in enumerate(classes):
        class_to_col[int(c)] = i

    # Construimos scores para las acciones válidas
    scores = []

    for accion_id in ids_validas:
        accion_id = int(accion_id)

        if accion_id in class_to_col:
            idx = class_to_col[accion_id]
            prob = proba[idx]
        else:
            prob = 0.0

        scores.append(prob)

    scores = np.array(scores, dtype=float)
    idx_mejor = int(np.argmax(scores))
    return int(ids_validas[idx_mejor]), scores

def evaluar_politica_ml(df: pd.DataFrame, X: pd.DataFrame, modelo: ClassifierMixin, mapa_acciones: dict[int, list[str]],
        *, topk: tuple[int, ...] = (3, 5), nombre_modelo: str | None = None) -> pd.DataFrame:
    """
    Evaluación en test con:
      1) Baseline vs policy
      2) Oracle (regret)
      3) Top-k (best@k, regret@k, hit@k)

    Nota: filtra acciones por fila según compuestos disponibles.
    """
    resultados: list[dict] = []

    for i in tqdm(range(len(df)), desc="Evaluando política"):
        fila = df.iloc[i]
        if isinstance(X, np.ndarray):
            x_estado = X[i].astype(np.float32, copy=False)
        else:
            x_estado = X.iloc[i].to_numpy(dtype=np.float32, copy=False)

        tiempos_cache: dict[int, float] = {}

        def tiempo_accion(accion_id: int) -> float:
            accion_id = int(accion_id)
            if accion_id not in tiempos_cache:
                estrategia = estrategia_desde_accion_id(accion_id, mapa_acciones)
                tiempo = simular_tiempo_carrera(fila, estrategia)
                tiempos_cache[accion_id] = float(tiempo) if np.isfinite(tiempo) else np.nan
            return tiempos_cache[accion_id]

        # Baseline
        baseline = elegir_estrategia_baseline(fila)
        if baseline is None:
            continue

        tiempo_carrera_baseline = simular_tiempo_carrera(fila, baseline)
        if not np.isfinite(tiempo_carrera_baseline):
            continue

        # Acciones válidas
        ids_validas = np.array(acciones_validas_para_fila(fila, mapa_acciones), dtype=int)
        if len(ids_validas) == 0:
            continue

        # Elegir acción según ML (greedy por prob)
        try:
            accion_id_pi, p_values = elegir_accion_modelo_ml(modelo, x_estado, ids_validas)
        except Exception:
            continue

        estrategia_pi = estrategia_desde_accion_id(int(accion_id_pi), mapa_acciones)

        tiempo_carrera_pi = tiempo_accion(int(accion_id_pi))
        if not np.isfinite(tiempo_carrera_pi):
            continue

        # Ranking descendente por probabilidad (p_values está en el orden de ids_validas)
        order_desc = np.argsort(-p_values)
        ranking_ids = ids_validas[order_desc]

        # Oracle
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

        # Top-k
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

def evaluar_clasificacion_ml(df: pd.DataFrame, X, y_true, modelo: ClassifierMixin, mapa_acciones: dict[int, list[str]],
    *, topk: tuple[int, ...] = (3, 5), nombre_modelo: str | None = None) -> dict:
    """
    Evalúa el modelo como clasificador, pero filtrando por acciones válidas
    en cada fila antes de decidir top-1 y top-k.
    """

    y_true = np.asarray(y_true, dtype=int)

    n = 0
    n_top1 = 0
    topk_hits = {k: 0 for k in topk}

    y_pred_filtrada = []
    y_true_filtrada = []

    for i in range(len(df)):
        fila = df.iloc[i]

        if isinstance(X, np.ndarray):
            x_estado = X[i].astype(np.float32, copy=False)
        else:
            x_estado = X.iloc[i].to_numpy(dtype=np.float32, copy=False)

        true_label = int(y_true[i])

        ids_validas = np.array(acciones_validas_para_fila(fila, mapa_acciones), dtype=int)
        if len(ids_validas) == 0:
            continue

        # Si la acción real no es válida según tus reglas actuales, la saltamos
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

        order_desc = np.argsort(-scores_validas)
        ranking_ids = ids_validas[order_desc]

        for k in topk:
            if true_label in ranking_ids[:k]:
                topk_hits[k] += 1

    resultados = {
        "n_muestras_validas_eval": int(n),
        "accuracy_filtrada": float(n_top1 / n) if n > 0 else np.nan,
    }

    if n > 0:
        resultados["macro_f1_filtrada"] = float(
            f1_score(y_true_filtrada, y_pred_filtrada, average="macro", zero_division=0)
        )
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