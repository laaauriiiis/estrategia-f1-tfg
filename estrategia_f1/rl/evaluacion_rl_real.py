"""
evaluacion_real.py
Evaluación complementaria del modelo Q usando resultados reales observados.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.base import RegressorMixin


def evaluar_q_en_escenario_real(
    df: pd.DataFrame,
    X: pd.DataFrame | np.ndarray,
    modelo: RegressorMixin,
    representacion_accion: dict[int, np.ndarray],
    *,
    nombre_modelo: str | None = None,
) -> pd.DataFrame:
    """
    Evalúa el modelo Q sobre pares (estado, acción_real) observados en los datos.

    IMPORTANTE:
    - Esta función NO evalúa la acción greedy recomendada por la policy en escenario real,
      porque normalmente no disponemos del finish_time_s real de esa acción recomendada.
    - Solo evalúa acciones realmente observadas en el dataset:
        (s, a_real, finish_time_s)

    Para cada fila:
      1) toma el estado s
      2) toma la acción real action_id
      3) calcula Q(s, a_real)
      4) lo guarda junto al finish_time_s real

    Devuelve un DataFrame con una fila por observación válida.
    """
    resultados: list[dict] = []

    for i in tqdm(range(len(df)), desc="Evaluando Q en escenario real"):
        fila = df.iloc[i]

        # Estado
        if isinstance(X, np.ndarray):
            x_estado = X[i].astype(np.float32, copy=False)
        else:
            x_estado = X.iloc[i].to_numpy(dtype=np.float32, copy=False)

        # Campos mínimos necesarios
        accion_real = fila.get("action_id", np.nan)
        finish_time_real = fila.get("finish_time_s", np.nan)

        if pd.isna(accion_real) or pd.isna(finish_time_real):
            continue

        try:
            accion_real = int(accion_real)
            finish_time_real = float(finish_time_real)
        except (TypeError, ValueError):
            continue

        if not np.isfinite(finish_time_real):
            continue

        # La acción debe tener representación
        if accion_real not in representacion_accion:
            continue

        x_accion = np.asarray(representacion_accion[accion_real], dtype=np.float32)
        X_input = np.concatenate([x_estado, x_accion], axis=0)[None, :]

        # Predicción Q para la acción real observada
        try:
            q_real_observada = float(np.asarray(modelo.predict(X_input)).reshape(-1)[0])
        except Exception:
            continue

        row = {
            "season": fila.get("season"),
            "race_id": fila.get("race_id"),
            "circuit_key": fila.get("circuit_key"),

            "accion_real_id": accion_real,
            "strategy_compounds_real": fila.get("strategy_compounds"),
            "n_stints_real": fila.get("n_stints"),

            "finish_time_real": finish_time_real,
            "q_real_observada": q_real_observada,

            "dnf": fila.get("dnf"),
            "dns": fila.get("dns"),
            "dsq": fila.get("dsq"),
        }

        if nombre_modelo is not None:
            row["modelo_q"] = str(nombre_modelo)

        resultados.append(row)

    df_resultados = pd.DataFrame(resultados)

    if df_resultados.empty:
        return df_resultados

    # Normalizaciones por carrera para facilitar análisis real
    grp_race = df_resultados.groupby(["season", "race_id"])["finish_time_real"]

    df_resultados["finish_time_race_mean"] = grp_race.transform("mean")
    df_resultados["finish_time_race_median"] = grp_race.transform("median")
    df_resultados["finish_time_race_min"] = grp_race.transform("min")

    df_resultados["finish_time_vs_race_mean"] = (
        df_resultados["finish_time_real"] - df_resultados["finish_time_race_mean"]
    )
    df_resultados["finish_time_vs_race_median"] = (
        df_resultados["finish_time_real"] - df_resultados["finish_time_race_median"]
    )
    df_resultados["finish_time_vs_race_min"] = (
        df_resultados["finish_time_real"] - df_resultados["finish_time_race_min"]
    )

    return df_resultados


def resumen_evaluacion_real(df_eval_real: pd.DataFrame) -> dict[str, float]:
    """
    Calcula un pequeño resumen de la evaluación real.

    Métricas incluidas:
    - correlación Pearson y Spearman entre Q observado y tiempo real
    - idem con tiempos normalizados por carrera
    - nº de muestras evaluadas

    Interpretación esperada:
    - Si Q alto significa mejor estrategia, entonces debería haber
      correlación NEGATIVA entre q_real_observada y finish_time_real
      (o sus versiones normalizadas).
    """
    if df_eval_real.empty:
        return {
            "n_muestras": 0,
            "pearson_q_vs_finish_time_real": np.nan,
            "spearman_q_vs_finish_time_real": np.nan,
            "pearson_q_vs_finish_time_vs_race_median": np.nan,
            "spearman_q_vs_finish_time_vs_race_median": np.nan,
            "pearson_q_vs_finish_time_vs_race_min": np.nan,
            "spearman_q_vs_finish_time_vs_race_min": np.nan,
        }

    s_q = df_eval_real["q_real_observada"]
    s_t = df_eval_real["finish_time_real"]
    s_t_med = df_eval_real["finish_time_vs_race_median"]
    s_t_min = df_eval_real["finish_time_vs_race_min"]

    resumen = {
        "n_muestras": int(len(df_eval_real)),

        "pearson_q_vs_finish_time_real": float(s_q.corr(s_t, method="pearson")),
        "spearman_q_vs_finish_time_real": float(s_q.corr(s_t, method="spearman")),

        "pearson_q_vs_finish_time_vs_race_median": float(s_q.corr(s_t_med, method="pearson")),
        "spearman_q_vs_finish_time_vs_race_median": float(s_q.corr(s_t_med, method="spearman")),

        "pearson_q_vs_finish_time_vs_race_min": float(s_q.corr(s_t_min, method="pearson")),
        "spearman_q_vs_finish_time_vs_race_min": float(s_q.corr(s_t_min, method="spearman")),
    }

    return resumen