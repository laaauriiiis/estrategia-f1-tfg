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

    # Proxies de recompensa real alineados con la idea de "mejor = mayor reward"
    df_resultados["reward_real_vs_race_mean"] = -df_resultados["finish_time_vs_race_mean"]
    df_resultados["reward_real_vs_race_median"] = -df_resultados["finish_time_vs_race_median"]
    df_resultados["reward_real_vs_race_min"] = -df_resultados["finish_time_vs_race_min"]

    # Rankings dentro de cada carrera
    # Menor tiempo real = mejor => rank 1 mejor
    df_resultados["rank_real"] = (
        df_resultados
        .groupby(["season", "race_id"])["finish_time_real"]
        .rank(method="average", ascending=True)
    )

    # Mayor Q = mejor => rank 1 mejor
    df_resultados["rank_q"] = (
        df_resultados
        .groupby(["season", "race_id"])["q_real_observada"]
        .rank(method="average", ascending=False)
    )

    return df_resultados


def _corr_por_carrera(
    df: pd.DataFrame,
    col_x: str,
    col_y: str,
    *,
    method: str,
    min_muestras_por_carrera: int = 3,
) -> float:
    """
    Calcula correlación dentro de cada carrera y devuelve la media.

    Esto evita mezclar directamente observaciones de carreras distintas
    en una sola correlación global.
    """
    valores: list[float] = []

    for _, grp in df.groupby(["season", "race_id"], sort=False):
        if len(grp) < min_muestras_por_carrera:
            continue

        corr = grp[col_x].corr(grp[col_y], method=method)
        if pd.notna(corr):
            valores.append(float(corr))

    if len(valores) == 0:
        return np.nan

    return float(np.mean(valores))


def resumen_evaluacion_real(df_eval_real: pd.DataFrame) -> dict[str, float]:
    """
    Resumen de evaluación real usando proxies de recompensa alineados con Q.

    IMPORTANTE:
    - Q(s,a) se entrenó contra una recompensa simulada relativa.
    - Por tanto, aquí se priorizan targets reales relativos dentro de carrera,
      no finish_time_real absoluto.
    - Las correlaciones principales se calculan por carrera y luego se promedian.
    """
    if df_eval_real.empty:
        return {
            "n_muestras": 0,
            "mean_pearson_q_vs_reward_real_vs_race_median": np.nan,
            "mean_spearman_q_vs_reward_real_vs_race_median": np.nan,
            "mean_pearson_q_vs_reward_real_vs_race_min": np.nan,
            "mean_spearman_q_vs_reward_real_vs_race_min": np.nan,
            "mean_spearman_rank_q_vs_real": np.nan,
            "pearson_global_q_vs_finish_time_real": np.nan,
            "spearman_global_q_vs_finish_time_real": np.nan,
        }

    s_q = df_eval_real["q_real_observada"]
    s_t = df_eval_real["finish_time_real"]

    resumen = {
        "n_muestras": int(len(df_eval_real)),

        # Métricas principales: por carrera, luego media
        "mean_pearson_q_vs_reward_real_vs_race_median": _corr_por_carrera(
            df_eval_real,
            "q_real_observada",
            "reward_real_vs_race_median",
            method="pearson",
        ),
        "mean_spearman_q_vs_reward_real_vs_race_median": _corr_por_carrera(
            df_eval_real,
            "q_real_observada",
            "reward_real_vs_race_median",
            method="spearman",
        ),

        "mean_pearson_q_vs_reward_real_vs_race_min": _corr_por_carrera(
            df_eval_real,
            "q_real_observada",
            "reward_real_vs_race_min",
            method="pearson",
        ),
        "mean_spearman_q_vs_reward_real_vs_race_min": _corr_por_carrera(
            df_eval_real,
            "q_real_observada",
            "reward_real_vs_race_min",
            method="spearman",
        ),

        # Ranking dentro de carrera
        "mean_spearman_rank_q_vs_real": _corr_por_carrera(
            df_eval_real,
            "rank_q",
            "rank_real",
            method="spearman",
        ),

        # Métricas secundarias / exploratorias globales
        "pearson_global_q_vs_finish_time_real": float(s_q.corr(s_t, method="pearson")),
        "spearman_global_q_vs_finish_time_real": float(s_q.corr(s_t, method="spearman")),
    }

    return resumen