"""
evaluacion_real.py

Evaluación complementaria del aproximador Q(s,a) frente a resultados reales observados.

Este módulo contiene la lógica necesaria para:
- Evaluar el valor Q asignado por el modelo a las estrategias reales observadas.
- Comparar las predicciones del aproximador con tiempos reales de carrera.
- Construir métricas relativas dentro de cada Gran Premio para evitar mezclar carreras distintas.
- Calcular proxies de recompensa real alineados con la formulación del modelo Q.
- Resumir correlaciones entre valor predicho, rendimiento real y ranking dentro de carrera.

Estas funciones no evalúan directamente la política recomendada en el mundo real,
sino la coherencia del aproximador Q sobre acciones históricamente observadas.
"""

# IMPORTS
from __future__ import annotations
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.base import RegressorMixin

# EVALUACIÓN -----------------------------------------------------------------------------------------------------------
def evaluar_q_en_escenario_real(df: pd.DataFrame, X: pd.DataFrame | np.ndarray, modelo: RegressorMixin, representacion_accion: dict[int, np.ndarray],
    *, nombre_modelo: str | None = None) -> pd.DataFrame:
    """
    Evalúa el aproximador Q sobre acciones reales observadas.

    Esta función no evalúa la política greedy recomendada por
    el modelo en escenario real, ya que normalmente no se dispone
    del tiempo real que habría obtenido una acción no ejecutada.
    En su lugar, calcula Q(s,a) únicamente para las estrategias
    históricamente observadas.

    Parámetros
    ----------
    df : pd.DataFrame
        Dataset con observaciones piloto-carrera.
    X : pd.DataFrame | np.ndarray
        Representación numérica de los estados.
    modelo : RegressorMixin
        Aproximador Q(s,a) previamente entrenado.
    representacion_accion : dict[int, np.ndarray]
        Representación numérica asociada a cada action_id.
    nombre_modelo : str | None, optional
        Nombre del modelo evaluado, usado para trazabilidad.

    Returns
    -------
    pd.DataFrame
        DataFrame con las acciones reales observadas, su tiempo
        real de carrera, el valor Q estimado y métricas relativas
        dentro de cada Gran Premio.
    """
    resultados: list[dict] = []

    for i in tqdm(range(len(df)), desc="Evaluando Q en escenario real"):
        fila = df.iloc[i]

        # Se recupera el estado correspondiente a la observación
        if isinstance(X, np.ndarray):
            x_estado = X[i].astype(np.float32, copy=False)
        else:
            x_estado = X.iloc[i].to_numpy(dtype=np.float32, copy=False)

        # Se utilizan únicamente acciones y tiempos realmente observados
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

        # La acción real debe pertenecer al espacio de acciones codificado
        if accion_real not in representacion_accion:
            continue

        x_accion = np.asarray(representacion_accion[accion_real], dtype=np.float32)
        X_input = np.concatenate([x_estado, x_accion], axis=0)[None, :]

        # Predicción del valor Q asignado a la acción histórica
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

    # Si no hay observaciones evaluables, se devuelve el DataFrame vacío
    if df_resultados.empty:
        return df_resultados

    # Se normaliza por carrera para evitar comparar tiempos absolutos entre Grandes Premios con distinta duración y características
    grp_race = df_resultados.groupby(["season", "race_id"])["finish_time_real"]

    df_resultados["finish_time_race_mean"] = grp_race.transform("mean")
    df_resultados["finish_time_race_median"] = grp_race.transform("median")
    df_resultados["finish_time_race_min"] = grp_race.transform("min")

    # Desviaciones respecto al contexto competitivo de cada carrera
    df_resultados["finish_time_vs_race_mean"] = (
        df_resultados["finish_time_real"] - df_resultados["finish_time_race_mean"]
    )
    df_resultados["finish_time_vs_race_median"] = (
        df_resultados["finish_time_real"] - df_resultados["finish_time_race_median"]
    )
    df_resultados["finish_time_vs_race_min"] = (
        df_resultados["finish_time_real"] - df_resultados["finish_time_race_min"]
    )

    # Proxies de recompensa real: mayor valor implica mejor rendimiento relativo
    df_resultados["reward_real_vs_race_mean"] = -df_resultados["finish_time_vs_race_mean"]
    df_resultados["reward_real_vs_race_median"] = -df_resultados["finish_time_vs_race_median"]
    df_resultados["reward_real_vs_race_min"] = -df_resultados["finish_time_vs_race_min"]

    # Ranking real: menor tiempo de carrera implica mejor posición relativa
    df_resultados["rank_real"] = (
        df_resultados
        .groupby(["season", "race_id"])["finish_time_real"]
        .rank(method="average", ascending=True)
    )

    # Ranking estimado: mayor Q implica mejor valoración del modelo
    df_resultados["rank_q"] = (
        df_resultados
        .groupby(["season", "race_id"])["q_real_observada"]
        .rank(method="average", ascending=False)
    )

    return df_resultados

# CÁLCULO -------------------------------------------------------------------------------------------------------------
def _corr_por_carrera(df: pd.DataFrame, col_x: str, col_y: str, *, metodo: str, min_muestras_por_carrera: int = 3) -> float:
    """
        Calcula correlaciones intra-carrera y devuelve su media.

        La correlación se calcula de forma independiente dentro
        de cada Gran Premio para evitar mezclar directamente
        observaciones pertenecientes a carreras distintas.

        Parámetros
        ----------
        df : pd.DataFrame
            Dataset con observaciones evaluadas.
        col_x : str
            Nombre de la primera variable.
        col_y : str
            Nombre de la segunda variable.
        metodo : str
            Método de correlación utilizado
            ("pearson" o "spearman").
        min_muestras_por_carrera : int, optional
            Número mínimo de observaciones requeridas para
            calcular correlación dentro de una carrera.

        Returns
        -------
        float
            Media de correlaciones válidas entre carreras.
            Devuelve np.nan si no puede calcularse ninguna.
        """
    valores: list[float] = []

    for _, grp in df.groupby(["season", "race_id"], sort=False):

        # Se ignoran carreras con pocas observaciones para evitar correlaciones poco representativas
        if len(grp) < min_muestras_por_carrera:
            continue

        corr = grp[col_x].corr(grp[col_y], method=metodo)
        if pd.notna(corr):
            valores.append(float(corr))

    if len(valores) == 0:
        return np.nan

    return float(np.mean(valores))

# RESUMEN POR CONSOLA --------------------------------------------------------------------------------------------------
def resumen_evaluacion_real(df_eval_real: pd.DataFrame) -> dict[str, float]:
    """
       Calcula un resumen de la evaluación real del aproximador Q(s,a).

       Las métricas principales comparan el valor Q asignado a las
       acciones reales observadas con proxies de recompensa real
       calculados dentro de cada carrera. Esto evita mezclar tiempos
       absolutos de Grandes Premios con duraciones y condiciones distintas.

       Parámetros
       ----------
       df_eval_real : pd.DataFrame
           Resultados obtenidos al evaluar Q(s,a) sobre acciones
           históricamente observadas.

       Returns
       -------
       dict[str, float]
           Diccionario con correlaciones entre Q, recompensas reales
           relativas, rankings de carrera y tiempos reales absolutos.
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

    # Series principales para las métricas globales exploratorias
    s_q = df_eval_real["q_real_observada"]
    s_t = df_eval_real["finish_time_real"]

    resumen = {
        "n_muestras": int(len(df_eval_real)),

        # Métricas principales: correlaciones calculadas dentro de cada carrera y promediadas después
        "mean_pearson_q_vs_reward_real_vs_race_median": _corr_por_carrera(
            df_eval_real,
            "q_real_observada",
            "reward_real_vs_race_median",
            metodo="pearson",
        ),
        "mean_spearman_q_vs_reward_real_vs_race_median": _corr_por_carrera(
            df_eval_real,
            "q_real_observada",
            "reward_real_vs_race_median",
            metodo="spearman",
        ),

        "mean_pearson_q_vs_reward_real_vs_race_min": _corr_por_carrera(
            df_eval_real,
            "q_real_observada",
            "reward_real_vs_race_min",
            metodo="pearson",
        ),
        "mean_spearman_q_vs_reward_real_vs_race_min": _corr_por_carrera(
            df_eval_real,
            "q_real_observada",
            "reward_real_vs_race_min",
            metodo="spearman",
        ),

        # Comparación entre ranking estimado por Q y ranking real dentro de cada Gran Premio
        "mean_spearman_rank_q_vs_real": _corr_por_carrera(
            df_eval_real,
            "rank_q",
            "rank_real",
            metodo="spearman",
        ),

        # Métricas globales secundarias: útiles como referencia, pero menos informativas por mezclar carreras distintas
        "pearson_global_q_vs_finish_time_real": float(s_q.corr(s_t, method="pearson")),
        "spearman_global_q_vs_finish_time_real": float(s_q.corr(s_t, method="spearman")),
    }

    return resumen