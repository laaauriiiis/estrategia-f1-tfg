"""
evaluacion_comun.py - Métricas unificadas para ML y RL
"""

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class ResultadosEvaluacion:
    """Resultados unificados para ML y RL"""
    tipo_modelo: Literal["ML", "RL"]
    nombre_modelo: str
    variante_dataset: str

    # Métricas de política
    regret_mean: float
    regret_median: float
    delta_vs_baseline_mean: float
    delta_vs_baseline_median: float
    pct_mejora_vs_baseline: float

    # Métricas de ranking/clasificación
    top1_hit_rate: float
    top3_hit_rate: float
    top5_hit_rate: float

    # Estadísticas adicionales
    n_muestras_evaluadas: int
    n_acciones_validas_promedio: float

    # Métricas específicas por tipo
    metricas_especificas: dict[str, Any]


def evaluar_modelo_unificado(
        modelo: Any,
        df_test: pd.DataFrame,
        X_test: np.ndarray,
        mapa_acciones: dict[int, list[str]],
        tipo_modelo: Literal["ML", "RL"],
        nombre_modelo: str,
        variante_dataset: str,
        topk: tuple[int, ...] = (3, 5)
) -> ResultadosEvaluacion:
    """Evaluación unificada que funciona tanto para ML como RL"""

    resultados_raw = []

    for i in tqdm(range(len(df_test)), desc=f"Evaluando {tipo_modelo}"):
        # ... lógica común de evaluación ...

        # Elegir acción según el tipo de modelo
        if tipo_modelo == "ML":
            accion_id, scores = elegir_accion_modelo_ml(modelo, x_estado, ids_validas)
        elif tipo_modelo == "RL":
            accion_id, scores = elegir_accion_modelo_rl(modelo, x_estado, ids_validas)

        # ... resto de la evaluación común ...

    return ResultadosEvaluacion(...)