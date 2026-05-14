"""
script_evaluar_rl.py
Compara RL sin filtros vs con filtros.
"""

from __future__ import annotations

import pandas as pd

from estrategia_f1.acciones import imprimir_resumen_evaluacion
from estrategia_f1.config import (
    DATASET_EXPERIMENTAL_CSV,
    SEED,
    TEST_SIZE,
    K_ACCIONES_MUESTREO,
    MODELOS_RL,
    TOPK,
    RL_RAW_RUNS_DIR,
    RL_FILTRADO_RUNS_DIR,
)
from estrategia_f1.rl.entrenamiento_rl import (
    ConfiguracionEntrenamientoRL,
    DireccionesRL,
    entrenar_rl_offline,
)
from estrategia_f1.rl.evaluacion_rl import evaluar_politica_rl
from estrategia_f1.rl.evaluacion_rl_real import (
    evaluar_q_en_escenario_real,
    resumen_evaluacion_real,
)

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
    """
    valores: list[float] = []

    for _, grp in df.groupby(["season", "race_id"], sort=False):
        if len(grp) < min_muestras_por_carrera:
            continue

        corr = grp[col_x].corr(grp[col_y], method=method)
        if pd.notna(corr):
            valores.append(float(corr))

    if len(valores) == 0:
        return float("nan")

    return float(sum(valores) / len(valores))

def main() -> None:
    df = pd.read_csv(DATASET_EXPERIMENTAL_CSV)

    resultados_todos: list[pd.DataFrame] = []
    resultados_reales_todos: list[pd.DataFrame] = []

    variantes = [
        ("raw", False, RL_RAW_RUNS_DIR),
        ("filtrado", True, RL_FILTRADO_RUNS_DIR),
    ]

    for nombre_modelo, params in MODELOS_RL.items():
        for nombre_variante, aplicar_filtros, base_runs_dir in variantes:
            print(f"================ ENTRENAMIENTO {nombre_modelo} | {nombre_variante} ================")

            run_dir = (
                base_runs_dir
                / nombre_modelo
                / f"seed={SEED}_K={K_ACCIONES_MUESTREO}_ts={TEST_SIZE}"
            )
            run_dir.mkdir(parents=True, exist_ok=True)

            ruta_modelo = run_dir / "modelo_q.joblib"
            ruta_pares = run_dir / "pares.npz"
            ruta_meta = run_dir / "meta.joblib"

            configuracion_entrenamiento = ConfiguracionEntrenamientoRL(
                seed=SEED,
                test_size=TEST_SIZE,
                k_acciones_muestreo=K_ACCIONES_MUESTREO,
                modelo_q=nombre_modelo,
                modelo_q_params=params,
            )

            paths = DireccionesRL(
                ruta_pares=ruta_pares,
                ruta_modelo=ruta_modelo,
                ruta_meta=ruta_meta,
            )

            entrenamiento = entrenar_rl_offline(
                df,
                configuracionRL=configuracion_entrenamiento,
                paths=paths,
                aplicar_filtros=aplicar_filtros,
            )

            print("\n---------------------------------------------------------------\n")
            print("Métricas del regresor Q:", entrenamiento["metricas_regresor"])
            print("Stats filtros:", entrenamiento.get("stats_filtros"))
            print("\n---------------------------------------------------------------\n")

            # ---------------- EVALUACIÓN SIMULADA ----------------
            resultados_test = evaluar_politica_rl(
                df=entrenamiento["df_test"],
                X=entrenamiento["X_test_estado"],
                modelo=entrenamiento["modelo"],
                mapa_acciones=entrenamiento["mapa_acciones"],
                representacion_accion=entrenamiento["representacion_accion"],
                topk=TOPK,
                nombre_modelo=nombre_modelo,
            )

            resultados_test["variante_dataset"] = nombre_variante
            imprimir_resumen_evaluacion(resultados_test)
            resultados_todos.append(resultados_test)

            # ---------------- EVALUACIÓN REAL ----------------
            print(f"\n--- Evaluación en escenario real ({nombre_modelo} | {nombre_variante}) ---\n")

            df_eval_real = evaluar_q_en_escenario_real(
                df=entrenamiento["df_test"],
                X=entrenamiento["X_test_estado"],
                modelo=entrenamiento["modelo"],
                representacion_accion=entrenamiento["representacion_accion"],
                nombre_modelo=nombre_modelo,
            )

            df_eval_real["variante_dataset"] = nombre_variante

            resumen_real = resumen_evaluacion_real(df_eval_real)

            print("Resumen evaluación real:")
            for k, v in resumen_real.items():
                print(f"{k}: {v}")

            resultados_reales_todos.append(df_eval_real)

            print("\n===============================================================\n")

    # ---------------- RESUMEN SIMULADO ----------------
    if resultados_todos:
        df_all = pd.concat(resultados_todos, ignore_index=True)

        print("================ RESUMEN DE TODOS LOS MODELOS =================")

        grp = df_all.groupby(["modelo_q", "variante_dataset"], dropna=False)
        resumen = grp.agg(
            n=("delta_policy_vs_baseline", "size"),
            delta_mean=("delta_policy_vs_baseline", "mean"),
            delta_median=("delta_policy_vs_baseline", "median"),
            pct_policy_mejora=("delta_policy_vs_baseline", lambda s: (s < 0).mean() * 100.0),
            regret_mean=("regret_policy", "mean"),
            regret_median=("regret_policy", "median"),
        ).reset_index()

        with pd.option_context("display.max_columns", None, "display.width", 180):
            print(resumen.sort_values(["modelo_q", "regret_mean", "delta_mean"]))

        print("\n===============================================================\n")
    else:
        print("No se generaron resultados simulados.")

    # ---------------- RESUMEN REAL ----------------
    if resultados_reales_todos:
        df_real_all = pd.concat(resultados_reales_todos, ignore_index=True)

        print("=============== RESUMEN EVALUACIÓN REAL =================")

        filas_resumen_real: list[dict] = []

        for (modelo_q, variante_dataset), df_grp in df_real_all.groupby(
                ["modelo_q", "variante_dataset"], dropna=False
        ):
            row = {
                "modelo_q": modelo_q,
                "variante_dataset": variante_dataset,
                "n": int(len(df_grp)),

                # Métricas principales: por carrera, luego media
                "mean_pearson_q_vs_reward_real_vs_race_median": _corr_por_carrera(
                    df_grp,
                    "q_real_observada",
                    "reward_real_vs_race_median",
                    method="pearson",
                ),
                "mean_spearman_q_vs_reward_real_vs_race_median": _corr_por_carrera(
                    df_grp,
                    "q_real_observada",
                    "reward_real_vs_race_median",
                    method="spearman",
                ),
                "mean_pearson_q_vs_reward_real_vs_race_min": _corr_por_carrera(
                    df_grp,
                    "q_real_observada",
                    "reward_real_vs_race_min",
                    method="pearson",
                ),
                "mean_spearman_q_vs_reward_real_vs_race_min": _corr_por_carrera(
                    df_grp,
                    "q_real_observada",
                    "reward_real_vs_race_min",
                    method="spearman",
                ),
                "mean_spearman_rank_q_vs_real": _corr_por_carrera(
                    df_grp,
                    "rank_q",
                    "rank_real",
                    method="spearman",
                ),

                # Métricas secundarias / exploratorias globales
                "pearson_global_q_vs_finish_time_real": float(
                    df_grp["q_real_observada"].corr(df_grp["finish_time_real"], method="pearson")
                ),
                "spearman_global_q_vs_finish_time_real": float(
                    df_grp["q_real_observada"].corr(df_grp["finish_time_real"], method="spearman")
                ),
            }

            filas_resumen_real.append(row)

        resumen_real_all = pd.DataFrame(filas_resumen_real)

        with pd.option_context("display.max_columns", None, "display.width", 220):
            print(resumen_real_all.sort_values(["modelo_q", "variante_dataset"]))

        print("\n=======================================================\n")
    else:
        print("No se generaron resultados reales.")


if __name__ == "__main__":
    main()