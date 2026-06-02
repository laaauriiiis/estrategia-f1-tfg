"""
script_evaluar_rl.py

Entrenamiento y evaluación experimental de modelos basados en valor Q(s,a).
"""

# IMPORTS
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
    DATOS_MEMORIA,
    EVALUACION_RL_DETALLE_CSV,
    EVALUACION_RL_RESUMEN_CSV,
    EVALUACION_RL_REAL_DETALLE_CSV,
    EVALUACION_RL_REAL_RESUMEN_CSV,
    EVALUACION_RL_MODELOS_CSV
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

def main() -> None:
    """
    Ejecuta el flujo completo de entrenamiento y evaluación RL.

    El proceso incluye:
    1. Carga del dataset experimental.
    2. Entrenamiento de aproximadores Q(s,a).
    3. Evaluación de políticas greedy mediante simulación.
    4. Evaluación del aproximador sobre acciones reales observadas.
    5. Comparación entre variantes raw y filtrada.
    6. Generación de resúmenes agregados de resultados.
    """

    df = pd.read_csv(DATASET_EXPERIMENTAL_CSV)

    resultados_todos: list[pd.DataFrame] = []
    resultados_reales_todos: list[pd.DataFrame] = []
    metricas_regresor_todas: list[dict] = []

    # Variantes experimentales:
    # - raw: dataset original
    # - filtrado: filtros aplicados únicamente sobre train
    variantes = [
        ("raw", False, RL_RAW_RUNS_DIR),
        ("filtrado", True, RL_FILTRADO_RUNS_DIR),
    ]

    # Ejecución de todos los modelos y variantes experimentales
    for nombre_modelo, params in MODELOS_RL.items():
        for nombre_variante, aplicar_filtros, base_runs_dir in variantes:
            print(f"================ ENTRENAMIENTO {nombre_modelo} | {nombre_variante} ================")

            # Directorio específico del experimento actual
            run_dir = (
                base_runs_dir
                / nombre_modelo
                / f"seed={SEED}_K={K_ACCIONES_MUESTREO}_ts={TEST_SIZE}"
            )
            run_dir.mkdir(parents=True, exist_ok=True)

            ruta_modelo = run_dir / "modelo_q.joblib"
            ruta_pares = run_dir / "pares.npz"
            ruta_meta = run_dir / "meta.joblib"

            # Configuración y rutas de persistencia del entrenamiento
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

            # Entrenamiento offline del aproximador Q(s,a)
            entrenamiento = entrenar_rl_offline(
                df,
                configuracionRL=configuracion_entrenamiento,
                paths=paths,
                aplicar_filtros=aplicar_filtros,
            )

            # Métricas básicas del aproximador entrenado
            print("\n---------------- MÉTRICAS REGRESOR Q ----------------")
            metricas_regresor = entrenamiento["metricas_regresor"]

            for k, v in metricas_regresor.items():
                if isinstance(v, float):
                    print(f"{k:<35}: {v:.4f}")
                else:
                    print(f"{k:<35}: {v}")

            print("\n------------------- STATS FILTROS ----------------------")
            stats_filtros = entrenamiento.get("stats_filtros", {})

            for k, v in stats_filtros.items():
                print(f"{k:<35}: {v}")

            fila_metricas_regresor = {
                **metricas_regresor,
                **{f"filtros_{k}": v for k, v in stats_filtros.items()},
                "modelo_q": nombre_modelo,
                "variante_dataset": nombre_variante,
            }

            metricas_regresor_todas.append(fila_metricas_regresor)

            print("\n---------------------------------------------------------------\n")
            # Evaluación simulada de la política greedy derivada de Q
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

            # Evaluación del aproximador Q sobre acciones reales observadas
            print(f"\n================ EVALUACIÓN REAL {nombre_modelo} | {nombre_variante} ================\n")
            df_eval_real = evaluar_q_en_escenario_real(
                df=entrenamiento["df_test"],
                X=entrenamiento["X_test_estado"],
                modelo=entrenamiento["modelo"],
                representacion_accion=entrenamiento["representacion_accion"],
                nombre_modelo=nombre_modelo,
            )

            df_eval_real["variante_dataset"] = nombre_variante

            resumen_real = resumen_evaluacion_real(df_eval_real)

            print("---------------- MÉTRICAS ESCENARIO REAL ----------------")
            for k, v in resumen_real.items():
                if isinstance(v, float):
                    print(f"{k:<55}: {v:.4f}")
                else:
                    print(f"{k:<55}: {v}")

            resultados_reales_todos.append(df_eval_real)

            print("\n===============================================================\n")

    if resultados_todos:
        df_all = pd.concat(resultados_todos, ignore_index=True)

        print("================ RESUMEN DE TODOS LOS MODELOS RL =================")
        grp = df_all.groupby(["modelo_q", "variante_dataset"], dropna=False)
        resumen = grp.agg(
            n=("delta_policy_vs_baseline", "size"),
            delta_mean=("delta_policy_vs_baseline", "mean"),
            delta_median=("delta_policy_vs_baseline", "median"),
            pct_policy_mejora=("delta_policy_vs_baseline", lambda s: (s < 0).mean() * 100.0),
            regret_mean=("regret_policy", "mean"),
            regret_median=("regret_policy", "median"),
        ).reset_index()

        DATOS_MEMORIA.mkdir(parents=True, exist_ok=True)

        df_all.to_csv(EVALUACION_RL_DETALLE_CSV, index=False)
        resumen.to_csv(EVALUACION_RL_RESUMEN_CSV, index=False)

        print("\nArchivos generados:")
        print(f"- {EVALUACION_RL_DETALLE_CSV}")
        print(f"- {EVALUACION_RL_RESUMEN_CSV}")

        df_metricas_regresor = pd.DataFrame(metricas_regresor_todas)
        df_metricas_regresor.to_csv(EVALUACION_RL_MODELOS_CSV, index=False)

        print(f"- {EVALUACION_RL_MODELOS_CSV}")

        with pd.option_context("display.max_columns", None, "display.width", 180):
            print(resumen.sort_values(["modelo_q", "regret_mean", "delta_mean"]))

        print("\n===============================================================\n")
    else:
        print("No se generaron resultados simulados.")

    if resultados_reales_todos:
        df_real_all = pd.concat(resultados_reales_todos, ignore_index=True)

        print("=============== RESUMEN EVALUACIÓN REAL =================")

        filas_resumen_real: list[dict] = []

        for (modelo_q, variante_dataset), df_grp in df_real_all.groupby(
                ["modelo_q", "variante_dataset"],
                dropna=False,
        ):
            # El resumen completo se delega al módulo evaluacion_real
            row = resumen_evaluacion_real(df_grp)

            # Información contextual del experimento
            row["modelo_q"] = modelo_q
            row["variante_dataset"] = variante_dataset

            filas_resumen_real.append(row)

        resumen_real_all = pd.DataFrame(filas_resumen_real)

        DATOS_MEMORIA.mkdir(parents=True, exist_ok=True)

        df_real_all.to_csv(EVALUACION_RL_REAL_DETALLE_CSV, index=False)
        resumen_real_all.to_csv(EVALUACION_RL_REAL_RESUMEN_CSV, index=False)

        print("\nArchivos generados:")
        print(f"- {EVALUACION_RL_REAL_DETALLE_CSV}")
        print(f"- {EVALUACION_RL_REAL_RESUMEN_CSV}")

        with pd.option_context(
                "display.max_columns", None,
                "display.width", 220,
        ):
            print(
                resumen_real_all.sort_values(
                    ["modelo_q", "variante_dataset"]
                )
            )

        print("\n=======================================================\n")

    else:
        print("No se generaron resultados reales.")

if __name__ == "__main__":
    main()