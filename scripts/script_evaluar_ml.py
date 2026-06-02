"""
script_evaluar_ml.py

Entrenamiento y evaluación experimental de modelos supervisados.
"""

# IMPORTS
from __future__ import annotations
import pandas as pd
from estrategia_f1.acciones import imprimir_resumen_evaluacion
from estrategia_f1.config import (
    DATASET_EXPERIMENTAL_CSV,
    SEED,
    TEST_SIZE,
    ML_FILTRADO_RUNS_DIR,
    ML_RAW_RUNS_DIR,
    MODELOS_ML,
    TOPK,
    DATOS_MEMORIA,
    EVALUACION_ML_DETALLE_CSV,
    EVALUACION_ML_RESUMEN_CSV,
    EVALUACION_ML_MODELOS_CSV
)
from estrategia_f1.ml.entrenamiento_ml import (
    ConfiguracionEntrenamientoML,
    DireccionesML,
    entrenar_ml,
)
from estrategia_f1.ml.evaluacion_ml import (
    evaluar_politica_ml,
    evaluar_clasificacion_ml,
)

def main() -> None:
    """
    Ejecuta el flujo completo de entrenamiento y evaluación supervisada.

    El proceso incluye:
    1. Carga del dataset experimental.
    2. Entrenamiento de múltiples clasificadores supervisados.
    3. Evaluación de clasificación sobre el conjunto de test.
    4. Evaluación estratégica mediante simulación.
    5. Comparación entre variantes raw y filtrada.
    6. Generación de resúmenes comparativos finales.
    """

    df = pd.read_csv(DATASET_EXPERIMENTAL_CSV)

    resultados_todos: list[pd.DataFrame] = []
    metricas_clasificacion_todas: list[dict] = []

    # Variantes experimentales:
    # - raw: dataset sin filtrado específico
    # - filtrado: dataset con filtros de outliers aplicados únicamente en train
    variantes = [
        ("raw", False, ML_RAW_RUNS_DIR),
        ("filtrado", True, ML_FILTRADO_RUNS_DIR),
    ]

    # Se ejecutan de todos los modelos y variantes experimentales
    for nombre_modelo, params in MODELOS_ML.items():
        for nombre_variante, aplicar_filtros, base_runs_dir in variantes:

            print(f"================ ENTRENAMIENTO {nombre_modelo} ({nombre_variante}) ================")

            # Directorio específico de ejecución para almacenar modelo, metadatos y artefactos de caché
            run_dir = (base_runs_dir / nombre_modelo / f"seed={SEED}_ts={TEST_SIZE}")
            run_dir.mkdir(parents=True, exist_ok=True)

            ruta_modelo = run_dir / "modelo.joblib"
            ruta_meta = run_dir / "meta.joblib"

            configuracion_entrenamiento = ConfiguracionEntrenamientoML(
                seed=SEED,
                test_size=TEST_SIZE,
                modelo=nombre_modelo,
                modelo_params=params,
            )

            paths = DireccionesML(
                ruta_modelo=ruta_modelo,
                ruta_meta=ruta_meta,
            )

            # Entrenamiento supervisado con split temporal fijo y filtrado opcional aplicado
            entrenamiento = entrenar_ml(df, configuracionML=configuracion_entrenamiento, paths=paths,
                                        aplicar_filtros=aplicar_filtros)

            print("\n---------------------------------------------------------------\n")
            # Evaluación clásica de clasificación restringida a acciones válidas para cada observación
            metricas_clf = evaluar_clasificacion_ml(
                df=entrenamiento["df_test"],
                X=entrenamiento["X_test_estado"],
                y_true=entrenamiento["y_test"],
                modelo=entrenamiento["modelo"],
                mapa_acciones=entrenamiento["mapa_acciones"],
                topk=TOPK,
                nombre_modelo=nombre_modelo,
            )

            print("---------------- MÉTRICAS CLASIFICACIÓN ----------------")
            for k, v in metricas_clf.items():
                if isinstance(v, float):
                    print(f"{k:<35}: {v:.4f}")
                else:
                    print(f"{k:<35}: {v}")

            print("\n------------------- STATS FILTROS ----------------------")
            stats_filtros = entrenamiento.get("stats_filtros", {})

            for k, v in stats_filtros.items():
                print(f"{k:<35}: {v}")

            fila_metricas_clf = {
                **metricas_clf,
                **{f"filtros_{k}": v for k, v in stats_filtros.items()},
                "variante_dataset": nombre_variante,
            }

            metricas_clasificacion_todas.append(fila_metricas_clf)
            print("\n---------------------------------------------------------------\n")
            # Evaluación estratégica mediante simulación de carrera usando la política greedy del clasificador
            resultados_test = evaluar_politica_ml(
                df=entrenamiento["df_test"],
                X=entrenamiento["X_test_estado"],
                modelo=entrenamiento["modelo"],
                mapa_acciones=entrenamiento["mapa_acciones"],
                topk=TOPK,
                nombre_modelo=nombre_modelo,
            )

            resultados_test["variante_dataset"] = nombre_variante
            imprimir_resumen_evaluacion(resultados_test)

            resultados_todos.append(resultados_test)
            print("\n===============================================================\n")

    if len(resultados_todos) == 0:
        print("No se generaron resultados.")
        return

    df_all = pd.concat(resultados_todos, ignore_index=True)

    print(f"================ RESUMEN DE TODOS LOS MODELOS ML =================")

    grp = df_all.groupby(["modelo", "variante_dataset"], dropna=False)
    resumen = grp.agg(
        n=("delta_policy_vs_baseline", "size"),
        delta_mean=("delta_policy_vs_baseline", "mean"),
        delta_median=("delta_policy_vs_baseline", "median"),
        pct_policy_mejora=("delta_policy_vs_baseline", lambda s: (s < 0).mean() * 100.0),
        regret_mean=("regret_policy", "mean"),
        regret_median=("regret_policy", "median"),
    ).reset_index()

    DATOS_MEMORIA.mkdir(parents=True, exist_ok=True)

    df_all.to_csv(EVALUACION_ML_DETALLE_CSV, index=False)
    resumen.to_csv(EVALUACION_ML_RESUMEN_CSV, index=False)

    print("\nArchivos generados:")
    print(f"- {EVALUACION_ML_DETALLE_CSV}")
    print(f"- {EVALUACION_ML_RESUMEN_CSV}")

    df_metricas_clf = pd.DataFrame(metricas_clasificacion_todas)
    df_metricas_clf.to_csv(EVALUACION_ML_MODELOS_CSV, index=False)

    print(f"- {EVALUACION_ML_MODELOS_CSV}")

    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(resumen.sort_values(["modelo", "variante_dataset", "regret_mean", "delta_mean"]))

    print("\n===============================================================\n")

    # Comparativa directa entre dataset raw y filtrado para analizar el impacto del preprocesado
    print("================ COMPARACIÓN RAW vs FILTRADO POR MODELO =================")
    for modelo in df_all["modelo"].unique():
        df_modelo = df_all[df_all["modelo"] == modelo]
        
        if len(df_modelo["variante_dataset"].unique()) == 2:
            print(f"\n--- {modelo} ---")
            
            raw_data = df_modelo[df_modelo["variante_dataset"] == "raw"]
            filtrado_data = df_modelo[df_modelo["variante_dataset"] == "filtrado"]
            
            print(f"RAW      -> Regret medio: {raw_data['regret_policy'].mean():.3f}, Delta medio: {raw_data['delta_policy_vs_baseline'].mean():.3f}")
            print(f"FILTRADO -> Regret medio: {filtrado_data['regret_policy'].mean():.3f}, Delta medio: {filtrado_data['delta_policy_vs_baseline'].mean():.3f}")
            
            mejora_regret = raw_data['regret_policy'].mean() - filtrado_data['regret_policy'].mean()
            mejora_delta = raw_data['delta_policy_vs_baseline'].mean() - filtrado_data['delta_policy_vs_baseline'].mean()
            
            print(f"MEJORA   -> Regret: {mejora_regret:+.3f}, Delta: {mejora_delta:+.3f}")

    print("\n===============================================================\n")


if __name__ == "__main__":
    main()