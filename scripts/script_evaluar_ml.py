"""
script_evaluar_ml.py
TODO
"""

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
)

from estrategia_f1.ml.entrenamiento_ml import (
    ConfiguracionEntrenamientoML,
    DireccionesML,
    entrenar_ml_v1,
)

from estrategia_f1.ml.evaluacion_ml import (
    evaluar_politica_ml,
    evaluar_clasificacion_ml,
)

def main() -> None:
    df = pd.read_csv(DATASET_EXPERIMENTAL_CSV)

    resultados_todos: list[pd.DataFrame] = []

    # Configuraciones para ejecutar ambas variantes
    variantes = [
        ("raw", False, ML_RAW_RUNS_DIR),
        ("filtrado", True, ML_FILTRADO_RUNS_DIR),
    ]

    for nombre_modelo, params in MODELOS_ML.items():
        for nombre_variante, aplicar_filtros, base_runs_dir in variantes:

            print(f"================ ENTRENAMIENTO {nombre_modelo} ({nombre_variante}) ================")

            # Carpeta por modelo + seed + test_size
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

            # ← El filtrado específico ahora se maneja dentro de entrenar_ml_v1
            entrenamiento = entrenar_ml_v1(df, configuracionML=configuracion_entrenamiento, paths=paths,
                                         aplicar_filtros=aplicar_filtros)

            print("\n---------------------------------------------------------------\n")
            metricas_clf = evaluar_clasificacion_ml(
                df=entrenamiento["df_test"],
                X=entrenamiento["X_test_estado"],
                y_true=entrenamiento["y_test"],
                modelo=entrenamiento["modelo"],
                mapa_acciones=entrenamiento["mapa_acciones"],
                topk=TOPK,
                nombre_modelo=nombre_modelo,
            )
            print("Métricas clasificación:", metricas_clf)
            print("Stats filtros:", entrenamiento.get("stats_filtros"))
            print("\n---------------------------------------------------------------\n")

            resultados_test = evaluar_politica_ml(
                df=entrenamiento["df_test"],
                X=entrenamiento["X_test_estado"],
                modelo=entrenamiento["modelo"],
                mapa_acciones=entrenamiento["mapa_acciones"],
                topk=TOPK,
                nombre_modelo=nombre_modelo,
            )

            # Agregar información sobre la variante
            resultados_test["variante_dataset"] = nombre_variante
            imprimir_resumen_evaluacion(resultados_test)

            resultados_todos.append(resultados_test)
            print("\n===============================================================\n")

    # Comparativa final DE TODAS LAS COMBINACIONES
    if len(resultados_todos) == 0:
        print("No se generaron resultados.")
        return

    df_all = pd.concat(resultados_todos, ignore_index=True)

    print(f"================ RESUMEN DE TODOS LOS MODELOS Y VARIANTES =================")

    grp = df_all.groupby(["modelo", "variante_dataset"], dropna=False)
    resumen = grp.agg(
        n=("delta_policy_vs_baseline", "size"),
        delta_mean=("delta_policy_vs_baseline", "mean"),
        delta_median=("delta_policy_vs_baseline", "median"),
        pct_policy_mejora=("delta_policy_vs_baseline", lambda s: (s < 0).mean() * 100.0),
        regret_mean=("regret_policy", "mean"),
        regret_median=("regret_policy", "median"),
    ).reset_index()

    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(resumen.sort_values(["modelo", "variante_dataset", "regret_mean", "delta_mean"]))

    print("\n===============================================================\n")

    # ANÁLISIS COMPARATIVO POR MODELO
    print("================ COMPARACIÓN RAW vs FILTRADO POR MODELO =================")
    
    for modelo in df_all["modelo"].unique():
        df_modelo = df_all[df_all["modelo"] == modelo]
        
        if len(df_modelo["variante_dataset"].unique()) == 2:  # Tiene ambas variantes
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