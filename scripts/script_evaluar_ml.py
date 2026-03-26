"""
script_evaluar_ml.py
TODO
"""

from __future__ import annotations

import pandas as pd

from estrategia_f1.acciones import imprimir_resumen_evaluacion
from estrategia_f1.config import (
    DATASET_SIM_CSV,
    SEED,
    TEST_SIZE,
    ML_RUNS_DIR,
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
    df = pd.read_csv(DATASET_SIM_CSV)

    resultados_todos: list[pd.DataFrame] = []

    for nombre_modelo, params in MODELOS_ML.items():

        print(f"================ ENTRENAMIENTO {nombre_modelo} ================")

        # Carpeta por modelo + seed + K + test_size
        run_dir = (ML_RUNS_DIR / nombre_modelo/ f"seed={SEED}_ts={TEST_SIZE}")
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

        entrenamiento = entrenar_ml_v1(df, configuracionML=configuracion_entrenamiento, paths=paths)

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
        print("\n---------------------------------------------------------------\n")

        resultados_test = evaluar_politica_ml(
            df=entrenamiento["df_test"],
            X=entrenamiento["X_test_estado"],
            modelo=entrenamiento["modelo"],
            mapa_acciones=entrenamiento["mapa_acciones"],
            topk=TOPK,
            nombre_modelo=nombre_modelo,
        )

        imprimir_resumen_evaluacion(resultados_test)

        resultados_todos.append(resultados_test)
        print("\n===============================================================\n")

    # Comparativa final
    if len(resultados_todos) == 0:
        print("No se generaron resultados.")
        return

    df_all = pd.concat(resultados_todos, ignore_index=True)

    print(f"================ RESUMEN DE TODOS LOS MODELOS =================")

    grp = df_all.groupby("modelo", dropna=False)
    resumen = grp.agg(
        n=("delta_policy_vs_baseline", "size"),
        delta_mean=("delta_policy_vs_baseline", "mean"),
        delta_median=("delta_policy_vs_baseline", "median"),
        pct_policy_mejora=("delta_policy_vs_baseline", lambda s: (s < 0).mean() * 100.0),
        regret_mean=("regret_policy", "mean"),
        regret_median=("regret_policy", "median"),
    ).reset_index()

    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(resumen.sort_values(["regret_mean", "delta_mean"]))

    print("\n===============================================================\n")


if __name__ == "__main__":
    main()
