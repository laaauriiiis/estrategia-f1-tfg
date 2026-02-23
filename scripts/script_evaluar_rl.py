"""
script_evaluar_rl.py
TODO
"""

from __future__ import annotations

import pandas as pd

from estrategia_f1.acciones import imprimir_resumen_evaluacion
from estrategia_f1.config import (
    DATASET_SIM_CSV,
    SEED,
    TEST_SIZE,
    K_ACCIONES_MUESTREO,
    RL_RUNS_DIR,
    MODELOS_RL,
    TOPK,
)

from estrategia_f1.rl.entrenamiento_rl import (
    ConfiguracionEntrenamientoRL,
    DireccionesRL,
    entrenar_rl_offline,
)

from estrategia_f1.rl.evaluacion_rl import (
    evaluar_politica_rl,
)

def main() -> None:

    df = pd.read_csv(DATASET_SIM_CSV)

    resultados_todos: list[pd.DataFrame] = []

    for nombre_modelo, params in MODELOS_RL.items():

        print(f"================ ENTRENAMIENTO {nombre_modelo} ================")

        # Carpeta por modelo + seed + K + test_size
        run_dir = (RL_RUNS_DIR / nombre_modelo / f"seed={SEED}_K={K_ACCIONES_MUESTREO}_ts={TEST_SIZE}")
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

        print("[RUN] carpeta:", paths.ruta_pares.parent, flush=True)
        print("[RUN] contenido antes:", [p.name for p in paths.ruta_pares.parent.glob("*")], flush=True)

        entrenamiento = entrenar_rl_offline(df, configuracionRL=configuracion_entrenamiento, paths=paths)

        print("[RUN] contenido después:", [p.name for p in paths.ruta_pares.parent.glob("*")], flush=True)

        print("\n---------------------------------------------------------------\n")
        print("Métricas del regresor Q:", entrenamiento["metricas_regresor"])
        print("\n---------------------------------------------------------------\n")

        resultados_test = evaluar_politica_rl(
            df=entrenamiento["df_test"],
            X=entrenamiento["X_test_estado"],
            modelo=entrenamiento["modelo"],
            mapa_acciones=entrenamiento["mapa_acciones"],
            representacion_accion=entrenamiento["representacion_accion"],
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

    grp = df_all.groupby("modelo_q", dropna=False)
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
