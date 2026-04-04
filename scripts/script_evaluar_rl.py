"""
script_evaluar_rl.py
TODO
"""

from __future__ import annotations

import numpy as np
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

from estrategia_f1.rl.evaluacion_rl_real import (
    evaluar_q_en_escenario_real,
    resumen_evaluacion_real,
)


def imprimir_checks_evaluacion_real(df_eval_real: pd.DataFrame, *, nombre_modelo: str) -> None:
    """
    Checks de sanidad para entender si la evaluación real tiene sentido
    o si podría haber algún problema en datos/código.
    """
    print(f"\n################ CHECKS EVALUACIÓN REAL: {nombre_modelo} ################")

    if df_eval_real.empty:
        print("df_eval_real está vacío.")
        print("####################################################################\n")
        return

    # 1) Describe de Q
    print("\n[CHECK 1] describe() de q_real_observada")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df_eval_real["q_real_observada"].describe())

    # 2) Ejemplos crudos
    print("\n[CHECK 2] primeras 10 filas: Q vs tiempo real")
    cols_preview = [
        c for c in [
            "season",
            "race_id",
            "circuit_key",
            "accion_real_id",
            "strategy_compounds_real",
            "n_stints_real",
            "q_real_observada",
            "finish_time_real",
            "finish_time_vs_race_median",
            "dnf",
            "dns",
            "dsq",
        ] if c in df_eval_real.columns
    ]
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df_eval_real[cols_preview].head(10))

    # 3) Top Q altos: ¿parecen buenos en real?
    print("\n[CHECK 3] top 15 por Q más alto")
    cols_topq = [
        c for c in [
            "season",
            "race_id",
            "circuit_key",
            "strategy_compounds_real",
            "q_real_observada",
            "finish_time_real",
            "finish_time_vs_race_median",
            "finish_time_vs_race_min",
            "dnf",
            "dns",
            "dsq",
        ] if c in df_eval_real.columns
    ]
    top_q = df_eval_real.sort_values("q_real_observada", ascending=False).head(15)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(top_q[cols_topq])

    # 4) Distribución del target real normalizado
    if "finish_time_vs_race_median" in df_eval_real.columns:
        print("\n[CHECK 4] describe() de finish_time_vs_race_median")
        with pd.option_context("display.max_columns", None, "display.width", 160):
            print(df_eval_real["finish_time_vs_race_median"].describe())

    # 4b) Outliers del target real normalizado
    if "finish_time_vs_race_median" in df_eval_real.columns:
        print("\n[CHECK 4b] filas con finish_time_vs_race_median más extremo (más bajo)")
        cols_outliers = [
            c for c in [
                "season",
                "race_id",
                "circuit_key",
                "accion_real_id",
                "strategy_compounds_real",
                "q_real_observada",
                "finish_time_real",
                "finish_time_vs_race_median",
                "finish_time_vs_race_min",
                "dnf",
                "dns",
                "dsq",
            ] if c in df_eval_real.columns
        ]
        with pd.option_context("display.max_columns", None, "display.width", 220):
            print(
                df_eval_real
                .sort_values("finish_time_vs_race_median", ascending=True)[cols_outliers]
                .head(15)
            )

        print("\n[CHECK 4c] filas con finish_time_vs_race_median más extremo (más alto)")
        with pd.option_context("display.max_columns", None, "display.width", 220):
            print(
                df_eval_real
                .sort_values("finish_time_vs_race_median", ascending=False)[cols_outliers]
                .head(15)
            )

        # Trim 1%-99% para ver sensibilidad a extremos
        q01 = df_eval_real["finish_time_vs_race_median"].quantile(0.01)
        q99 = df_eval_real["finish_time_vs_race_median"].quantile(0.99)

        df_trim = df_eval_real[
            df_eval_real["finish_time_vs_race_median"].between(q01, q99)
        ].copy()

        print(f"\n[CHECK 4d] trim 1%-99% en finish_time_vs_race_median")
        print(f"q01 = {q01}")
        print(f"q99 = {q99}")
        print(f"filas originales = {len(df_eval_real)} | filas tras trim = {len(df_trim)}")

        if len(df_trim) >= 5:
            corr_pearson_trim_real = df_trim["q_real_observada"].corr(df_trim["finish_time_real"], method="pearson")
            corr_spearman_trim_real = df_trim["q_real_observada"].corr(df_trim["finish_time_real"], method="spearman")
            corr_pearson_trim_med = df_trim["q_real_observada"].corr(df_trim["finish_time_vs_race_median"], method="pearson")
            corr_spearman_trim_med = df_trim["q_real_observada"].corr(df_trim["finish_time_vs_race_median"], method="spearman")

            print("Correlaciones tras trim:")
            print(f"Pearson trim   Q vs finish_time_real           : {corr_pearson_trim_real}")
            print(f"Spearman trim  Q vs finish_time_real           : {corr_spearman_trim_real}")
            print(f"Pearson trim   Q vs finish_time_vs_race_median: {corr_pearson_trim_med}")
            print(f"Spearman trim  Q vs finish_time_vs_race_median: {corr_spearman_trim_med}")

    # 5) Filtro limpio: sin DNF/DNS/DSQ
    df_limpio = df_eval_real.copy()

    for col in ["dnf", "dns", "dsq"]:
        if col in df_limpio.columns:
            df_limpio = df_limpio[df_limpio[col].fillna(0).astype(int) == 0]

    print(f"\n[CHECK 5] filas totales: {len(df_eval_real)} | filas limpias sin DNF/DNS/DSQ: {len(df_limpio)}")

    if len(df_limpio) >= 5:
        print("[CHECK 5b] correlaciones en subset limpio")
        corr_pearson_real = df_limpio["q_real_observada"].corr(df_limpio["finish_time_real"], method="pearson")
        corr_spearman_real = df_limpio["q_real_observada"].corr(df_limpio["finish_time_real"], method="spearman")

        print(f"Pearson limpio   Q vs finish_time_real           : {corr_pearson_real}")
        print(f"Spearman limpio  Q vs finish_time_real           : {corr_spearman_real}")

        if "finish_time_vs_race_median" in df_limpio.columns:
            corr_pearson_med = df_limpio["q_real_observada"].corr(df_limpio["finish_time_vs_race_median"], method="pearson")
            corr_spearman_med = df_limpio["q_real_observada"].corr(df_limpio["finish_time_vs_race_median"], method="spearman")

            print(f"Pearson limpio   Q vs finish_time_vs_race_median: {corr_pearson_med}")
            print(f"Spearman limpio  Q vs finish_time_vs_race_median: {corr_spearman_med}")

    # 6) Baseline aleatorio
    print("\n[CHECK 6] baseline aleatorio")
    rng = np.random.default_rng(12345)
    q_random = rng.standard_normal(len(df_eval_real))

    corr_rand_real = pd.Series(q_random).corr(df_eval_real["finish_time_real"], method="spearman")
    print(f"Spearman q_random vs finish_time_real: {corr_rand_real}")

    if "finish_time_vs_race_median" in df_eval_real.columns:
        corr_rand_med = pd.Series(q_random).corr(df_eval_real["finish_time_vs_race_median"], method="spearman")
        print(f"Spearman q_random vs finish_time_vs_race_median: {corr_rand_med}")

    # 7) Bins por percentiles de Q
    print("\n[CHECK 7] medias por bins de Q")
    try:
        df_bins = df_eval_real.copy()
        df_bins["q_bin"] = pd.qcut(df_bins["q_real_observada"], q=5, duplicates="drop")

        cols_group = ["finish_time_real"]
        if "finish_time_vs_race_median" in df_bins.columns:
            cols_group.append("finish_time_vs_race_median")

        resumen_bins = df_bins.groupby("q_bin", observed=False)[cols_group].mean()

        with pd.option_context("display.max_columns", None, "display.width", 160):
            print(resumen_bins)
    except Exception as e:
        print(f"No se pudo calcular qcut/bins: {e}")

    print("\n####################################################################\n")


def main() -> None:
    df = pd.read_csv(DATASET_SIM_CSV)

    resultados_todos: list[pd.DataFrame] = []
    resultados_reales_todos: list[pd.DataFrame] = []

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

        entrenamiento = entrenar_rl_offline(
            df,
            configuracionRL=configuracion_entrenamiento,
            paths=paths,
        )

        print("[RUN] contenido después:", [p.name for p in paths.ruta_pares.parent.glob("*")], flush=True)

        print("\n---------------------------------------------------------------\n")
        print("Métricas del regresor Q:", entrenamiento["metricas_regresor"])
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

        imprimir_resumen_evaluacion(resultados_test)
        resultados_todos.append(resultados_test)

        # ---------------- EVALUACIÓN REAL ----------------
        print(f"\n--- Evaluación en escenario real ({nombre_modelo}) ---\n")

        df_eval_real = evaluar_q_en_escenario_real(
            df=entrenamiento["df_test"],
            X=entrenamiento["X_test_estado"],
            modelo=entrenamiento["modelo"],
            representacion_accion=entrenamiento["representacion_accion"],
            nombre_modelo=nombre_modelo,
        )

        resumen_real = resumen_evaluacion_real(df_eval_real)

        print("Resumen evaluación real:")
        for k, v in resumen_real.items():
            print(f"{k}: {v}")

        # ---------------- CHECKS EXTRA ----------------
        imprimir_checks_evaluacion_real(
            df_eval_real,
            nombre_modelo=nombre_modelo,
        )

        resultados_reales_todos.append(df_eval_real)

        print("\n===============================================================\n")

    # ---------------- RESUMEN SIMULADO ----------------
    if len(resultados_todos) > 0:
        df_all = pd.concat(resultados_todos, ignore_index=True)

        print("================ RESUMEN DE TODOS LOS MODELOS =================")

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
    else:
        print("No se generaron resultados simulados.")

    # ---------------- RESUMEN REAL ----------------
    if len(resultados_reales_todos) > 0:
        df_real_all = pd.concat(resultados_reales_todos, ignore_index=True)

        print("=============== RESUMEN EVALUACIÓN REAL =================")

        grp_real = df_real_all.groupby("modelo_q", dropna=False)

        resumen_real_all = grp_real.agg(
            n=("q_real_observada", "size"),
            pearson_q_vs_finish_time_real=(
                "q_real_observada",
                lambda s: s.corr(df_real_all.loc[s.index, "finish_time_real"], method="pearson"),
            ),
            spearman_q_vs_finish_time_real=(
                "q_real_observada",
                lambda s: s.corr(df_real_all.loc[s.index, "finish_time_real"], method="spearman"),
            ),
            pearson_q_vs_finish_time_vs_race_median=(
                "q_real_observada",
                lambda s: s.corr(df_real_all.loc[s.index, "finish_time_vs_race_median"], method="pearson"),
            ),
            spearman_q_vs_finish_time_vs_race_median=(
                "q_real_observada",
                lambda s: s.corr(df_real_all.loc[s.index, "finish_time_vs_race_median"], method="spearman"),
            ),
        ).reset_index()

        with pd.option_context("display.max_columns", None, "display.width", 160):
            print(resumen_real_all)

        print("\n=======================================================\n")
    else:
        print("No se generaron resultados reales.")


if __name__ == "__main__":
    main()