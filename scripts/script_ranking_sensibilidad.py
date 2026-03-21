"""
script_sensibilidad_ranking.py
TODO
"""
from __future__ import annotations

import itertools
from contextlib import contextmanager

import pandas as pd

from estrategia_f1.config import DATASET_SIM_CSV, SEED
import estrategia_f1.config as cfg
from estrategia_f1.sim.simulador import simular_tiempo_carrera


def generar_estrategias() -> list[list[str]]:
    """
    Genera estrategias candidatas con 2, 3 y 4 stints.
    """
    compuestos = cfg.COMPUESTOS
    estrategias: list[list[str]] = []

    for k in range(cfg.MIN_STINTS, cfg.MAX_STINTS + 1):
        for estrategia in itertools.product(compuestos, repeat=k):
            estrategias.append(list(estrategia))

    return estrategias


def rankear_estrategias(fila: pd.Series, estrategias: list[list[str]]) -> pd.DataFrame:
    """
    Simula todas las estrategias válidas para una fila y devuelve su ranking.
    """
    resultados = []

    for estrategia in estrategias:
        try:
            tiempo = simular_tiempo_carrera(fila, estrategia)
        except Exception:
            continue

        if pd.isna(tiempo):
            continue

        resultados.append({
            "estrategia": tuple(estrategia),
            "tiempo_simulado": float(tiempo),
        })

    df_rank = pd.DataFrame(resultados)

    if df_rank.empty:
        return df_rank

    df_rank = df_rank.sort_values("tiempo_simulado", ascending=True).reset_index(drop=True)
    df_rank["rank"] = range(1, len(df_rank) + 1)
    return df_rank


def calcular_spearman(df_base: pd.DataFrame, df_var: pd.DataFrame) -> float:
    """
    Calcula correlación de Spearman entre rankings base y variante.
    """
    df_merge = df_base[["estrategia", "rank"]].merge(
        df_var[["estrategia", "rank"]],
        on="estrategia",
        how="inner",
        suffixes=("_base", "_var"),
    )

    if len(df_merge) < 2:
        return float("nan")

    return float(df_merge["rank_base"].corr(df_merge["rank_var"], method="spearman"))


def calcular_top3_overlap(df_base: pd.DataFrame, df_var: pd.DataFrame) -> float:
    """
    Calcula el solapamiento del top 3 entre ranking base y variante.
    """
    top3_base = set(df_base.head(3)["estrategia"])
    top3_var = set(df_var.head(3)["estrategia"])

    if not top3_base:
        return 0.0

    return len(top3_base.intersection(top3_var)) / 3.0


def calcular_pct_tiempos_cambiados(df_base: pd.DataFrame, df_var: pd.DataFrame, tol: float = 1e-9) -> float:
    """
    Calcula el porcentaje de estrategias cuyo tiempo cambia entre ranking base y variante.
    """
    df_merge = df_base[["estrategia", "tiempo_simulado"]].merge(
        df_var[["estrategia", "tiempo_simulado"]],
        on="estrategia",
        how="inner",
        suffixes=("_base", "_var"),
    )

    if df_merge.empty:
        return 0.0

    cambios = (df_merge["tiempo_simulado_base"] - df_merge["tiempo_simulado_var"]).abs() > tol
    return float(cambios.mean() * 100.0)


@contextmanager
def parchear_config(**kwargs):
    """
    Cambia temporalmente parámetros del config durante la simulación.
    """
    valores_originales = {}

    for key, value in kwargs.items():
        valores_originales[key] = getattr(cfg, key)
        setattr(cfg, key, value)

    try:
        yield
    finally:
        for key, value in valores_originales.items():
            setattr(cfg, key, value)


def aplicar_perturbacion_fila(fila: pd.Series, nombre_escenario: str) -> pd.Series:
    """
    Aplica perturbaciones directamente sobre la fila cuando el parámetro relevante
    no depende solo de config.py (por ejemplo pit_loss_s).
    """
    fila_var = fila.copy()

    if nombre_escenario == "pit_loss_menos_10":
        pit_loss = pd.to_numeric(fila_var.get("pit_loss_s", pd.NA), errors="coerce")
        if pd.notna(pit_loss):
            fila_var["pit_loss_s"] = float(pit_loss) * 0.9

    elif nombre_escenario == "pit_loss_mas_10":
        pit_loss = pd.to_numeric(fila_var.get("pit_loss_s", pd.NA), errors="coerce")
        if pd.notna(pit_loss):
            fila_var["pit_loss_s"] = float(pit_loss) * 1.1

    elif nombre_escenario == "pit_loss_menos_50":
        pit_loss = pd.to_numeric(fila_var.get("pit_loss_s", pd.NA), errors="coerce")
        if pd.notna(pit_loss):
            fila_var["pit_loss_s"] = float(pit_loss) * 0.5

    elif nombre_escenario == "pit_loss_mas_50":
        pit_loss = pd.to_numeric(fila_var.get("pit_loss_s", pd.NA), errors="coerce")
        if pd.notna(pit_loss):
            fila_var["pit_loss_s"] = float(pit_loss) * 1.5

    return fila_var


def imprimir_check_manual(
    fila: pd.Series,
    estrategias_test: list[list[str]],
    nombre_escenario: str,
    parametros_cfg: dict,
) -> None:
    """
    Imprime un check manual para comprobar que los tiempos cambian realmente.
    """
    print("\n================ CHECK MANUAL DE VALIDACIÓN ================")
    print(f"Escenario: {nombre_escenario}")
    print(f"Temporada: {fila.get('season')} | Carrera: {fila.get('race_id')}")
    print("------------------------------------------------------------")
    print("BASE")

    tiempos_base = []
    for estrategia in estrategias_test:
        tiempo = simular_tiempo_carrera(fila, estrategia)
        tiempos_base.append((tuple(estrategia), tiempo))
        print(f"{tuple(estrategia)} -> {tiempo:.3f}")

    fila_var = aplicar_perturbacion_fila(fila, nombre_escenario)

    print("------------------------------------------------------------")
    print("VARIANTE")

    with parchear_config(**parametros_cfg):
        for estrategia in estrategias_test:
            tiempo = simular_tiempo_carrera(fila_var, estrategia)
            print(f"{tuple(estrategia)} -> {tiempo:.3f}")

    print("============================================================\n")


def evaluar_escenario(
    df_muestra: pd.DataFrame,
    estrategias: list[list[str]],
    nombre_escenario: str,
    **parametros_cfg,
) -> pd.DataFrame:
    """
    Evalúa un escenario perturbado frente al ranking base.
    """
    filas_resultado = []

    for idx, fila in df_muestra.iterrows():
        df_base = rankear_estrategias(fila, estrategias)

        if df_base.empty:
            continue

        fila_var = aplicar_perturbacion_fila(fila, nombre_escenario)

        with parchear_config(**parametros_cfg):
            df_var = rankear_estrategias(fila_var, estrategias)

        if df_var.empty:
            continue

        mejor_base = df_base.iloc[0]["estrategia"]
        mejor_var = df_var.iloc[0]["estrategia"]

        filas_resultado.append({
            "index": idx,
            "race_id": fila.get("race_id"),
            "season": fila.get("season"),
            "escenario": nombre_escenario,
            "top1_igual": mejor_base == mejor_var,
            "top3_overlap": calcular_top3_overlap(df_base, df_var),
            "spearman": calcular_spearman(df_base, df_var),
            "pct_tiempos_cambiados": calcular_pct_tiempos_cambiados(df_base, df_var),
            "mejor_base": mejor_base,
            "mejor_var": mejor_var,
        })

    return pd.DataFrame(filas_resultado)


def resumir_resultados(df_resultados: pd.DataFrame) -> pd.DataFrame:
    """
    Resume los resultados por escenario.
    """
    if df_resultados.empty:
        return pd.DataFrame()

    resumen = (
        df_resultados
        .groupby("escenario", as_index=False)
        .agg(
            n=("index", "count"),
            top1_estable_pct=("top1_igual", "mean"),
            top3_overlap_medio=("top3_overlap", "mean"),
            spearman_medio=("spearman", "mean"),
            pct_tiempos_cambiados_medio=("pct_tiempos_cambiados", "mean"),
        )
    )

    resumen["top1_estable_pct"] = resumen["top1_estable_pct"] * 100.0
    return resumen


def main() -> None:
    df = pd.read_csv(DATASET_SIM_CSV)

    df_muestra = (
        df.drop_duplicates(subset=["season", "race_id", "circuit_key"])
        .sample(30, random_state=SEED)
        .copy()
    )

    estrategias = generar_estrategias()

    escenarios = [
        ("wear_baja_menos_10", {"WEAR_MAP": {"baja": 0.90, "media": 1.10, "alta": 1.20}}),
        ("wear_baja_mas_10", {"WEAR_MAP": {"baja": 1.10, "media": 1.10, "alta": 1.20}}),
        ("pen_vida_menos_10", {"PENALIZACION_VIDA_UTIL": cfg.PENALIZACION_VIDA_UTIL * 0.9}),
        ("pen_vida_mas_10", {"PENALIZACION_VIDA_UTIL": cfg.PENALIZACION_VIDA_UTIL * 1.1}),
        ("pit_loss_menos_10", {}),
        ("pit_loss_mas_10", {}),
        ("pen_stint_menos_10", {"PENALIZACION_STINT": cfg.PENALIZACION_STINT * 0.9}),
        ("pen_stint_mas_10", {"PENALIZACION_STINT": cfg.PENALIZACION_STINT * 1.1}),
    ]

    # Checks extremos para validar que el script reacciona
    escenarios_extremos = [
        ("pen_vida_mas_100", {"PENALIZACION_VIDA_UTIL": cfg.PENALIZACION_VIDA_UTIL * 2.0}),
        ("pen_stint_mas_100", {"PENALIZACION_STINT": cfg.PENALIZACION_STINT * 2.0}),
        ("pit_loss_mas_50", {}),
    ]

    # Check manual sobre una fila concreta
    fila_check = df_muestra.iloc[0].copy()
    estrategias_test = [
        ["MEDIUM", "HARD"],
        ["SOFT", "HARD"],
        ["HARD", "MEDIUM", "HARD"],
    ]

    imprimir_check_manual(
        fila=fila_check,
        estrategias_test=estrategias_test,
        nombre_escenario="pit_loss_mas_50",
        parametros_cfg={},
    )

    imprimir_check_manual(
        fila=fila_check,
        estrategias_test=estrategias_test,
        nombre_escenario="pen_vida_mas_100",
        parametros_cfg={"PENALIZACION_VIDA_UTIL": cfg.PENALIZACION_VIDA_UTIL * 2.0},
    )

    resultados = []

    print("=== ESCENARIOS PRINCIPALES ===")
    for nombre, params in escenarios:
        print(f"Evaluando escenario: {nombre}")
        df_escenario = evaluar_escenario(
            df_muestra=df_muestra,
            estrategias=estrategias,
            nombre_escenario=nombre,
            **params,
        )
        resultados.append(df_escenario)

    print("=== ESCENARIOS EXTREMOS DE VALIDACIÓN ===")
    for nombre, params in escenarios_extremos:
        print(f"Evaluando escenario: {nombre}")
        df_escenario = evaluar_escenario(
            df_muestra=df_muestra,
            estrategias=estrategias,
            nombre_escenario=nombre,
            **params,
        )
        resultados.append(df_escenario)

    df_resultados = pd.concat(resultados, ignore_index=True)
    resumen = resumir_resultados(df_resultados)

    print("\n================ SENSIBILIDAD DEL RANKING ================")
    print(resumen.to_string(index=False))
    print("==========================================================\n")

    print("Ejemplos donde cambia la mejor estrategia:")
    cambios = df_resultados[~df_resultados["top1_igual"]].copy()
    if cambios.empty:
        print("No hubo cambios en la mejor estrategia.")
    else:
        print(
            cambios[
                [
                    "escenario",
                    "season",
                    "race_id",
                    "mejor_base",
                    "mejor_var",
                    "top3_overlap",
                    "spearman",
                    "pct_tiempos_cambiados",
                ]
            ].head(20).to_string(index=False)
        )

if __name__ == "__main__":
    main()