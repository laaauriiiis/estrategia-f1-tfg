"""
script_sensibilidad_ranking.py

Análisis de sensibilidad del ranking de estrategias del simulador.

[MEMORIA]
"""

# IMPORTS
from __future__ import annotations
import itertools
from contextlib import contextmanager
import pandas as pd
from estrategia_f1.config import DATASET_EXPERIMENTAL_CSV, SEED
import estrategia_f1.config as cfg
from estrategia_f1.sim.simulador import simular_tiempo_carrera

# ESPACIO DE ESTRATEGIAS -----------------------------------------------------------------------------------------------
def generar_estrategias() -> list[list[str]]:
    """
    Genera todas las estrategias candidatas posibles del simulador.

    Se construyen combinaciones de compuestos con entre
    MIN_STINTS y MAX_STINTS stints.

    Returns
    -------
    list[list[str]]
        Lista de estrategias candidatas representadas
        como secuencias de compuestos.
    """
    compuestos = cfg.COMPUESTOS
    estrategias: list[list[str]] = []

    for k in range(cfg.MIN_STINTS, cfg.MAX_STINTS + 1):
        for estrategia in itertools.product(compuestos, repeat=k):
            estrategias.append(list(estrategia))

    return estrategias

def rankear_estrategias(fila: pd.Series, estrategias: list[list[str]]) -> pd.DataFrame:
    """
    Simula y ordena estrategias válidas para una carrera concreta.

    Parámetros
    ----------
    fila : pd.Series
        Observación piloto-carrera utilizada como contexto
        de simulación.
    estrategias : list[list[str]]
        Estrategias candidatas a evaluar.

    Returns
    -------
    pd.DataFrame
        Ranking de estrategias válidas ordenadas por
        tiempo simulado ascendente.
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

# MÉTRICAS -------------------------------------------------------------------------------------------------------------
def calcular_spearman(df_base: pd.DataFrame, df_var: pd.DataFrame) -> float:
    """
    Calcula la correlación de Spearman entre dos rankings de estrategias.

    Parámetros
    ----------
    df_base : pd.DataFrame
        Ranking base de estrategias.
    df_var : pd.DataFrame
        Ranking obtenido tras aplicar una perturbación
        o variante del simulador.

    Returns
    -------
    float
        Correlación de Spearman entre ambos rankings.
        Devuelve NaN si no hay suficientes estrategias
        comunes para calcular la correlación.
    """
    df_merge = df_base[["estrategia", "rank"]].merge(
        df_var[["estrategia", "rank"]],
        on="estrategia",
        how="inner",
        suffixes=("_base", "_var"),
    )

    # La correlación no es válida con menos de dos observaciones
    if len(df_merge) < 2:
        return float("nan")

    return float(df_merge["rank_base"].corr(df_merge["rank_var"], method="spearman"))

def calcular_top3_overlap(df_base: pd.DataFrame, df_var: pd.DataFrame) -> float:
    """
    Calcula el solapamiento del top-3 entre dos rankings de estrategias.

    Parámetros
    ----------
    df_base : pd.DataFrame
        Ranking base de estrategias.
    df_var : pd.DataFrame
        Ranking obtenido tras aplicar una perturbación
        o variante del simulador.

    Returns
    -------
    float
        Proporción de estrategias compartidas entre
        los tres primeros puestos de ambos rankings.
    """

    top3_base = set(df_base.head(3)["estrategia"])
    top3_var = set(df_var.head(3)["estrategia"])

    if not top3_base:
        return 0.0

    return len(top3_base.intersection(top3_var)) / 3.0

def calcular_pct_tiempos_cambiados(df_base: pd.DataFrame, df_var: pd.DataFrame, tol: float = 1e-9) -> float:
    """
    Calcula el porcentaje de estrategias cuyo tiempo simulado cambia
    entre el ranking base y una variante del simulador.

    Parámetros
    ----------
    df_base : pd.DataFrame
        Ranking base de estrategias.
    df_var : pd.DataFrame
        Ranking obtenido tras aplicar una perturbación
        o variante del simulador.
    tol : float, optional
        Tolerancia mínima para considerar que un tiempo
        ha cambiado entre ambos rankings.

    Returns
    -------
    float
        Porcentaje de estrategias cuyo tiempo simulado
        difiere entre ambos rankings.
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

def resumir_resultados(df_resultados: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un resumen agregado del análisis de sensibilidad.

    Las métricas se agrupan por escenario perturbado
    para analizar la estabilidad media de los rankings.

    Parámetros
    ----------
    df_resultados : pd.DataFrame
        Resultados detallados del análisis de sensibilidad.

    Returns
    -------
    pd.DataFrame
        Resumen agregado por escenario con métricas
        medias de estabilidad y variación del ranking.
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

# MODIFICACIÓN TEMPORAL ------------------------------------------------------------------------------------------------
@contextmanager
def parchear_config(**kwargs):
    """
    Modifica temporalmente parámetros de configuración del simulador.

    Los valores originales se restauran automáticamente
    al finalizar el bloque de contexto.

    Parámetros
    ----------
    **kwargs
        Parámetros de configuración y valores temporales
        que se desean aplicar sobre cfg.
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
    Aplica perturbaciones específicas directamente sobre una fila del dataset.

    Se utiliza para modificar parámetros dependientes de la observación,
    como el pit loss, que no se controlan únicamente desde config.py.

    Parámetros
    ----------
    fila : pd.Series
        Observación piloto-carrera original.
    nombre_escenario : str
        Identificador del escenario de perturbación aplicado.

    Returns
    -------
    pd.Series
        Copia de la fila con las perturbaciones aplicadas.
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

# EVALUACIÓN -----------------------------------------------------------------------------------------------------------
def imprimir_check_manual(fila: pd.Series, estrategias_test: list[list[str]], nombre_escenario: str, parametros_cfg: dict) -> None:
    """
    Imprime una comprobación manual de un escenario perturbado.

    Permite comparar visualmente los tiempos simulados antes
    y después de aplicar una perturbación sobre una misma
    observación y un conjunto reducido de estrategias.

    Parámetros
    ----------
    fila : pd.Series
        Observación piloto-carrera utilizada como caso de prueba.
    estrategias_test : list[list[str]]
        Estrategias concretas que se desean comparar.
    nombre_escenario : str
        Identificador del escenario perturbado.
    parametros_cfg : dict
        Parámetros temporales de configuración aplicados
        durante la simulación de la variante.

    Returns
    -------
    None
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

def evaluar_escenario(df_muestra: pd.DataFrame, estrategias: list[list[str]], nombre_escenario: str, **parametros_cfg) -> pd.DataFrame:
    """
    Evalúa la estabilidad del ranking ante un escenario perturbado.

    Para cada observación del dataset:
    - Se calcula el ranking base de estrategias.
    - Se aplica una perturbación sobre la simulación.
    - Se recalcula el ranking perturbado.
    - Se comparan ambos rankings mediante métricas de estabilidad.

    Parámetros
    ----------
    df_muestra : pd.DataFrame
        Subconjunto de observaciones piloto-carrera utilizado
        en el análisis de sensibilidad.
    estrategias : list[list[str]]
        Estrategias candidatas evaluadas por el simulador.
    nombre_escenario : str
        Identificador del escenario perturbado.
    **parametros_cfg
        Parámetros temporales aplicados sobre config.py
        durante la simulación perturbada.

    Returns
    -------
    pd.DataFrame
        Resultados agregados del análisis de estabilidad
        para cada observación evaluada.
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

# SCRIPT ---------------------------------------------------------------------------------------------------------------
def main() -> None:
    """
    Ejecuta el análisis completo de sensibilidad del ranking.

    El proceso incluye:
    1. Carga del dataset experimental.
    2. Selección de una muestra representativa de carreras.
    3. Generación del espacio de estrategias candidatas.
    4. Evaluación de escenarios perturbados del simulador.
    5. Comparación entre rankings base y rankings modificados.
    6. Generación de métricas agregadas de estabilidad.
    7. Impresión de ejemplos donde cambia la estrategia óptima.
    """

    df = pd.read_csv(DATASET_EXPERIMENTAL_CSV)

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