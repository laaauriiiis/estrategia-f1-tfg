"""
evaluacion_sim.py

Evaluación empírica del simulador frente a tiempos reales observados.

La validación consiste en simular la estrategia real observada en cada
observación piloto-carrera y comparar el tiempo simulado con el tiempo real.
"""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd

from estrategia_f1.sim.simulador import simular_tiempo_carrera


COMPUESTOS_VALIDOS = {"SOFT", "MEDIUM", "HARD"}


# Helpers privados -----------------------------------------------------------------------------------------------------
def _obtener_valor(fila: pd.Series, key: str, default="N/A"):
    valor = fila.get(key, default)
    if valor is None:
        return default
    if isinstance(valor, float) and np.isnan(valor):
        return default
    return valor


def _formatear_valor(valor, decimales: int = 2) -> str:
    try:
        if valor is None or (isinstance(valor, float) and np.isnan(valor)):
            return "N/A"
        return f"{float(valor):.{decimales}f}"
    except Exception:
        return str(valor)


def obtener_nombre_circuito(fila: pd.Series, df_circuitos: pd.DataFrame | None) -> str:
    """
    Devuelve el nombre corto del circuito a partir del circuit_key.
    """
    if df_circuitos is None:
        return "N/A"

    if "circuit_key" not in fila.index:
        return "N/A"

    circuit_key = fila.get("circuit_key", np.nan)
    if pd.isna(circuit_key):
        return "N/A"

    if not {"circuit_key", "circuit_short_name"}.issubset(df_circuitos.columns):
        return str(circuit_key)

    filas_coincidentes = df_circuitos.loc[
        df_circuitos["circuit_key"] == circuit_key,
        "circuit_short_name",
    ]

    if len(filas_coincidentes):
        return str(filas_coincidentes.iloc[0])

    return str(circuit_key)


def _parsear_estrategia(estrategia_raw) -> list[str] | None:
    """
    Convierte strategy_compounds a list[str].

    Solo acepta compuestos slick válidos: SOFT, MEDIUM y HARD.
    """
    if estrategia_raw is None:
        return None

    if isinstance(estrategia_raw, float) and np.isnan(estrategia_raw):
        return None

    if isinstance(estrategia_raw, str):
        try:
            estrategia = ast.literal_eval(estrategia_raw)
        except Exception:
            return None
    else:
        estrategia = estrategia_raw

    if not isinstance(estrategia, (list, tuple)):
        return None

    estrategia_limpia: list[str] = []

    for compuesto in estrategia:
        compuesto_str = str(compuesto).strip().upper()

        if compuesto_str not in COMPUESTOS_VALIDOS:
            return None

        estrategia_limpia.append(compuesto_str)

    if not estrategia_limpia:
        return None

    return estrategia_limpia


def _id_gp(fila: pd.Series) -> tuple:
    """
    Identificador robusto de GP usando temporada + race_id.
    """
    return fila.get("season", np.nan), fila.get("race_id", np.nan)


# Evaluación numérica --------------------------------------------------------------------------------------------------
def evaluar_simulador_en_fila(fila: pd.Series, estrategia: list[str]) -> dict:
    """
    Calcula métricas de error del simulador frente al tiempo real para una fila.

    La estrategia utilizada debe ser la estrategia real observada en esa carrera.
    """
    tiempo_sim = simular_tiempo_carrera(fila, estrategia)
    tiempo_real = pd.to_numeric(fila.get("finish_time_s", np.nan), errors="coerce")

    tiempo_sim = float(tiempo_sim) if np.isfinite(tiempo_sim) else np.nan
    tiempo_real = float(tiempo_real) if np.isfinite(tiempo_real) else np.nan

    if not np.isfinite(tiempo_sim) or not np.isfinite(tiempo_real) or tiempo_real == 0:
        return {
            "tiempo_simulado": tiempo_sim,
            "tiempo_real": tiempo_real,
            "error": np.nan,
            "error_absoluto": np.nan,
            "error_porcentual_total": np.nan,
            "error_absoluto_porcentual": np.nan,
            "error_por_vuelta": np.nan,
        }

    error = tiempo_sim - tiempo_real
    error_absoluto = abs(error)

    n_vueltas = pd.to_numeric(fila.get("n_laps", np.nan), errors="coerce")
    if np.isfinite(n_vueltas) and int(n_vueltas) > 0:
        error_por_vuelta = error / float(n_vueltas)
    else:
        error_por_vuelta = np.nan

    return {
        "tiempo_simulado": tiempo_sim,
        "tiempo_real": tiempo_real,
        "error": float(error),
        "error_absoluto": float(error_absoluto),
        "error_porcentual_total": float((error / tiempo_real) * 100.0),
        "error_absoluto_porcentual": float((error_absoluto / tiempo_real) * 100.0),
        "error_por_vuelta": float(error_por_vuelta) if np.isfinite(error_por_vuelta) else np.nan,
    }


def evaluar_simulador_en_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evalúa el simulador sobre todas las filas del dataset usando la estrategia real observada.

    Devuelve un DataFrame con el detalle por observación piloto-carrera.
    """
    resultados: list[dict] = []

    for idx, fila in df.iterrows():
        estrategia = _parsear_estrategia(fila.get("strategy_compounds", None))

        base = {
            "index": idx,
            "season": fila.get("season", np.nan),
            "race_id": fila.get("race_id", np.nan),
            "gp_id": _id_gp(fila),
            "circuit_key": fila.get("circuit_key", np.nan),
            "n_laps": pd.to_numeric(fila.get("n_laps", np.nan), errors="coerce"),
            "wear_index": fila.get("wear_index", np.nan),
            "track_temp_cat": fila.get("track_temp_cat", np.nan),
            "weather_condition": fila.get("weather_condition", np.nan),
            "n_stints": pd.to_numeric(fila.get("n_stints", np.nan), errors="coerce"),
            "action_id": pd.to_numeric(fila.get("action_id", np.nan), errors="coerce"),
            "strategy_compounds": fila.get("strategy_compounds", np.nan),
            "estrategia": estrategia,
        }

        if estrategia is None:
            resultados.append({
                **base,
                "valida": False,
                "motivo": "strategy_compounds no válida",
                "tiempo_simulado": np.nan,
                "tiempo_real": pd.to_numeric(fila.get("finish_time_s", np.nan), errors="coerce"),
                "error": np.nan,
                "error_absoluto": np.nan,
                "error_porcentual_total": np.nan,
                "error_absoluto_porcentual": np.nan,
                "error_por_vuelta": np.nan,
            })
            continue

        evaluacion = evaluar_simulador_en_fila(fila, estrategia)

        es_valida = (
            np.isfinite(evaluacion["tiempo_simulado"])
            and np.isfinite(evaluacion["tiempo_real"])
            and np.isfinite(evaluacion["error"])
        )

        resultados.append({
            **base,
            "valida": bool(es_valida),
            "motivo": "ok" if es_valida else "simulación o tiempo real no válido",
            **evaluacion,
        })

    df_resultados = pd.DataFrame(resultados)

    # Métricas relativas dentro de cada GP.
    # Esto ayuda a validar no solo tiempos absolutos, sino también desviaciones respecto al contexto de carrera.
    df_validas = df_resultados[df_resultados["valida"]].copy()

    if not df_validas.empty:
        columnas_gp = ["season", "race_id"]

        mediana_real_gp = df_validas.groupby(columnas_gp)["tiempo_real"].transform("median")
        mediana_sim_gp = df_validas.groupby(columnas_gp)["tiempo_simulado"].transform("median")

        df_resultados.loc[df_validas.index, "tiempo_real_vs_mediana_gp"] = (
            df_validas["tiempo_real"] - mediana_real_gp
        )
        df_resultados.loc[df_validas.index, "tiempo_simulado_vs_mediana_gp"] = (
            df_validas["tiempo_simulado"] - mediana_sim_gp
        )
        df_resultados.loc[df_validas.index, "error_relativo_gp"] = (
            df_resultados.loc[df_validas.index, "tiempo_simulado_vs_mediana_gp"]
            - df_resultados.loc[df_validas.index, "tiempo_real_vs_mediana_gp"]
        )
        df_resultados.loc[df_validas.index, "error_absoluto_relativo_gp"] = (
            df_resultados.loc[df_validas.index, "error_relativo_gp"].abs()
        )

    return df_resultados


def resumir_evaluacion_simulador(df_resultados: pd.DataFrame) -> dict:
    """
    Calcula métricas agregadas de la evaluación global del simulador.
    """
    if df_resultados.empty:
        return {
            "n_total_observaciones": 0,
            "n_observaciones_validas": 0,
            "n_observaciones_invalidas": 0,
            "n_carreras_validas": 0,
            "mae_medio": np.nan,
            "mediana_error_absoluto": np.nan,
            "rmse_global": np.nan,
            "error_medio": np.nan,
            "error_absoluto_porcentual_medio": np.nan,
            "error_porcentual_medio": np.nan,
            "error_por_vuelta_medio": np.nan,
            "pearson_tiempo_real_simulado": np.nan,
            "spearman_tiempo_real_simulado": np.nan,
            "pearson_relativo_gp": np.nan,
            "spearman_relativo_gp": np.nan,
            "mae_relativo_gp_medio": np.nan,
        }

    df_validas = df_resultados[
        df_resultados["valida"]
        & df_resultados["tiempo_real"].notna()
        & df_resultados["tiempo_simulado"].notna()
    ].copy()

    if df_validas.empty:
        return {
            "n_total_observaciones": int(len(df_resultados)),
            "n_observaciones_validas": 0,
            "n_observaciones_invalidas": int(len(df_resultados)),
            "n_carreras_validas": 0,
            "mae_medio": np.nan,
            "mediana_error_absoluto": np.nan,
            "rmse_global": np.nan,
            "error_medio": np.nan,
            "error_absoluto_porcentual_medio": np.nan,
            "error_porcentual_medio": np.nan,
            "error_por_vuelta_medio": np.nan,
            "pearson_tiempo_real_simulado": np.nan,
            "spearman_tiempo_real_simulado": np.nan,
            "pearson_relativo_gp": np.nan,
            "spearman_relativo_gp": np.nan,
            "mae_relativo_gp_medio": np.nan,
        }

    errores = pd.to_numeric(df_validas["error"], errors="coerce")
    errores_abs = pd.to_numeric(df_validas["error_absoluto"], errors="coerce")
    tiempos_reales = pd.to_numeric(df_validas["tiempo_real"], errors="coerce")
    tiempos_simulados = pd.to_numeric(df_validas["tiempo_simulado"], errors="coerce")
    errores_pct = pd.to_numeric(df_validas["error_porcentual_total"], errors="coerce")
    errores_abs_pct = pd.to_numeric(df_validas["error_absoluto_porcentual"], errors="coerce")
    errores_por_vuelta = pd.to_numeric(df_validas["error_por_vuelta"], errors="coerce")

    rmse_global = np.sqrt(np.nanmean(np.square(errores)))

    if len(df_validas) >= 2:
        pearson = tiempos_reales.corr(tiempos_simulados, method="pearson")
        spearman = tiempos_reales.corr(tiempos_simulados, method="spearman")
    else:
        pearson = np.nan
        spearman = np.nan

    # Validación relativa dentro de cada GP.
    if {
        "tiempo_real_vs_mediana_gp",
        "tiempo_simulado_vs_mediana_gp",
        "error_absoluto_relativo_gp",
    }.issubset(df_validas.columns):
        real_rel = pd.to_numeric(df_validas["tiempo_real_vs_mediana_gp"], errors="coerce")
        sim_rel = pd.to_numeric(df_validas["tiempo_simulado_vs_mediana_gp"], errors="coerce")
        error_abs_rel_gp = pd.to_numeric(df_validas["error_absoluto_relativo_gp"], errors="coerce")

        if real_rel.notna().sum() >= 2 and sim_rel.notna().sum() >= 2:
            pearson_rel_gp = real_rel.corr(sim_rel, method="pearson")
            spearman_rel_gp = real_rel.corr(sim_rel, method="spearman")
        else:
            pearson_rel_gp = np.nan
            spearman_rel_gp = np.nan

        mae_relativo_gp_medio = np.nanmean(error_abs_rel_gp)
    else:
        pearson_rel_gp = np.nan
        spearman_rel_gp = np.nan
        mae_relativo_gp_medio = np.nan

    n_carreras_validas = (
        df_validas[["season", "race_id"]]
        .drop_duplicates()
        .shape[0]
    )

    return {
        "n_total_observaciones": int(len(df_resultados)),
        "n_observaciones_validas": int(len(df_validas)),
        "n_observaciones_invalidas": int(len(df_resultados) - len(df_validas)),
        "n_carreras_validas": int(n_carreras_validas),
        "mae_medio": float(np.nanmean(errores_abs)),
        "mediana_error_absoluto": float(np.nanmedian(errores_abs)),
        "rmse_global": float(rmse_global),
        "error_medio": float(np.nanmean(errores)),
        "error_absoluto_porcentual_medio": float(np.nanmean(errores_abs_pct)),
        "error_porcentual_medio": float(np.nanmean(errores_pct)),
        "error_por_vuelta_medio": float(np.nanmean(errores_por_vuelta)),
        "pearson_tiempo_real_simulado": float(pearson) if pd.notna(pearson) else np.nan,
        "spearman_tiempo_real_simulado": float(spearman) if pd.notna(spearman) else np.nan,
        "pearson_relativo_gp": float(pearson_rel_gp) if pd.notna(pearson_rel_gp) else np.nan,
        "spearman_relativo_gp": float(spearman_rel_gp) if pd.notna(spearman_rel_gp) else np.nan,
        "mae_relativo_gp_medio": float(mae_relativo_gp_medio) if np.isfinite(mae_relativo_gp_medio) else np.nan,
    }


def resumir_evaluacion_por_gp(df_resultados: pd.DataFrame) -> pd.DataFrame:
    """
    Resume la evaluación del simulador por carrera, usando season + race_id.
    """
    df_validas = df_resultados[
        df_resultados["valida"]
        & df_resultados["tiempo_real"].notna()
        & df_resultados["tiempo_simulado"].notna()
    ].copy()

    if df_validas.empty:
        return pd.DataFrame(columns=[
            "season",
            "race_id",
            "n_observaciones",
            "mae_medio",
            "mediana_error_absoluto",
            "rmse_global",
            "error_medio",
            "error_absoluto_porcentual_medio",
            "pearson_tiempo_real_simulado",
            "spearman_tiempo_real_simulado",
        ])

    filas: list[dict] = []

    for (season, race_id), grupo in df_validas.groupby(["season", "race_id"], dropna=False):
        errores = pd.to_numeric(grupo["error"], errors="coerce")
        errores_abs = pd.to_numeric(grupo["error_absoluto"], errors="coerce")
        errores_abs_pct = pd.to_numeric(grupo["error_absoluto_porcentual"], errors="coerce")
        tiempos_reales = pd.to_numeric(grupo["tiempo_real"], errors="coerce")
        tiempos_simulados = pd.to_numeric(grupo["tiempo_simulado"], errors="coerce")

        if len(grupo) >= 2:
            pearson = tiempos_reales.corr(tiempos_simulados, method="pearson")
            spearman = tiempos_reales.corr(tiempos_simulados, method="spearman")
        else:
            pearson = np.nan
            spearman = np.nan

        filas.append({
            "season": season,
            "race_id": race_id,
            "n_observaciones": int(len(grupo)),
            "mae_medio": float(np.nanmean(errores_abs)),
            "mediana_error_absoluto": float(np.nanmedian(errores_abs)),
            "rmse_global": float(np.sqrt(np.nanmean(np.square(errores)))),
            "error_medio": float(np.nanmean(errores)),
            "error_absoluto_porcentual_medio": float(np.nanmean(errores_abs_pct)),
            "pearson_tiempo_real_simulado": float(pearson) if pd.notna(pearson) else np.nan,
            "spearman_tiempo_real_simulado": float(spearman) if pd.notna(spearman) else np.nan,
        })

    return (
        pd.DataFrame(filas)
        .sort_values(by="mae_medio", ascending=True)
        .reset_index(drop=True)
    )


def resumir_evaluacion_por_grupo(df_resultados: pd.DataFrame, columna: str) -> pd.DataFrame:
    """
    Resume la evaluación del simulador agrupando por una columna del dataset.
    """
    if columna not in df_resultados.columns:
        raise KeyError(f"La columna '{columna}' no existe en df_resultados")

    df_validas = df_resultados[
        df_resultados["valida"]
        & df_resultados["tiempo_real"].notna()
        & df_resultados["tiempo_simulado"].notna()
    ].copy()

    if df_validas.empty:
        return pd.DataFrame(columns=[
            columna,
            "n_observaciones",
            "mae_medio",
            "mediana_error_absoluto",
            "rmse_global",
            "error_medio",
            "error_absoluto_porcentual_medio",
            "error_por_vuelta_medio",
            "pearson_tiempo_real_simulado",
            "spearman_tiempo_real_simulado",
        ])

    filas_resumen: list[dict] = []

    for valor, grupo in df_validas.groupby(columna, dropna=False):
        errores = pd.to_numeric(grupo["error"], errors="coerce")
        errores_abs = pd.to_numeric(grupo["error_absoluto"], errors="coerce")
        errores_abs_pct = pd.to_numeric(grupo["error_absoluto_porcentual"], errors="coerce")
        errores_por_vuelta = pd.to_numeric(grupo["error_por_vuelta"], errors="coerce")
        tiempos_reales = pd.to_numeric(grupo["tiempo_real"], errors="coerce")
        tiempos_simulados = pd.to_numeric(grupo["tiempo_simulado"], errors="coerce")

        if len(grupo) >= 2:
            pearson = tiempos_reales.corr(tiempos_simulados, method="pearson")
            spearman = tiempos_reales.corr(tiempos_simulados, method="spearman")
        else:
            pearson = np.nan
            spearman = np.nan

        filas_resumen.append({
            columna: valor,
            "n_observaciones": int(len(grupo)),
            "mae_medio": float(np.nanmean(errores_abs)),
            "mediana_error_absoluto": float(np.nanmedian(errores_abs)),
            "rmse_global": float(np.sqrt(np.nanmean(np.square(errores)))),
            "error_medio": float(np.nanmean(errores)),
            "error_absoluto_porcentual_medio": float(np.nanmean(errores_abs_pct)),
            "error_por_vuelta_medio": float(np.nanmean(errores_por_vuelta)),
            "pearson_tiempo_real_simulado": float(pearson) if pd.notna(pearson) else np.nan,
            "spearman_tiempo_real_simulado": float(spearman) if pd.notna(spearman) else np.nan,
        })

    return (
        pd.DataFrame(filas_resumen)
        .sort_values(by="mae_medio", ascending=True)
        .reset_index(drop=True)
    )


def guardar_evaluacion_simulador(
    df_resultados: pd.DataFrame,
    ruta_detalle: str,
    ruta_resumen_gp: str | None = None,
) -> None:
    """
    Guarda los resultados de validación en CSV para usarlos en la memoria/anexos.
    """
    df_resultados.to_csv(ruta_detalle, index=False)

    if ruta_resumen_gp is not None:
        resumen_gp = resumir_evaluacion_por_gp(df_resultados)
        resumen_gp.to_csv(ruta_resumen_gp, index=False)


# Output ---------------------------------------------------------------------------------------------------------------
def imprimir_resumen_simulador(
    fila: pd.Series,
    estrategia: list[str],
    evaluacion: dict,
    df_circuitos: pd.DataFrame | None = None,
) -> None:
    """
    Imprime de manera legible una evaluación del simulador.
    """
    temporada = _obtener_valor(fila, "season")
    id_carrera = _obtener_valor(fila, "race_id")
    circuito = obtener_nombre_circuito(fila, df_circuitos)

    n_vueltas = _obtener_valor(fila, "n_laps")
    pit_loss = pd.to_numeric(fila.get("pit_loss_s", np.nan), errors="coerce")
    pit_loss = float(pit_loss) if np.isfinite(pit_loss) else np.nan

    desgaste = _obtener_valor(fila, "wear_index")
    temp_cat = _obtener_valor(fila, "track_temp_cat")
    meteo = _obtener_valor(fila, "weather_condition")
    prob_sc = _obtener_valor(fila, "sc_prob")

    print("\n================ EVALUACIÓN DEL SIMULADOR ================")
    print(f"Temporada {temporada} | Carrera {id_carrera} | {circuito}")
    print("----------------------------------------------------------")
    print(f"{n_vueltas} vueltas | {_formatear_valor(pit_loss)} s de pit loss | {len(estrategia) - 1} paradas")
    print(
        f"Temperatura {temp_cat} | Desgaste {desgaste} | "
        f"Carrera en {meteo} | {_formatear_valor(prob_sc)} probabilidad de SC"
    )
    print("----------------------------------------------------------")
    print(f"Estrategia real simulada: {estrategia}")
    print("----------------------------------------------------------")
    print(f"Tiempo real: {_formatear_valor(evaluacion.get('tiempo_real'))} s")
    print(f"Tiempo simulado: {_formatear_valor(evaluacion.get('tiempo_simulado'))} s")
    print("----------------------------------------------------------")
    print(f"Error total: {_formatear_valor(evaluacion.get('error'))} s")
    print(f"Error absoluto: {_formatear_valor(evaluacion.get('error_absoluto'))} s")
    print(f"Error por vuelta: {_formatear_valor(evaluacion.get('error_por_vuelta'), 3)} s/vuelta")
    print(f"Error relativo total: {_formatear_valor(evaluacion.get('error_porcentual_total'))} %")
    print("==========================================================\n")


def imprimir_resumen_global_simulador(resumen: dict) -> None:
    """
    Imprime de manera legible el resumen agregado de la evaluación del simulador.
    """
    print("\n================ RESUMEN GLOBAL DEL SIMULADOR ================")
    print(f"Total de observaciones: {resumen.get('n_total_observaciones', 0)}")
    print(f"Observaciones válidas: {resumen.get('n_observaciones_validas', 0)}")
    print(f"Observaciones inválidas: {resumen.get('n_observaciones_invalidas', 0)}")
    print(f"Carreras/GP válidos: {resumen.get('n_carreras_validas', 0)}")
    print("--------------------------------------------------------------")
    print(f"MAE medio: {_formatear_valor(resumen.get('mae_medio'))} s")
    print(f"Mediana error absoluto: {_formatear_valor(resumen.get('mediana_error_absoluto'))} s")
    print(f"RMSE global: {_formatear_valor(resumen.get('rmse_global'))} s")
    print(f"Error medio: {_formatear_valor(resumen.get('error_medio'))} s")
    print(f"Error relativo absoluto medio: {_formatear_valor(resumen.get('error_absoluto_porcentual_medio'))} %")
    print(f"Error relativo medio: {_formatear_valor(resumen.get('error_porcentual_medio'))} %")
    print(f"Error por vuelta medio: {_formatear_valor(resumen.get('error_por_vuelta_medio'), 3)} s/vuelta")
    print("--------------------------------------------------------------")
    print(f"Pearson tiempo real/simulado: {_formatear_valor(resumen.get('pearson_tiempo_real_simulado'), 4)}")
    print(f"Spearman tiempo real/simulado: {_formatear_valor(resumen.get('spearman_tiempo_real_simulado'), 4)}")
    print("--------------------------------------------------------------")
    print(f"MAE relativo por GP medio: {_formatear_valor(resumen.get('mae_relativo_gp_medio'))} s")
    print(f"Pearson relativo por GP: {_formatear_valor(resumen.get('pearson_relativo_gp'), 4)}")
    print(f"Spearman relativo por GP: {_formatear_valor(resumen.get('spearman_relativo_gp'), 4)}")
    print("==============================================================\n")