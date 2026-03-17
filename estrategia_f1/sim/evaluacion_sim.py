"""
evaluacion_sim.py
TODO
"""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd

from estrategia_f1.sim.simulador import simular_tiempo_carrera


# Helpers privados------------------------------------------------------------------------------------------------------
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

    filas_coincidentes = df_circuitos.loc[df_circuitos["circuit_key"] == circuit_key, "circuit_short_name"]

    # Si hay alguna fila coincidente, devolvemos la primera
    if len(filas_coincidentes):
        return filas_coincidentes.iloc[0]
    else:
        return str(circuit_key)

def _parsear_estrategia(estrategia_raw) -> list[str] | None:
    """
    Convierte strategy_compounds a list[str] si viene en formato string/lista.
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
        if compuesto_str:
            estrategia_limpia.append(compuesto_str)

    if not estrategia_limpia:
        return None

    return estrategia_limpia

# Evaluación numérica---------------------------------------------------------------------------------------------------
def evaluar_simulador_en_fila(fila: pd.Series, estrategia: list[str]) -> dict:
    """
    Calcula métricas de error del simulador frente al tiempo real para una única carrera y estrategia real observada.
    """
    tiempo_sim = simular_tiempo_carrera(fila, estrategia)

    tiempo_real = pd.to_numeric(fila.get("finish_time_s", np.nan), errors="coerce")

    if np.isfinite(tiempo_real):
        tiempo_real = float(tiempo_real)
    else:
        tiempo_real = np.nan

    # Si no se puede evaluar la carrera, devuelve los parámetros como NaNs
    if not np.isfinite(tiempo_sim) or not np.isfinite(tiempo_real) or tiempo_real == 0:
        return {
            "tiempo_simulado": float(tiempo_sim) if np.isfinite(tiempo_sim) else np.nan,
            "tiempo_real": float(tiempo_real) if np.isfinite(tiempo_real) else np.nan,
            "error": np.nan,
            "mae": np.nan,
            "error_porcentual_total": np.nan,
            "error_absoluto_porcentual": np.nan,
            "error_por_vuelta": np.nan,
        }

    error = float(tiempo_sim - tiempo_real)
    mae = abs(error)

    n_vueltas = pd.to_numeric(fila.get("n_laps", np.nan), errors="coerce")
    if np.isfinite(n_vueltas) and int(n_vueltas) > 0:
        error_por_vuelta = error / n_vueltas
    else:
        error_por_vuelta = np.nan

    error_porcentual_total = (error / tiempo_real) * 100.0
    error_absoluto_porcentual = (mae / tiempo_real) * 100.0

    return {
        "tiempo_simulado": float(tiempo_sim),
        "tiempo_real": float(tiempo_real),
        "error": error,
        "mae": mae,
        "error_porcentual_total": error_porcentual_total,
        "error_absoluto_porcentual": error_absoluto_porcentual,
        "error_por_vuelta": float(error_por_vuelta),
    }

def evaluar_simulador_en_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evalúa el simulador sobre todas las filas del dataset utilizando la estrategia real observada.
    Devuelve un DataFrame con el detalle por observación.
    """
    resultados: list[dict] = []

    for idx, fila in df.iterrows():
        estrategia = _parsear_estrategia(fila.get("strategy_compounds", None))

        if estrategia is None:
            resultados.append({
                "index": idx,
                "season": fila.get("season", np.nan),
                "race_id": fila.get("race_id", np.nan),
                "circuit_key": fila.get("circuit_key", np.nan),
                "n_laps": pd.to_numeric(fila.get("n_laps", np.nan), errors="coerce"),
                "wear_index": fila.get("wear_index", np.nan),
                "track_temp_cat": fila.get("track_temp_cat", np.nan),
                "weather_condition": fila.get("weather_condition", np.nan),
                "n_stints": pd.to_numeric(fila.get("n_stints", np.nan), errors="coerce"),
                "action_id": pd.to_numeric(fila.get("action_id", np.nan), errors="coerce"),
                "strategy_compounds": fila.get("strategy_compounds", np.nan),
                "estrategia": None,
                "valida": False,
                "motivo": "strategy_compounds no válida",
                "tiempo_simulado": np.nan,
                "tiempo_real": pd.to_numeric(fila.get("finish_time_s", np.nan), errors="coerce"),
                "error": np.nan,
                "mae": np.nan,
                "error_porcentual_total": np.nan,
                "error_absoluto_porcentual": np.nan,
                "error_por_vuelta": np.nan,
            })
            continue

        evaluacion = evaluar_simulador_en_fila(fila, estrategia)

        resultados.append({
            "index": idx,
            "season": fila.get("season", np.nan),
            "race_id": fila.get("race_id", np.nan),
            "circuit_key": fila.get("circuit_key", np.nan),
            "n_laps": pd.to_numeric(fila.get("n_laps", np.nan), errors="coerce"),
            "wear_index": fila.get("wear_index", np.nan),
            "track_temp_cat": fila.get("track_temp_cat", np.nan),
            "weather_condition": fila.get("weather_condition", np.nan),
            "n_stints": pd.to_numeric(fila.get("n_stints", np.nan), errors="coerce"),
            "action_id": pd.to_numeric(fila.get("action_id", np.nan), errors="coerce"),
            "strategy_compounds": fila.get("strategy_compounds", np.nan),
            "estrategia": estrategia,
            "valida": True,
            "motivo": "ok",
            "tiempo_simulado": evaluacion["tiempo_simulado"],
            "tiempo_real": evaluacion["tiempo_real"],
            "error": evaluacion["error"],
            "mae": evaluacion["mae"],
            "error_porcentual_total": evaluacion["error_porcentual_total"],
            "error_absoluto_porcentual": evaluacion["error_absoluto_porcentual"],
            "error_por_vuelta": evaluacion["error_por_vuelta"],
        })

    return pd.DataFrame(resultados)


def resumir_evaluacion_simulador(df_resultados: pd.DataFrame) -> dict:
    """
    Calcula métricas agregadas de la evaluación global del simulador.
    """
    if df_resultados.empty:
        return {
            "n_total": 0,
            "n_validas": 0,
            "n_invalidas": 0,
            "mae_medio": np.nan,
            "mediana_mae": np.nan,
            "rmse_global": np.nan,
            "error_medio": np.nan,
            "error_absoluto_porcentual_medio": np.nan,
            "error_porcentual_medio": np.nan,
            "error_por_vuelta_medio": np.nan,
            "correlacion_tiempo_real_simulado": np.nan,
        }

    df_validas = df_resultados[
        df_resultados["valida"]
        & df_resultados["tiempo_real"].notna()
        & df_resultados["tiempo_simulado"].notna()
    ].copy()

    if df_validas.empty:
        return {
            "n_total": int(len(df_resultados)),
            "n_validas": 0,
            "n_invalidas": int(len(df_resultados)),
            "mae_medio": np.nan,
            "mediana_mae": np.nan,
            "rmse_global": np.nan,
            "error_medio": np.nan,
            "error_absoluto_porcentual_medio": np.nan,
            "error_porcentual_medio": np.nan,
            "error_por_vuelta_medio": np.nan,
            "correlacion_tiempo_real_simulado": np.nan,
        }

    errores = pd.to_numeric(df_validas["error"], errors="coerce")
    maes = pd.to_numeric(df_validas["mae"], errors="coerce")
    tiempos_reales = pd.to_numeric(df_validas["tiempo_real"], errors="coerce")
    tiempos_simulados = pd.to_numeric(df_validas["tiempo_simulado"], errors="coerce")
    errores_pct = pd.to_numeric(df_validas["error_porcentual_total"], errors="coerce")
    errores_abs_pct = pd.to_numeric(df_validas["error_absoluto_porcentual"], errors="coerce")
    errores_por_vuelta = pd.to_numeric(df_validas["error_por_vuelta"], errors="coerce")

    rmse_global = np.sqrt(np.nanmean(np.square(errores)))

    if len(df_validas) >= 2:
        correlacion = tiempos_reales.corr(tiempos_simulados)
    else:
        correlacion = np.nan

    return {
        "n_total": int(len(df_resultados)),
        "n_validas": int(len(df_validas)),
        "n_invalidas": int(len(df_resultados) - len(df_validas)),
        "mae_medio": float(np.nanmean(maes)),
        "mediana_mae": float(np.nanmedian(maes)),
        "rmse_global": float(rmse_global),
        "error_medio": float(np.nanmean(errores)),
        "error_absoluto_porcentual_medio": float(np.nanmean(errores_abs_pct)),
        "error_porcentual_medio": float(np.nanmean(errores_pct)),
        "error_por_vuelta_medio": float(np.nanmean(errores_por_vuelta)),
        "correlacion_tiempo_real_simulado": float(correlacion) if pd.notna(correlacion) else np.nan,
    }


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
            "n",
            "mae_medio",
            "mediana_mae",
            "rmse_global",
            "error_medio",
            "error_absoluto_porcentual_medio",
            "error_por_vuelta_medio",
        ])

    filas_resumen: list[dict] = []

    for valor, grupo in df_validas.groupby(columna, dropna=False):
        errores = pd.to_numeric(grupo["error"], errors="coerce")
        maes = pd.to_numeric(grupo["mae"], errors="coerce")
        errores_abs_pct = pd.to_numeric(grupo["error_absoluto_porcentual"], errors="coerce")
        errores_por_vuelta = pd.to_numeric(grupo["error_por_vuelta"], errors="coerce")

        rmse_global = np.sqrt(np.nanmean(np.square(errores)))

        filas_resumen.append({
            columna: valor,
            "n": int(len(grupo)),
            "mae_medio": float(np.nanmean(maes)),
            "mediana_mae": float(np.nanmedian(maes)),
            "rmse_global": float(rmse_global),
            "error_medio": float(np.nanmean(errores)),
            "error_absoluto_porcentual_medio": float(np.nanmean(errores_abs_pct)),
            "error_por_vuelta_medio": float(np.nanmean(errores_por_vuelta)),
        })

    return pd.DataFrame(filas_resumen).sort_values(by="mae_medio", ascending=True).reset_index(drop=True)


# Output----------------------------------------------------------------------------------------------------------------
def imprimir_resumen_simulador(fila: pd.Series, estrategia: list[str], evaluacion: dict,
        df_circuitos: pd.DataFrame | None = None) -> None:
    """
    Imprime de manera legible una evaluación del simulador.
    """

    temporada = _obtener_valor(fila, "season")
    id_carrera = _obtener_valor(fila, "race_id")
    circuito = obtener_nombre_circuito(fila, df_circuitos)

    n_vueltas = _obtener_valor(fila, "n_laps")
    pit_loss = pd.to_numeric(fila.get("pit_loss_s", np.nan), errors="coerce")

    if np.isfinite(pit_loss):
        pit_loss = float(pit_loss)
    else:
        pit_loss = np.nan

    desgaste = _obtener_valor(fila, "wear_index")
    temp_cat = _obtener_valor(fila, "track_temp_cat")
    meteo = _obtener_valor(fila, "weather_condition")
    prob_sc = _obtener_valor(fila, "sc_prob")

    print("\n================ EVALUACIÓN DEL SIMULADOR ================")
    print(f"Temporada {temporada} | Carrera {id_carrera} | {circuito}")
    print("----------------------------------------------------------")
    print(f"{n_vueltas} vueltas | {_formatear_valor(pit_loss)} s de pit loss | {len(estrategia) - 1} paradas")
    print(f"Temperatura {temp_cat} | Desgaste {desgaste} | Carrera en {meteo} | {_formatear_valor(prob_sc)} probabilidad de SC")
    print("----------------------------------------------------------")
    print(f"Estrategia a aproximar: {estrategia}")
    print("----------------------------------------------------------")
    print(f"Tiempo real: {_formatear_valor(evaluacion.get('tiempo_real'))} s")
    print(f"Tiempo simulado: {_formatear_valor(evaluacion.get('tiempo_simulado'))} s")
    print("----------------------------------------------------------")
    print(f"Error total: {_formatear_valor(evaluacion.get('error'))} s")
    print(f"Error absoluto: {_formatear_valor(evaluacion.get('mae'))} s")
    print(f"Error por vuelta: {_formatear_valor(evaluacion.get('error_por_vuelta'), 3)} s/vuelta")
    print(f"Error relativo total: {_formatear_valor(evaluacion.get('error_porcentual_total'))} %")
    print("==========================================================\n")

def imprimir_resumen_global_simulador(resumen: dict) -> None:
    """
    Imprime de manera legible el resumen agregado de la evaluación del simulador.
    """
    print("\n================ RESUMEN GLOBAL DEL SIMULADOR ================")
    print(f"Total de observaciones: {resumen.get('n_total', 0)}")
    print(f"Observaciones válidas: {resumen.get('n_validas', 0)}")
    print(f"Observaciones inválidas: {resumen.get('n_invalidas', 0)}")
    print("--------------------------------------------------------------")
    print(f"MAE medio: {_formatear_valor(resumen.get('mae_medio'))} s")
    print(f"Mediana MAE: {_formatear_valor(resumen.get('mediana_mae'))} s")
    print(f"RMSE global: {_formatear_valor(resumen.get('rmse_global'))} s")
    print(f"Error medio: {_formatear_valor(resumen.get('error_medio'))} s")
    print(f"Error relativo absoluto medio: {_formatear_valor(resumen.get('error_absoluto_porcentual_medio'))} %")
    print(f"Error relativo medio: {_formatear_valor(resumen.get('error_porcentual_medio'))} %")
    print(f"Error por vuelta medio: {_formatear_valor(resumen.get('error_por_vuelta_medio'), 3)} s/vuelta")
    print(f"Correlación tiempo real/simulado: {_formatear_valor(resumen.get('correlacion_tiempo_real_simulado'), 4)}")
    print("==============================================================\n")