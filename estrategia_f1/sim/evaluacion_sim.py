"""
evaluacion_sim.py
TODO
"""

from __future__ import annotations

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

    return {
        "tiempo_simulado": float(tiempo_sim),
        "tiempo_real": float(tiempo_real),
        "error": error,
        "mae": mae,
        "error_porcentual_total": error_porcentual_total,
        "error_por_vuelta": float(error_por_vuelta),
    }


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