"""
evaluacion_sim.py

Validación empírica del simulador frente a tiempos reales observados.

Este módulo contiene la lógica necesaria para:
- Validar el simulador comparando tiempos simulados y tiempos reales
  usando la estrategia histórica observada en cada carrera.
- Calcular métricas de error absolutas, relativas y por vuelta
  tanto a nivel individual como agregado.
- Analizar la consistencia del simulador dentro de cada Gran Premio
  mediante métricas relativas respecto al contexto de carrera.
- Generar resúmenes globales y agrupados por circuito, temporada
  u otras variables experimentales.
- Exportar e imprimir los resultados de validación para su análisis
  experimental y su inclusión en la memoria del TFG.

Estas funciones permiten cuantificar el grado de realismo del
entorno de simulación antes de utilizarlo para evaluar políticas.
"""

# IMPORTS
from __future__ import annotations
import ast
import numpy as np
import pandas as pd
from estrategia_f1.acciones import normalizar_estrategia
from estrategia_f1.config import COMPUESTOS
from estrategia_f1.sim.simulador import simular_tiempo_carrera

# HELPERS ESPECÍFICOS --------------------------------------------------------------------------------------------------
def _obtener_valor(fila: pd.Series, key: str, default="N/A"):
    """
    Recupera un valor de una observación de forma segura.
    Si la clave no existe, el valor es None o contiene
    un NaN numérico, devuelve un valor por defecto.

    Parámetros
    ----------
    fila : pd.Series
        Observación de entrada.
    key : str
        Nombre de la columna a recuperar.
    default : str, optional
        Valor utilizado cuando el dato no está disponible.

    Returns
    -------
    Any
        Valor recuperado o valor por defecto.
    """
    valor = fila.get(key, default)
    if valor is None:
        return default
    if isinstance(valor, float) and np.isnan(valor):
        return default
    return valor

def _formatear_valor(valor, decimales: int = 2) -> str:
    """
    Convierte un valor numérico a una representación legible.
    Si el valor no existe o contiene un NaN, devuelve una
    representación textual por defecto. Si el valor no puede
    convertirse a float, se devuelve su representación textual.

    Parámetros
    ----------
    valor : Any
        Valor que se desea formatear.
    decimales : int, optional
        Número de decimales mostrados para valores numéricos.

    Returns
    -------
    str
        Representación formateada del valor.
    """
    try:
        if valor is None or (isinstance(valor, float) and np.isnan(valor)):
            return "N/A"
        return f"{float(valor):.{decimales}f}"
    except Exception:
        return str(valor)

def _id_gp(fila: pd.Series) -> tuple:
    """
    Construye un identificador único de Gran Premio.

    La combinación de temporada y race_id permite
    distinguir carreras de forma consistente entre
    distintas temporadas del dataset.

    Parámetros
    ----------
    fila : pd.Series
        Observación piloto-carrera.

    Returns
    -------
    tuple
        Identificador del Gran Premio con el formato:
        (season, race_id)
    """
    return fila.get("season", np.nan), fila.get("race_id", np.nan)

def obtener_nombre_circuito(fila: pd.Series, df_circuitos: pd.DataFrame | None) -> str:
    """
    Recupera el nombre corto del circuito asociado a una carrera.

    Parámetros
    ----------
    fila : pd.Series
        Observación piloto-carrera.
    df_circuitos : pd.DataFrame | None
        Tabla de referencia con la correspondencia entre
        circuit_key y circuit_short_name.

    Returns
    -------
    str
        Nombre corto del circuito si está disponible.
        En caso contrario, devuelve "N/A" o el propio
        identificador del circuito.
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

# EVALUACIÓN INDIVIDUAL ------------------------------------------------------------------------------------------------
def evaluar_simulador_en_fila(fila: pd.Series, estrategia: list[str]) -> dict:
    """
    Evalúa el simulador sobre una observación piloto-carrera.

    La función ejecuta el simulador utilizando la estrategia
    real observada en carrera y compara el tiempo simulado
    frente al tiempo real registrado.

    Parámetros
    ----------
    fila : pd.Series
        Observación piloto-carrera con las variables necesarias
        para la simulación y el tiempo real observado.
    estrategia : list[str]
        Estrategia de neumáticos realmente utilizada en carrera.

    Returns
    -------
    dict
        Diccionario con métricas de validación:
        - tiempo simulado.
        - tiempo real.
        - error firmado.
        - error absoluto.
        - error porcentual.
        - error por vuelta.

        Si la simulación no es válida, las métricas de error
        se devuelven como np.nan.
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

# EVALUACIÓN DEL DATASET -----------------------------------------------------------------------------------------------
def evaluar_simulador_en_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
        Evalúa el simulador sobre el dataset experimental.

        Para cada observación piloto-carrera, se recupera la estrategia
        real observada, se simula su tiempo de carrera y se compara con
        el tiempo real registrado. Además, se calculan métricas relativas
        dentro de cada Gran Premio para analizar la coherencia del simulador
        respecto al contexto de carrera.

        Parámetros
        ----------
        df : pd.DataFrame
            Dataset experimental con observaciones piloto-carrera,
            estrategias reales y tiempos finales observados.

        Returns
        -------
        pd.DataFrame
            DataFrame con el detalle de validación por observación,
            incluyendo tiempos reales y simulados, errores absolutos,
            errores relativos y motivo de validez.
        """
    resultados: list[dict] = []

    for idx, fila in df.iterrows():
        estrategia = normalizar_estrategia(fila.get("strategy_compounds", None))

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
            # Las estrategias no parseables o incompatibles se registran como inválidas, pero no se eliminan
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

        # Evaluación individual de la estrategia real observada
        evaluacion = evaluar_simulador_en_fila(fila, estrategia)

        # Una observación solo se considera válida si dispone de tiempo real, tiempo simulado y error numéricamente evaluable
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

    # Métricas relativas dentro de cada GP
    # Esto ayuda a validar no solo tiempos absolutos, sino también desviaciones respecto al contexto de carrera
    df_validas = df_resultados[df_resultados["valida"]].copy()

    if not df_validas.empty:
        columnas_gp = ["season", "race_id"]

        # Medianas de referencia dentro de cada carrera
        mediana_real_gp = df_validas.groupby(columnas_gp)["tiempo_real"].transform("median")
        mediana_sim_gp = df_validas.groupby(columnas_gp)["tiempo_simulado"].transform("median")

        # Desviación real y simulada respecto al contexto de carrera
        df_resultados.loc[df_validas.index, "tiempo_real_vs_mediana_gp"] = (
            df_validas["tiempo_real"] - mediana_real_gp
        )
        df_resultados.loc[df_validas.index, "tiempo_simulado_vs_mediana_gp"] = (
            df_validas["tiempo_simulado"] - mediana_sim_gp
        )

        # Error relativo: diferencia entre la desviación simulada y la desviación real dentro del mismo Gran Premio
        df_resultados.loc[df_validas.index, "error_relativo_gp"] = (
            df_resultados.loc[df_validas.index, "tiempo_simulado_vs_mediana_gp"]
            - df_resultados.loc[df_validas.index, "tiempo_real_vs_mediana_gp"]
        )
        df_resultados.loc[df_validas.index, "error_absoluto_relativo_gp"] = (
            df_resultados.loc[df_validas.index, "error_relativo_gp"].abs()
        )

    return df_resultados

# RESÚMENES ------------------------------------------------------------------------------------------------------------
def resumir_evaluacion_simulador(df_resultados: pd.DataFrame) -> dict:
    """
    Calcula métricas agregadas de validación del simulador.

    A partir de las evaluaciones individuales piloto-carrera,
    la función calcula métricas globales de error, correlación
    y consistencia relativa dentro de cada Gran Premio.

    Parámetros
    ----------
    df_resultados : pd.DataFrame
        Resultados detallados de validación obtenidos sobre
        el dataset experimental.

    Returns
    -------
    dict
        Diccionario con métricas agregadas de validación,
        incluyendo número de observaciones válidas, errores
        absolutos, correlaciones y métricas relativas por GP.
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

    # Puede ocurrir que existan filas, pero ninguna sea numéricamente evaluable tras la simulación
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

    # Las correlaciones solo son interpretables con al menos dos observaciones válidas
    if len(df_validas) >= 2:
        pearson = tiempos_reales.corr(tiempos_simulados, method="pearson")
        spearman = tiempos_reales.corr(tiempos_simulados, method="spearman")
    else:
        pearson = np.nan
        spearman = np.nan

    # Validación relativa dentro de cada GP
    # Permite analizar si el simulador conserva la posición relativa de cada piloto en carrera
    if {
        "tiempo_real_vs_mediana_gp",
        "tiempo_simulado_vs_mediana_gp",
        "error_absoluto_relativo_gp",
    }.issubset(df_validas.columns):
        real_rel = pd.to_numeric(df_validas["tiempo_real_vs_mediana_gp"], errors="coerce")
        sim_rel = pd.to_numeric(df_validas["tiempo_simulado_vs_mediana_gp"], errors="coerce")
        error_abs_rel_gp = pd.to_numeric(df_validas["error_absoluto_relativo_gp"], errors="coerce")

        # Correlación entre desviaciones reales y simuladas respecto a la mediana del mismo Gran Premio
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
    Resume la validación del simulador por Gran Premio.

    Agrupa las observaciones válidas por temporada y carrera,
    calculando métricas agregadas de error y correlación entre
    tiempos reales y simulados para cada Gran Premio.

    Parámetros
    ----------
    df_resultados : pd.DataFrame
        Resultados detallados de la evaluación del simulador.

    Returns
    -------
    pd.DataFrame
        DataFrame con un resumen por carrera, ordenado de menor
        a mayor error absoluto medio.
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

    # Cada grupo representa un Gran Premio concreto
    for (season, race_id), grupo in df_validas.groupby(["season", "race_id"], dropna=False):
        errores = pd.to_numeric(grupo["error"], errors="coerce")
        errores_abs = pd.to_numeric(grupo["error_absoluto"], errors="coerce")
        errores_abs_pct = pd.to_numeric(grupo["error_absoluto_porcentual"], errors="coerce")
        tiempos_reales = pd.to_numeric(grupo["tiempo_real"], errors="coerce")
        tiempos_simulados = pd.to_numeric(grupo["tiempo_simulado"], errors="coerce")

        if len(grupo) >= 2 and tiempos_reales.nunique() > 1 and tiempos_simulados.nunique() > 1:
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

    # Se ordenan las carreras desde las mejor simuladas hasta las que presentan mayor error medio
    return (
        pd.DataFrame(filas)
        .sort_values(by="mae_medio", ascending=True)
        .reset_index(drop=True)
    )

def resumir_evaluacion_por_grupo(df_resultados: pd.DataFrame, columna: str) -> pd.DataFrame:
    """
    Resume la validación del simulador agrupando por una variable.

    Permite analizar el comportamiento del simulador en distintos
    subconjuntos del dataset, por ejemplo por circuito, temporada,
    desgaste, temperatura de pista o número de stints.

    Parámetros
    ----------
    df_resultados : pd.DataFrame
        Resultados detallados de la evaluación del simulador.

    columna : str
        Nombre de la columna por la que se desea agrupar.

    Returns
    -------
    pd.DataFrame
        DataFrame con métricas agregadas para cada valor de la
        columna indicada, ordenado de menor a mayor error absoluto
        medio.
    """
    # La columna de agrupación debe existir para evitar análisis ambiguos
    if columna not in df_resultados.columns:
        raise KeyError(f"La columna '{columna}' no existe en df_resultados.")

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

    # Cada grupo permite estudiar el error del simulador bajo una condición concreta
    for valor, grupo in df_validas.groupby(columna, dropna=False):
        errores = pd.to_numeric(grupo["error"], errors="coerce")
        errores_abs = pd.to_numeric(grupo["error_absoluto"], errors="coerce")
        errores_abs_pct = pd.to_numeric(grupo["error_absoluto_porcentual"], errors="coerce")
        errores_por_vuelta = pd.to_numeric(grupo["error_por_vuelta"], errors="coerce")
        tiempos_reales = pd.to_numeric(grupo["tiempo_real"], errors="coerce")
        tiempos_simulados = pd.to_numeric(grupo["tiempo_simulado"], errors="coerce")

        if len(grupo) >= 2 and tiempos_reales.nunique() > 1 and tiempos_simulados.nunique() > 1:
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

    # Se ordena por MAE para identificar los grupos mejor y peor simulados
    return (
        pd.DataFrame(filas_resumen)
        .sort_values(by="mae_medio", ascending=True)
        .reset_index(drop=True)
    )

# EXPORTACIÓN DE RESULTADOS --------------------------------------------------------------------------------------------
def guardar_evaluacion_simulador(df_resultados: pd.DataFrame, ruta_detalle: str, ruta_resumen_gp: str | None = None) -> None:
    """
        Exporta los resultados de validación del simulador.

        Parámetros
        ----------
        df_resultados : pd.DataFrame
            Resultados detallados de la evaluación del simulador.
        ruta_detalle : str
            Ruta del archivo CSV donde guardar el detalle completo.
        ruta_resumen_gp : str | None, optional
            Ruta del archivo CSV donde guardar el resumen por
            Gran Premio. Si es None, no se exporta.

        Returns
        -------
        None
        """
    df_resultados.to_csv(ruta_detalle, index=False)

    if ruta_resumen_gp is not None:
        resumen_gp = resumir_evaluacion_por_gp(df_resultados)
        resumen_gp.to_csv(ruta_resumen_gp, index=False)


# OUTPUT POR CONSOLA ---------------------------------------------------------------------------------------------------
def imprimir_resumen_simulador(fila: pd.Series, estrategia: list[str], evaluacion: dict, df_circuitos: pd.DataFrame | None = None) -> None:
    """
    Imprime un resumen legible de la validación del simulador.

    Parámetros
    ----------
    fila : pd.Series
        Observación piloto-carrera evaluada.
    estrategia : list[str]
        Estrategia de neumáticos utilizada en la simulación.
    evaluacion : dict
        Métricas de validación obtenidas para la observación.
    df_circuitos : pd.DataFrame | None, optional
        Tabla de referencia con nombres de circuitos.

    Returns
    -------
    None
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
    Imprime un resumen global de la validación del simulador.

    Parámetros
    ----------
    resumen : dict
        Métricas agregadas obtenidas mediante la evaluación
        global del simulador.

    Returns
    -------
    None
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