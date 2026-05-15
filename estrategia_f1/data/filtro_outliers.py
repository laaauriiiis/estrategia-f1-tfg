"""
filtro_outliers.py

Filtrado y análisis de observaciones atípicas del dataset experimental (outliers).

Este módulo contiene la lógica necesaria para:
- Aplicar filtros de calidad sobre tiempos finales, tiempos por vuelta y estados de carrera.
- Eliminar observaciones atípicas respecto al rendimiento mediano de cada Gran Premio.
- Generar estadísticas de trazabilidad sobre las observaciones eliminadas y conservadas.

Estas funciones permiten preparar versiones filtradas del dataset
experimental antes del entrenamiento y la evaluación de los modelos.
"""

# IMPORTS
import pandas as pd

def filtrar_dataset(df: pd.DataFrame, *, tipo_pipeline: str = "both") -> tuple[pd.DataFrame, dict]:
    """
    Filtra el dataset experimental aplicando criterios
    de calidad y eliminación de outliers.

    Parámetros
    ----------
    df : pd.DataFrame
        Dataset experimental que se desea filtrar.
    tipo_pipeline : str, optional
        Tipo de pipeline experimental para el que se
        prepara el dataset:
        - "ml" : aplica filtros adaptados al enfoque supervisado.
        - "rl" : aplica filtros más restrictivos adaptados al enfoque basado en valor.
        - "both" : aplica el filtrado completo por defecto.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Tupla formada por:
        - DataFrame filtrado.
        - Diccionario con estadísticas del proceso
          de filtrado, incluyendo observaciones
          eliminadas y porcentaje retenido.
    """
    # Se conserva una copia del dataset original para calcular estadísticas de trazabilidad del filtrado
    df_original = df.copy()
    n_original = len(df_original)

    stats = {
        "n_original": n_original,
        "filtros_aplicados": {},
        "n_final": 0,
        "porcentaje_retenido": 0.0,
        "tipo_pipeline": tipo_pipeline
    }

    print(f"Dataset original: {n_original:,} filas (pipeline: {tipo_pipeline})")

    # 1) Eliminación de observaciones con tiempos finales incompatibles con una carrera completa
    df = df[df["finish_time_s"] > 2000]
    n_tiempo = len(df)
    stats["filtros_aplicados"]["finish_time_s_>_2000"] = {
        "eliminadas": n_original - n_tiempo,
        "restantes": n_tiempo
    }
    print(f"Después de finish_time_s > 2000: {n_tiempo:,} filas (-{n_original - n_tiempo:,})")

    # 2) Validación del ritmo medio por vuelta para descartar registros físicamente no plausibles
    if "s_per_lap" in df.columns:
        df = df[(df["s_per_lap"] > 50) & (df["s_per_lap"] < 250)]
        n_lap = len(df)
        stats["filtros_aplicados"]["s_per_lap_50_250"] = {
            "eliminadas": n_tiempo - n_lap,
            "restantes": n_lap
        }
        print(f"Después de 50 < s_per_lap < 250: {n_lap:,} filas (-{n_tiempo - n_lap:,})")
    else:
        n_lap = len(df)
        print("Columna 's_per_lap' no encontrada, saltando filtro")

    # 3) Eliminación de abandonos, no salidas y descalificaciones cuando la información está disponible
    condiciones_validas = []

    if "dnf" in df.columns:
        condiciones_validas.append(df["dnf"].isna() | (df["dnf"] == 0) | (df["dnf"] == False))

    if "dns" in df.columns:
        condiciones_validas.append(df["dns"].isna() | (df["dns"] == 0) | (df["dns"] == False))

    if "dsq" in df.columns:
        condiciones_validas.append(df["dsq"].isna() | (df["dsq"] == 0) | (df["dsq"] == False))

    if condiciones_validas:
        mascara_valida = condiciones_validas[0]
        for condicion in condiciones_validas[1:]:
            mascara_valida = mascara_valida & condicion

        df = df[mascara_valida]
        n_status = len(df)
        stats["filtros_aplicados"]["no_dnf_dns_dsq"] = {
            "eliminadas": n_lap - n_status,
            "restantes": n_status
        }
        print(f"Después de eliminar DNF/DNS/DSQ: {n_status:,} filas (-{n_lap - n_status:,})")
    else:
        n_status = len(df)
        print("Columnas DNF/DNS/DSQ no encontradas, saltando filtro")

    # 4) Cálculo del rendimiento relativo dentro de cada GP, utilizando la mediana de la carrera como referencia
    df["delta_vs_race"] = df["finish_time_s"] - df.groupby("race_id")["finish_time_s"].transform("median")

    limite_superior = 120
    limite_inferior = -60

    df = df[(df["delta_vs_race"] >= limite_inferior) & (df["delta_vs_race"] <= limite_superior)]
    n_delta = len(df)
    stats["filtros_aplicados"]["delta_vs_race_simetrico"] = {
        "eliminadas": n_status - n_delta,
        "restantes": n_delta,
        "limite_inferior": limite_inferior,
        "limite_superior": limite_superior
    }
    print(
        f"Después de filtro simétrico delta_vs_race [{limite_inferior}, {limite_superior}]s: {n_delta:,} filas (-{n_status - n_delta:,})")

    # 6) Filtrado específico según el pipeline experimental:
    # RL utiliza IQR por mayor sensibilidad a rewards extremos,
    # mientras que ML emplea percentiles más tolerantes
    if len(df) == 0:
        n_final = 0
        print("ADVERTENCIA: Dataset vacío después del filtrado básico")
    elif tipo_pipeline in ["rl", "both"]:
        Q1 = df["delta_vs_race"].quantile(0.25)
        Q3 = df["delta_vs_race"].quantile(0.75)
        IQR = Q3 - Q1
        limite_inf_iqr = Q1 - 1.5 * IQR
        limite_sup_iqr = Q3 + 1.5 * IQR

        df = df[(df["delta_vs_race"] >= limite_inf_iqr) & (df["delta_vs_race"] <= limite_sup_iqr)]
        n_final = len(df)

        stats["filtros_aplicados"]["outliers_iqr_rl"] = {
            "eliminadas": n_delta - n_final,
            "restantes": n_final,
            "Q1": Q1,
            "Q3": Q3,
            "IQR": IQR,
            "limite_inferior": limite_inf_iqr,
            "limite_superior": limite_sup_iqr
        }
        print(f"Después de filtro IQR (RL): {n_final:,} filas (-{n_delta - n_final:,})")
        print(f"  Q1: {Q1:.1f}s, Q3: {Q3:.1f}s, IQR: {IQR:.1f}s")
        print(f"  Límites IQR: [{limite_inf_iqr:.1f}, {limite_sup_iqr:.1f}]s")

    elif tipo_pipeline == "ml":
        Q05 = df["delta_vs_race"].quantile(0.05)
        Q95 = df["delta_vs_race"].quantile(0.95)
        df = df[(df["delta_vs_race"] >= Q05) & (df["delta_vs_race"] <= Q95)]
        n_final = len(df)

        stats["filtros_aplicados"]["percentiles_ml"] = {
            "eliminadas": n_delta - n_final,
            "restantes": n_final,
            "Q05": Q05,
            "Q95": Q95
        }
        print(f"Después de filtro percentiles (ML): {n_final:,} filas (-{n_delta - n_final:,})")
        print(f"  Límites percentiles 5-95: [{Q05:.1f}, {Q95:.1f}]s")

    else:
        n_final = n_delta
        print(f"Tipo de pipeline '{tipo_pipeline}' no reconocido, sin filtrado adicional")

    stats["n_final"] = n_final
    stats["porcentaje_retenido"] = (n_final / n_original * 100) if n_original > 0 else 0

    print(f"\n=== RESUMEN FILTRADO ({tipo_pipeline.upper()}) ===")
    print(f"   Original: {n_original:,} filas")
    print(f"   Final: {n_final:,} filas")
    print(f"   Retenido: {stats['porcentaje_retenido']:.1f}%")
    print(f"   Eliminado: {n_original - n_final:,} filas ({100 - stats['porcentaje_retenido']:.1f}%)")

    # La columna auxiliar delta_vs_race solo se conserva si ya formaba parte del dataset original
    if "delta_vs_race" not in df_original.columns:
        df = df.drop(columns=["delta_vs_race"])

    return df.reset_index(drop=True), stats


def imprimir_distribucion_filtrado(df_antes: pd.DataFrame, df_despues: pd.DataFrame) -> None:
    """
    Muestra estadísticas descriptivas antes y después
    del proceso de filtrado.

    Parámetros
    ----------
    df_antes : pd.DataFrame
        Dataset original antes de aplicar los filtros.
    df_despues : pd.DataFrame
        Dataset resultante tras aplicar el filtrado.

    Returns
    -------
    None
        La función imprime por consola estadísticas
        comparativas para facilitar la validación
        empírica del proceso de depuración.
    """
    print("\n=== DISTRIBUCIONES ANTES/DESPUÉS ===")

    # Comparación del tiempo total de carrera
    if "finish_time_s" in df_antes.columns:
        print(f"finish_time_s:")
        print(
            f"  Antes - Media: {df_antes['finish_time_s'].mean():.0f}s, Mediana: {df_antes['finish_time_s'].median():.0f}s")
        if len(df_despues) > 0:
            print(
                f"  Después - Media: {df_despues['finish_time_s'].mean():.0f}s, Mediana: {df_despues['finish_time_s'].median():.0f}s")
        else:
            print(f"  Después - Dataset vacío")

    # Comparación del ritmo medio por vuelta
    if "s_per_lap" in df_antes.columns and "s_per_lap" in df_despues.columns:
        print(f"s_per_lap:")
        print(f"  Antes - Media: {df_antes['s_per_lap'].mean():.1f}s, Mediana: {df_antes['s_per_lap'].median():.1f}s")
        if len(df_despues) > 0:
            print(
                f"  Después - Media: {df_despues['s_per_lap'].mean():.1f}s, Mediana: {df_despues['s_per_lap'].median():.1f}s")

    # Comprobación del impacto del filtrado sobre la cobertura de Grandes Premios disponibles
    if "race_id" in df_antes.columns:
        print(f"Carreras únicas:")
        print(f"  Antes: {df_antes['race_id'].nunique()}")
        if len(df_despues) > 0 and "race_id" in df_despues.columns:
            print(f"  Después: {df_despues['race_id'].nunique()}")
        else:
            print(f"  Después: 0")