import pandas as pd


def filtrar_dataset(df: pd.DataFrame, *, tipo_pipeline: str = "both") -> tuple[pd.DataFrame, dict]:
    """
    Filtra el dataset aplicando criterios específicos para RL/ML.

    Args:
        tipo_pipeline: "ml", "rl", o "both" para aplicar filtros específicos
    """
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

    # 1. Filtro por finish_time_s > 2000
    df = df[df["finish_time_s"] > 2000]
    n_tiempo = len(df)
    stats["filtros_aplicados"]["finish_time_s_>_2000"] = {
        "eliminadas": n_original - n_tiempo,
        "restantes": n_tiempo
    }
    print(f"Después de finish_time_s > 2000: {n_tiempo:,} filas (-{n_original - n_tiempo:,})")

    # 2. Filtro por s_per_lap válido
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

    # 3. Filtro por DNF/DNS/DSQ
    condiciones_validas = []

    if "dnf" in df.columns:
        condiciones_validas.append(df["dnf"].isna() | (df["dnf"] == 0) | (df["dnf"] == False))

    if "dns" in df.columns:
        condiciones_validas.append(df["dns"].isna() | (df["dns"] == 0) | (df["dns"] == False))

    if "dsq" in df.columns:
        condiciones_validas.append(df["dsq"].isna() | (df["dsq"] == 0) | (df["dsq"] == False))

    if condiciones_validas:
        # Combinar todas las condiciones con AND
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

    # 4. Calcular delta_vs_race para filtrado posterior
    df["delta_vs_race"] = df["finish_time_s"] - df.groupby("race_id")["finish_time_s"].transform("median")

    # Filtro simétrico básico: eliminar outliers extremos tanto rápidos como lentos
    limite_superior = 120  # segundos
    limite_inferior = -60  # permitir algunos coches más rápidos, pero no outliers extremos

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

    # 5. Filtros específicos según el tipo de pipeline
    if len(df) == 0:
        n_final = 0
        print("ADVERTENCIA: Dataset vacío después del filtrado básico")
    elif tipo_pipeline in ["rl", "both"]:
        # RL necesita filtrado más estricto (IQR)
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
        # ML más tolerante a outliers, solo filtro extremo (percentiles)
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
        # Tipo de pipeline no reconocido, usar dataset sin filtro adicional
        n_final = n_delta
        print(f"Tipo de pipeline '{tipo_pipeline}' no reconocido, sin filtrado adicional")

    # Estadísticas finales
    stats["n_final"] = n_final
    stats["porcentaje_retenido"] = (n_final / n_original * 100) if n_original > 0 else 0

    print(f"\n=== RESUMEN FILTRADO ({tipo_pipeline.upper()}) ===")
    print(f"   Original: {n_original:,} filas")
    print(f"   Final: {n_final:,} filas")
    print(f"   Retenido: {stats['porcentaje_retenido']:.1f}%")
    print(f"   Eliminado: {n_original - n_final:,} filas ({100 - stats['porcentaje_retenido']:.1f}%)")

    # Remover la columna temporal si no existía antes
    if "delta_vs_race" not in df_original.columns:
        df = df.drop(columns=["delta_vs_race"])

    return df.reset_index(drop=True), stats


def imprimir_distribucion_filtrado(df_antes: pd.DataFrame, df_despues: pd.DataFrame) -> None:
    """
    Imprime estadísticas comparativas antes/después del filtrado.
    """
    print("\n=== DISTRIBUCIONES ANTES/DESPUÉS ===")

    # Finish time
    if "finish_time_s" in df_antes.columns:
        print(f"finish_time_s:")
        print(
            f"  Antes - Media: {df_antes['finish_time_s'].mean():.0f}s, Mediana: {df_antes['finish_time_s'].median():.0f}s")
        if len(df_despues) > 0:
            print(
                f"  Después - Media: {df_despues['finish_time_s'].mean():.0f}s, Mediana: {df_despues['finish_time_s'].median():.0f}s")
        else:
            print(f"  Después - Dataset vacío")

    # S per lap
    if "s_per_lap" in df_antes.columns and "s_per_lap" in df_despues.columns:
        print(f"s_per_lap:")
        print(f"  Antes - Media: {df_antes['s_per_lap'].mean():.1f}s, Mediana: {df_antes['s_per_lap'].median():.1f}s")
        if len(df_despues) > 0:
            print(
                f"  Después - Media: {df_despues['s_per_lap'].mean():.1f}s, Mediana: {df_despues['s_per_lap'].median():.1f}s")

    # Carreras únicas
    if "race_id" in df_antes.columns:
        print(f"Carreras únicas:")
        print(f"  Antes: {df_antes['race_id'].nunique()}")
        if len(df_despues) > 0 and "race_id" in df_despues.columns:
            print(f"  Después: {df_despues['race_id'].nunique()}")
        else:
            print(f"  Después: 0")