def filtrar_dataset_rl(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Filtra el dataset aplicando criterios específicos para RL.

    Filtros aplicados:
    1. finish_time_s > 2000
    2. 50 < s_per_lap < 250
    3. No DNF/DNS/DSQ
    4. delta_vs_race < 120 segundos
    5. Outliers por IQR en delta_vs_race

    Returns:
        tuple: (df_filtrado, stats_filtrado)
    """
    df_original = df.copy()
    n_original = len(df_original)

    stats = {
        "n_original": n_original,
        "filtros_aplicados": {},
        "n_final": 0,
        "porcentaje_retenido": 0.0
    }

    print(f"Dataset original: {n_original:,} filas")

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
        condiciones_validas.append(df["dnf"] != 1)
        condiciones_validas.append(df["dnf"] != True)
        condiciones_validas.append(df["dnf"].isna() | (df["dnf"] == 0))

    if "dns" in df.columns:
        condiciones_validas.append(df["dns"] != 1)
        condiciones_validas.append(df["dns"] != True)
        condiciones_validas.append(df["dns"].isna() | (df["dns"] == 0))

    if "dsq" in df.columns:
        condiciones_validas.append(df["dsq"] != 1)
        condiciones_validas.append(df["dsq"] != True)
        condiciones_validas.append(df["dsq"].isna() | (df["dsq"] == 0))

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

    # 4. Calcular delta_vs_race y filtrar por < 120 segundos
    df["delta_vs_race"] = df["finish_time_s"] - df.groupby("race_id")["finish_time_s"].transform("median")

    df = df[df["delta_vs_race"] < 120]
    n_delta = len(df)
    stats["filtros_aplicados"]["delta_vs_race_<_120"] = {
        "eliminadas": n_status - n_delta,
        "restantes": n_delta
    }
    print(f"Después de delta_vs_race < 120s: {n_delta:,} filas (-{n_status - n_delta:,})")

    # 5. Filtro por outliers IQR en delta_vs_race
    if len(df) > 0:
        Q1 = df["delta_vs_race"].quantile(0.25)
        Q3 = df["delta_vs_race"].quantile(0.75)
        IQR = Q3 - Q1
        limite_superior = Q3 + 1.5 * IQR

        df = df[df["delta_vs_race"] <= limite_superior]
        n_iqr = len(df)
        stats["filtros_aplicados"]["outliers_iqr"] = {
            "eliminadas": n_delta - n_iqr,
            "restantes": n_iqr,
            "Q1": Q1,
            "Q3": Q3,
            "IQR": IQR,
            "limite_superior": limite_superior
        }
        print(f"Después de eliminar outliers IQR: {n_iqr:,} filas (-{n_delta - n_iqr:,})")
        print(f"  Q1: {Q1:.1f}s, Q3: {Q3:.1f}s, IQR: {IQR:.1f}s, límite: {limite_superior:.1f}s")
    else:
        n_iqr = 0

    # Estadísticas finales
    stats["n_final"] = n_iqr
    stats["porcentaje_retenido"] = (n_iqr / n_original * 100) if n_original > 0 else 0

    print(f"\n📊 RESUMEN FILTRADO:")
    print(f"   Original: {n_original:,} filas")
    print(f"   Final: {n_iqr:,} filas")
    print(f"   Retenido: {stats['porcentaje_retenido']:.1f}%")
    print(f"   Eliminado: {n_original - n_iqr:,} filas ({100 - stats['porcentaje_retenido']:.1f}%)")

    # Remover la columna temporal si no existía antes
    if "delta_vs_race" not in df_original.columns:
        df = df.drop(columns=["delta_vs_race"])

    return df.reset_index(drop=True), stats


def imprimir_distribucion_filtrado(df_antes: pd.DataFrame, df_despues: pd.DataFrame) -> None:
    """
    Imprime estadísticas comparativas antes/después del filtrado.
    """
    print("\n📈 DISTRIBUCIONES ANTES/DESPUÉS:")

    # Finish time
    if "finish_time_s" in df_antes.columns:
        print(f"finish_time_s:")
        print(
            f"  Antes - Media: {df_antes['finish_time_s'].mean():.0f}s, Mediana: {df_antes['finish_time_s'].median():.0f}s")
        print(
            f"  Después - Media: {df_despues['finish_time_s'].mean():.0f}s, Mediana: {df_despues['finish_time_s'].median():.0f}s")

    # S per lap
    if "s_per_lap" in df_antes.columns and "s_per_lap" in df_despues.columns:
        print(f"s_per_lap:")
        print(f"  Antes - Media: {df_antes['s_per_lap'].mean():.1f}s, Mediana: {df_antes['s_per_lap'].median():.1f}s")
        print(
            f"  Después - Media: {df_despues['s_per_lap'].mean():.1f}s, Mediana: {df_despues['s_per_lap'].median():.1f}s")

    # Carreras únicas
    if "race_id" in df_antes.columns:
        print(f"Carreras únicas:")
        print(f"  Antes: {df_antes['race_id'].nunique()}")
        print(f"  Después: {df_despues['race_id'].nunique()}")