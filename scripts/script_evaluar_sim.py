"""
script_evaluar_sim.py

Validación empírica del simulador de estrategias de carrera.
"""

# IMPORTS
from __future__ import annotations
import pandas as pd
from estrategia_f1.config import (
    DATASET_EXPERIMENTAL_CSV,
    SEED,
    CIRCUITOS_CSV,
    DATOS_MEMORIA,
)
from estrategia_f1.sim.evaluacion_sim import (
    evaluar_simulador_en_fila,
    evaluar_simulador_en_dataset,
    resumir_evaluacion_simulador,
    resumir_evaluacion_por_grupo,
    resumir_evaluacion_por_gp,
    imprimir_resumen_simulador,
    imprimir_resumen_global_simulador,
    guardar_evaluacion_simulador,
)

def main() -> None:
    """
    Ejecuta el flujo completo de validación empírica del simulador.

    El proceso incluye:
    1. Carga del dataset experimental y de la información de circuitos.
    2. Evaluación del simulador sobre estrategias reales observadas.
    3. Cálculo de métricas globales de error y correlación.
    4. Generación de resúmenes por carrera y por grupos de análisis.
    5. Exportación de resultados para memoria y anexos.
    6. Impresión de ejemplos individuales de simulación.
    """

    df = pd.read_csv(DATASET_EXPERIMENTAL_CSV)
    df_circuitos = pd.read_csv(CIRCUITOS_CSV)

    print("\n================ VALIDACIÓN EMPÍRICA DEL SIMULADOR ================")
    print(f"Dataset cargado: {len(df)} observaciones")
    print(f"Carreras/GP únicos: {df[['season', 'race_id']].drop_duplicates().shape[0]}")

    print("\nColumnas principales disponibles:")
    columnas_interes = [
        "season",
        "race_id",
        "circuit_key",
        "n_laps",
        "strategy_compounds",
        "finish_time_s",
        "track_temp_cat",
        "wear_index",
        "n_stints",
    ]
    print([col for col in columnas_interes if col in df.columns])

    # Evaluación global del simulador sobre todas las observaciones
    df_resultados = evaluar_simulador_en_dataset(df)
    resumen_global = resumir_evaluacion_simulador(df_resultados)

    imprimir_resumen_global_simulador(resumen_global)

    # Exportación de resultados para análisis externo y anexos
    guardar_evaluacion_simulador(
        df_resultados,
        DATOS_MEMORIA / "evaluacion_simulador_detalle.csv",
        DATOS_MEMORIA / "evaluacion_simulador_por_gp.csv",
    )

    print("Archivos generados:")
    print("- evaluacion_simulador_detalle.csv")
    print("- evaluacion_simulador_por_gp.csv")

    # Ejemplo individual de simulación para inspección manual
    df_validas = df_resultados[df_resultados["valida"]].copy()

    if not df_validas.empty:
        fila_resultado = df_validas.sample(1, random_state=SEED).iloc[0]
        idx_original = fila_resultado["index"]

        fila_original = df.loc[idx_original]
        estrategia = fila_resultado["estrategia"]

        evaluacion = evaluar_simulador_en_fila(fila_original, estrategia)
        imprimir_resumen_simulador(fila_original, estrategia, evaluacion, df_circuitos)
    else:
        print("\nNo hay observaciones válidas para imprimir un ejemplo individual.")

    # Resumen por Gran Premio
    print("\nResumen por GP:")
    df_por_gp = resumir_evaluacion_por_gp(df_resultados)
    print(df_por_gp.to_string(index=False))
    print()

    # Resúmenes agrupados por variables relevantes del contexto de carrera
    grupos = [
        ("temperatura de pista", "track_temp_cat"),
        ("desgaste", "wear_index"),
        ("número de stints", "n_stints"),
        ("circuito", "circuit_key"),
    ]

    for nombre, columna in grupos:
        if columna not in df_resultados.columns:
            print(f"\nNo se puede calcular resumen por {nombre}: falta columna {columna}.")
            continue

        print(f"\nResumen por {nombre}:")
        df_grupo = resumir_evaluacion_por_grupo(df_resultados, columna)

        if columna == "circuit_key" and not df_grupo.empty:
            if {"circuit_key", "circuit_short_name"}.issubset(df_circuitos.columns):
                df_grupo = df_grupo.merge(
                    df_circuitos[["circuit_key", "circuit_short_name"]].drop_duplicates(),
                    on="circuit_key",
                    how="left",
                )

                columnas = ["circuit_short_name"] + [
                    col for col in df_grupo.columns if col != "circuit_short_name"
                ]
                df_grupo = df_grupo[columnas]

        print(df_grupo.to_string(index=False))
        print()

if __name__ == "__main__":
    main()