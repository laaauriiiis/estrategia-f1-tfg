"""
script_evaluar_sim.py
TODO
"""
import ast
import pandas as pd

from estrategia_f1.config import (
    DATASET_SIM_CSV,
    SEED,
    CIRCUITOS_CSV,
)

from estrategia_f1.sim.evaluacion_sim import (
    evaluar_simulador_en_fila,
    evaluar_simulador_en_dataset,
    resumir_evaluacion_simulador,
    resumir_evaluacion_por_grupo,
    imprimir_resumen_simulador,
    imprimir_resumen_global_simulador,
)


def main() -> None:
    df = pd.read_csv(DATASET_SIM_CSV)
    df_circuitos = pd.read_csv(CIRCUITOS_CSV)

    print("\nColumnas del dataset:")
    print(df.columns)

    print("\n¿Existe track_temp?")
    print("track_temp" in df.columns)

    print("\n¿Existe track_temp_cat?")
    print("track_temp_cat" in df.columns)

    if "track_temp" in df.columns:
        print("\nValores de track_temp:")
        print(df["track_temp"].describe())

    if "track_temp_cat" in df.columns:
        print("\nValores de track_temp_cat:")
        print(df["track_temp_cat"].value_counts())

    print("\nEjemplo de filas:")
    print(df[["track_temp_cat"]].head())

    # Parseamos la estrategia una sola vez al cargar el dataset
    if "strategy_compounds" in df.columns:
        df["strategy_compounds"] = df["strategy_compounds"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

    # Escogemos una fila aleatoria
    fila = df.sample(1, random_state=SEED).iloc[0]

    estrategia = fila.get("strategy_compounds", None)

    if estrategia is None:
        raise ValueError("La fila seleccionada no tiene estrategia válida.")

    # Evaluación individual
    evaluacion = evaluar_simulador_en_fila(fila, estrategia)
    imprimir_resumen_simulador(fila, estrategia, evaluacion, df_circuitos)

    # Evaluación global
    df_resultados = evaluar_simulador_en_dataset(df)
    resumen_global = resumir_evaluacion_simulador(df_resultados)

    imprimir_resumen_global_simulador(resumen_global)

    # Resúmenes por grupo
    print("Resumen por temperatura de pista:")
    print(resumir_evaluacion_por_grupo(df_resultados, "track_temp_cat").to_string(index=False))
    print()

    print("Resumen por desgaste:")
    print(resumir_evaluacion_por_grupo(df_resultados, "wear_index").to_string(index=False))
    print()

    print("Resumen por número de stints:")
    print(resumir_evaluacion_por_grupo(df_resultados, "n_stints").to_string(index=False))
    print()

    print("Resumen por circuito:")
    df_por_circuito = resumir_evaluacion_por_grupo(df_resultados, "circuit_key")

    if not df_por_circuito.empty and "circuit_key" in df_por_circuito.columns:
        if {"circuit_key", "circuit_short_name"}.issubset(df_circuitos.columns):
            df_por_circuito = df_por_circuito.merge(
                df_circuitos[["circuit_key", "circuit_short_name"]].drop_duplicates(),
                on="circuit_key",
                how="left",
            )

            columnas = ["circuit_short_name"] + [col for col in df_por_circuito.columns if col != "circuit_short_name"]
            df_por_circuito = df_por_circuito[columnas]

    print(df_por_circuito.to_string(index=False))
    print()


if __name__ == "__main__":
    main()