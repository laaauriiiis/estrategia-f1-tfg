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
    imprimir_resumen_simulador,
)

def main() -> None:
    df = pd.read_csv(DATASET_SIM_CSV)
    df_circuitos = pd.read_csv(CIRCUITOS_CSV)

    # Escogemos una fila aleatoria
    fila = df.sample(1, random_state=SEED).iloc[0]

    estrategia_raw = fila.get("strategy_compounds", None)

    if isinstance(estrategia_raw, str):
        estrategia = ast.literal_eval(estrategia_raw)
    else:
        estrategia = estrategia_raw

    if estrategia is None:
        raise ValueError("La fila seleccionada no tiene estrategia válida.")

    evaluacion = evaluar_simulador_en_fila(fila, estrategia)
    imprimir_resumen_simulador(fila, estrategia, evaluacion, df_circuitos)


if __name__ == "__main__":
    main()