"""
script_crear_datasets.py
TODO
"""

from __future__ import annotations

from estrategia_f1.config import (
    ID_COLS, ESTADO_COLS,
    FILTER_COLS, ACCION_COLS,
    TIEMPO_COL, DATASET_ML_CSV, DATASET_RL_CSV, DATASET_SIM_CSV,
)
from estrategia_f1.data.dataset_builder import construir_dataset, construir_datasets_derivados


def main():
    df = construir_dataset([2023, 2024, 2025], eliminar_dnfs=False)

    dataset_simulador, dataset_RL, dataset_ML = construir_datasets_derivados(
        df,
        id_cols=ID_COLS,
        estado_cols=ESTADO_COLS,
        accion_cols=ACCION_COLS,
        tiempo_col=TIEMPO_COL,
        filter_cols=FILTER_COLS,
    )

    out_sim = DATASET_SIM_CSV
    out_rl = DATASET_RL_CSV
    out_ml = DATASET_ML_CSV

    dataset_simulador.to_csv(out_sim, index=False)
    dataset_RL.to_csv(out_rl, index=False)
    dataset_ML.to_csv(out_ml, index=False)

    print("Guardados:")
    print(" -", out_sim)
    print(" -", out_rl)
    print(" -", out_ml)
    print("dataset_simulador:", dataset_simulador.shape)
    print("dataset_RL:", dataset_RL.shape)
    print("dataset_ML:", dataset_ML.shape)


if __name__ == "__main__":
    main()
