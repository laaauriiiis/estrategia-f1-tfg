"""
script_crear_datasets.py

Generación del dataset experimental del proyecto.

Este script construye el dataset base a partir de los datos históricos
y exporta el dataset final utilizado en los experimentos del TFG.
"""

# IMPORTS
from __future__ import annotations
from estrategia_f1.config import (
    ID_COLS, ESTADO_COLS,
    FILTER_COLS, ACCION_COLS,
    TIEMPO_COL,
    DATASET_EXPERIMENTAL_CSV,
)
from estrategia_f1.data.dataset_builder import construir_dataset, preparar_dataset_experimental


def main() -> None:
    """
    Construye y guarda el dataset experimental.
    """
    df_base = construir_dataset([2023, 2024, 2025], eliminar_dnfs=False)

    dataset_experimental = preparar_dataset_experimental(
        df_base,
        id_cols=ID_COLS,
        estado_cols=ESTADO_COLS,
        accion_cols=ACCION_COLS,
        tiempo_col=TIEMPO_COL,
        filter_cols=FILTER_COLS,
    )

    DATASET_EXPERIMENTAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    dataset_experimental.to_csv(DATASET_EXPERIMENTAL_CSV, index=False)

    print("Dataset experimental guardado:")
    print(" -", DATASET_EXPERIMENTAL_CSV)
    print("shape:", dataset_experimental.shape)


if __name__ == "__main__":
    main()
