"""
script_crear_datasets.py

Generación y exportación del dataset experimental utilizado en el TFG.
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
    Ejecuta el proceso completo de generación del dataset experimental.

    El proceso incluye:
    1. Construcción del dataset base a partir de temporadas históricas.
    2. Preparación del dataset experimental final.
    3. Selección de variables de estado, acción y filtrado.
    4. Exportación del dataset resultante a formato CSV.
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
