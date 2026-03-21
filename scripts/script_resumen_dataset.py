from __future__ import annotations

import pandas as pd

from estrategia_f1.acciones import construir_mapa_acciones
from estrategia_f1.config import DATASET_SIM_CSV, DATASET_ML_CSV


def formatear_estrategia(estrategia: list[str]) -> str:
    return "(" + ", ".join(estrategia) + ")"


def calcular_resumen_dataset(path_csv: str | None = None) -> None:
    # Puedes usar DATASET_SIM_CSV o DATASET_ML_CSV.
    # Para acciones y top estrategias, mejor dataset_simulador.
    csv_path = path_csv or DATASET_SIM_CSV
    df = pd.read_csv(csv_path)

    mapa_acciones = construir_mapa_acciones()
    n_acciones = len(mapa_acciones)

    # ---------- Resumen general ----------
    resumen = {
        "Número de filas (observaciones)": len(df),
        "Número de carreras": df["race_id"].nunique(),
        "Número de temporadas": df["season"].nunique(),
        "Número de circuitos": df["circuit_key"].nunique(),
        "Número de acciones posibles": n_acciones,
    }

    # Estas dos solo existirán si decides guardarlas en el dataset
    if "driver_number" in df.columns:
        resumen["Número de pilotos"] = df["driver_number"].nunique()

    if "constructor_id" in df.columns:
        resumen["Número de equipos"] = df["constructor_id"].nunique()

    print("\n================ RESUMEN DEL DATASET ================\n")
    for k, v in resumen.items():
        print(f"{k}: {v}")

    # ---------- Distribución de acciones ----------
    # Reindexamos sobre las 108 acciones para contar también las que no aparecen
    freq_abs = df["action_id"].value_counts().reindex(range(n_acciones), fill_value=0).sort_index()
    freq_pct = (freq_abs / len(df)) * 100

    resumen_acciones = {
        "Frecuencia máxima (%)": freq_pct.max(),
        "Frecuencia mínima (%)": freq_pct.min(),
        "Frecuencia media (%)": freq_pct.mean(),
        "Desviación estándar": freq_pct.std(),
        "Nº acciones con frecuencia < 1%": (freq_pct < 1).sum(),
        "Nº acciones con frecuencia < 0.1%": (freq_pct < 0.1).sum(),
        "Nº acciones no observadas": (freq_pct == 0).sum(),
        "Nº acciones observadas": (freq_pct > 0).sum(),
    }

    print("\n================ DISTRIBUCIÓN DE ACCIONES ================\n")
    for k, v in resumen_acciones.items():
        if isinstance(v, float):
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")

    # ---------- Top estrategias ----------
    top_freq = df["action_id"].value_counts().head(10)

    print("\n================ TOP ESTRATEGIAS ================\n")
    for action_id, count in top_freq.items():
        estrategia = mapa_acciones[int(action_id)]
        pct = 100 * count / len(df)
        print(
            f"action_id={int(action_id):>3} | "
            f"{formatear_estrategia(estrategia):<30} | "
            f"{count:>4} obs | {pct:>6.2f}%"
        )


if __name__ == "__main__":
    calcular_resumen_dataset()