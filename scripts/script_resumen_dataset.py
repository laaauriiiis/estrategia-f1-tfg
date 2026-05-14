from __future__ import annotations

from pathlib import Path

import pandas as pd

from estrategia_f1.acciones import construir_mapa_acciones
from estrategia_f1.config import DATASET_EXPERIMENTAL_CSV


def formatear_estrategia(estrategia: list[str]) -> str:
    return "(" + ", ".join(estrategia) + ")"


def detectar_columna_equipo(df: pd.DataFrame) -> str | None:
    posibles = [
        "constructor_id",
        "team_id",
        "team_name",
        "constructor_name",
        "team",
    ]
    for col in posibles:
        if col in df.columns:
            return col
    return None


def detectar_columna_piloto(df: pd.DataFrame) -> str | None:
    posibles = [
        "driver_number",
        "driver_id",
        "driver_name",
        "full_name",
    ]
    for col in posibles:
        if col in df.columns:
            return col
    return None


def construir_tabla_acciones(
    df: pd.DataFrame,
    mapa_acciones: dict[int, list[str]],
) -> pd.DataFrame:
    n_filas = len(df)
    n_acciones = len(mapa_acciones)

    freq_abs = (
        df["action_id"]
        .value_counts()
        .reindex(range(n_acciones), fill_value=0)
        .sort_index()
    )

    freq_pct = (freq_abs / n_filas) * 100 if n_filas > 0 else 0.0

    tabla = pd.DataFrame(
        {
            "action_id": list(range(n_acciones)),
            "estrategia": [formatear_estrategia(mapa_acciones[i]) for i in range(n_acciones)],
            "frecuencia_absoluta": freq_abs.values,
            "frecuencia_pct": freq_pct.values,
        }
    )

    tabla["observada"] = tabla["frecuencia_absoluta"] > 0
    return tabla


def imprimir_resumen_general(df: pd.DataFrame, n_acciones: int, tabla_acciones: pd.DataFrame) -> None:
    resumen = {
        "Número de filas (observaciones)": len(df),
        "Número de carreras": df["race_id"].nunique() if "race_id" in df.columns else "N/D",
        "Número de temporadas": df["season"].nunique() if "season" in df.columns else "N/D",
        "Número de circuitos": df["circuit_key"].nunique() if "circuit_key" in df.columns else "N/D",
        "Número de acciones posibles": n_acciones,
        "Número de acciones observadas": int(tabla_acciones["observada"].sum()),
        "Número de acciones no observadas": int((~tabla_acciones["observada"]).sum()),
    }

    col_piloto = detectar_columna_piloto(df)
    if col_piloto is not None:
        resumen["Número de pilotos"] = df[col_piloto].nunique()

    col_equipo = detectar_columna_equipo(df)
    if col_equipo is not None:
        resumen["Número de equipos"] = df[col_equipo].nunique()

    print("\n================ RESUMEN DEL DATASET ================\n")
    for k, v in resumen.items():
        print(f"{k}: {v}")


def imprimir_resumen_acciones(tabla_acciones: pd.DataFrame) -> None:
    freq_pct = tabla_acciones["frecuencia_pct"]

    resumen_acciones = {
        "Frecuencia máxima (%)": freq_pct.max(),
        "Frecuencia mínima (%)": freq_pct.min(),
        "Frecuencia media (%)": freq_pct.mean(),
        "Desviación estándar": freq_pct.std(),
        "Nº acciones con frecuencia < 1%": int((freq_pct < 1).sum()),
        "Nº acciones con frecuencia < 0.1%": int((freq_pct < 0.1).sum()),
        "Nº acciones no observadas": int((freq_pct == 0).sum()),
        "Nº acciones observadas": int((freq_pct > 0).sum()),
    }

    print("\n================ DISTRIBUCIÓN DE ACCIONES ================\n")
    for k, v in resumen_acciones.items():
        if isinstance(v, float):
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")


def imprimir_top_estrategias(tabla_acciones: pd.DataFrame, top_n: int = 10) -> None:
    top = (
        tabla_acciones[tabla_acciones["frecuencia_absoluta"] > 0]
        .sort_values(["frecuencia_absoluta", "action_id"], ascending=[False, True])
        .head(top_n)
    )

    print("\n================ TOP ESTRATEGIAS ================\n")
    for _, row in top.iterrows():
        print(
            f"action_id={int(row['action_id']):>3} | "
            f"{row['estrategia']:<35} | "
            f"{int(row['frecuencia_absoluta']):>4} obs | "
            f"{row['frecuencia_pct']:>6.2f}%"
        )


def imprimir_acciones_no_observadas(tabla_acciones: pd.DataFrame, max_mostrar: int = 30) -> None:
    no_obs = tabla_acciones[tabla_acciones["frecuencia_absoluta"] == 0]

    print("\n================ ACCIONES NO OBSERVADAS ================\n")
    print(f"Total: {len(no_obs)}")
    for _, row in no_obs.head(max_mostrar).iterrows():
        print(
            f"action_id={int(row['action_id']):>3} | "
            f"{row['estrategia']}"
        )

    if len(no_obs) > max_mostrar:
        print(f"... y {len(no_obs) - max_mostrar} más.")


def imprimir_tabla_completa_acciones(tabla_acciones: pd.DataFrame) -> None:
    print("\n================ TABLA COMPLETA DE LAS 108 ACCIONES ================\n")
    for _, row in tabla_acciones.iterrows():
        print(
            f"action_id={int(row['action_id']):>3} | "
            f"{row['estrategia']:<35} | "
            f"{int(row['frecuencia_absoluta']):>4} obs | "
            f"{row['frecuencia_pct']:>6.2f}% | "
            f"observada={bool(row['observada'])}"
        )


def imprimir_criterios_filtrado() -> None:
    print("\n================ CRITERIOS DE FILTRADO (DOCUMENTACIÓN) ================\n")
    criterios = [
        "Eliminación de observaciones sin finish_time_s",
        "Eliminación de observaciones sin n_laps_driver",
        "Filtrado de tiempos finales no plausibles: finish_time_s > 2000",
        "Filtrado de ritmo por vuelta no plausible: 50 < s_per_lap < 250",
        "Eliminación de estrategias sin action_id válido",
        "Exclusión de dnf, dns y dsq en los datasets de ML/RL",
    ]
    for i, criterio in enumerate(criterios, start=1):
        print(f"{i}. {criterio}")


def exportar_csv_acciones(tabla_acciones: pd.DataFrame, ruta_salida: str | Path = "resumen_acciones_108.csv") -> None:
    ruta_salida = Path(ruta_salida)
    tabla_acciones.to_csv(ruta_salida, index=False, encoding="utf-8")
    print(f"\nCSV exportado en: {ruta_salida.resolve()}")


def calcular_resumen_dataset(
    path_csv: str | None = None,
    exportar_csv: bool = True,
    imprimir_tabla_completa: bool = True,
) -> None:
    csv_path = path_csv or DATASET_EXPERIMENTAL_CSV
    df = pd.read_csv(csv_path)

    if "action_id" not in df.columns:
        raise ValueError("El dataset no contiene la columna 'action_id'.")

    mapa_acciones = construir_mapa_acciones()
    n_acciones = len(mapa_acciones)

    tabla_acciones = construir_tabla_acciones(df, mapa_acciones)

    imprimir_resumen_general(df, n_acciones, tabla_acciones)
    imprimir_resumen_acciones(tabla_acciones)
    imprimir_top_estrategias(tabla_acciones, top_n=10)
    imprimir_acciones_no_observadas(tabla_acciones)
    imprimir_criterios_filtrado()

    if imprimir_tabla_completa:
        imprimir_tabla_completa_acciones(tabla_acciones)

    if exportar_csv:
        exportar_csv_acciones(tabla_acciones)


if __name__ == "__main__":
    calcular_resumen_dataset()