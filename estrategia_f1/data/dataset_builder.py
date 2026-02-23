"""
dataset_builder.py
TODO
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from estrategia_f1.config import CIRCUITOS_CSV

from estrategia_f1.data.openf1_client import openf1_descargar
from estrategia_f1.data.dataset_features import (
    calcular_features_meteo,
    calcular_perdida_pit,
    calcular_flag_sc,
    calcular_features_neumaticos,
    calcular_vueltas_y_tiempos_finales,
    calcular_acciones_pilotos,
    categoria_por_cuantiles,
)

from estrategia_f1.acciones import (
    construir_mapa_acciones,
    construir_mapa_acciones_inverso,
)

# Construimos mapas
MAPA_ACCIONES = construir_mapa_acciones()
MAPA_INVERSO = construir_mapa_acciones_inverso(MAPA_ACCIONES)

# Circuitos-------------------------------------------------------------------------------------------------------------
def cargar_circuitos() -> pd.DataFrame:
    if not CIRCUITOS_CSV.exists():
        raise FileNotFoundError(
            f"No encuentro circuitos.csv en {CIRCUITOS_CSV}. "
            f"Colócalo en esa ruta o cambia CIRCUITOS_CSV."
        )

    circuitos = pd.read_csv(CIRCUITOS_CSV, sep=None, engine="python")
    if "track_length_km" not in circuitos.columns:
        raise ValueError("circuitos.csv debe incluir track_length_km")

    circuitos["circuit_key"] = pd.to_numeric(circuitos["circuit_key"], errors="coerce").astype("Int64")
    circuitos["track_length_km"] = pd.to_numeric(circuitos["track_length_km"], errors="coerce")
    return circuitos


# Helpers---------------------------------------------------------------------------------------------------------------
def convertir_a_datetime(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Convierte columnas a datetime UTC si existen.
    """
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    return df


def asignar_constante(df: pd.DataFrame, col: str, valor):
    """
    Asigna un valor constante (por carrera) a todas las filas.
    """
    if isinstance(valor, str) or valor is None:
        df[col] = valor
        return
    try:
        if isinstance(valor, float) and np.isnan(valor):
            df[col] = valor
            return
    except Exception:
        pass

    if isinstance(valor, (pd.Series, np.ndarray, list, tuple)):
        try:
            if len(valor) == 0:
                df[col] = np.nan
            elif len(valor) == 1:
                df[col] = valor[0]
            elif len(valor) == len(df):
                df[col] = list(valor)
            else:
                df[col] = valor[0]
        except TypeError:
            df[col] = valor
        return

    df[col] = valor


def existen_columnas(cols: list[str], frame: pd.DataFrame) -> list[str]:
    return [c for c in cols if c in frame.columns]


# Construcción dataset base---------------------------------------------------------------------------------------------
def construir_dataset(temporadas: list[int], eliminar_dnfs: bool = False) -> pd.DataFrame:
    """
    Construye el dataset base a nivel piloto-carrera:
    - une sessions/meetings
    - añade longitud circuito
    - calcula features por carrera
    - une resultados (session_result) + acciones reales (stints -> action_id)
    """
    circuitos = cargar_circuitos()

    # meetings por año
    meetings_all = []
    for y in temporadas:
        m = openf1_descargar("meetings", {"year": y})
        if m.empty:
            continue
        m["year"] = pd.to_numeric(m.get("year"), errors="coerce")
        m["meeting_key"] = pd.to_numeric(m.get("meeting_key"), errors="coerce")
        m["circuit_key"] = pd.to_numeric(m.get("circuit_key"), errors="coerce")
        m = convertir_a_datetime(m, ["date_start", "date_end"])
        meetings_all.append(m[["year", "meeting_key", "circuit_key", "date_start"]])
    meetings_all = pd.concat(meetings_all, ignore_index=True) if meetings_all else pd.DataFrame()

    # sessions Race por año
    sessions_all = []
    for y in temporadas:
        s = openf1_descargar("sessions", {"year": y, "session_type": "Race"})
        if s.empty:
            continue
        s["year"] = pd.to_numeric(s.get("year"), errors="coerce")
        s["session_key"] = pd.to_numeric(s.get("session_key"), errors="coerce")
        s["meeting_key"] = pd.to_numeric(s.get("meeting_key"), errors="coerce")
        s["circuit_key"] = pd.to_numeric(s.get("circuit_key"), errors="coerce")
        s = convertir_a_datetime(s, ["date_start", "date_end"])
        sessions_all.append(s[["year", "session_key", "meeting_key", "circuit_key", "date_start"]])
    sessions_all = pd.concat(sessions_all, ignore_index=True) if sessions_all else pd.DataFrame()

    if sessions_all.empty:
        raise RuntimeError("No he encontrado sesiones de tipo Race para esas temporadas.")

    # Join con meetings
    if not meetings_all.empty:
        sessions_all = sessions_all.merge(meetings_all, on=["year", "meeting_key"], how="left", suffixes=("", "_m"))
        sessions_all["circuit_key"] = sessions_all["circuit_key"].fillna(sessions_all["circuit_key_m"])
        sessions_all["date_start"] = sessions_all["date_start"].fillna(sessions_all["date_start_m"])
        sessions_all = sessions_all.drop(columns=[c for c in sessions_all.columns if c.endswith("_m")], errors="ignore")

    # Join longitud circuito
    sessions_all = sessions_all.merge(circuitos[["circuit_key", "track_length_km"]], on="circuit_key", how="left")

    # SC flags + sc_prob por circuito
    flags_sc = {}
    for _, r in tqdm(sessions_all.iterrows(), total=len(sessions_all), desc="SC/VSC por carrera"):
        sk = int(r["session_key"])
        flags_sc[sk] = calcular_flag_sc(r)

    sessions_all["sc_flag"] = sessions_all["session_key"].astype(int).map(flags_sc).fillna(0).astype(int)
    sc_prob_por_circuito = sessions_all.groupby("circuit_key")["sc_flag"].mean().to_dict()

    # rain_prob por (circuit_key, month)
    muestras_lluvia = []
    for _, r in tqdm(sessions_all.iterrows(), total=len(sessions_all), desc="Proxy lluvia por carrera"):
        wfeat = calcular_features_meteo(r)
        dt = pd.to_datetime(r.get("date_start"), utc=True, errors="coerce")
        month = int(dt.month) if pd.notna(dt) else np.nan
        rf = wfeat.get("rainfall_est", np.nan)
        muestras_lluvia.append(
            {"circuit_key": r["circuit_key"], "month": month, "rain_event": 1 if (pd.notna(rf) and rf > 0) else 0}
        )

    muestras_lluvia = pd.DataFrame(muestras_lluvia).dropna(subset=["circuit_key", "month"])
    prob_lluvia = (muestras_lluvia.groupby(["circuit_key", "month"])["rain_event"].mean()).to_dict()

    rp_vals = np.array(list(prob_lluvia.values()), dtype=float)
    rp_vals = rp_vals[np.isfinite(rp_vals)]
    rp_q33 = np.quantile(rp_vals, 0.33) if len(rp_vals) else 0.0
    rp_q66 = np.quantile(rp_vals, 0.66) if len(rp_vals) else 0.0

    # Construcción filas piloto-carrera
    filas_totales = []
    for _, r in tqdm(sessions_all.iterrows(), total=len(sessions_all), desc="Construyendo piloto-carrera"):
        season = int(r["year"])
        race_id = int(r["meeting_key"])
        circuit_key = int(r["circuit_key"]) if pd.notna(r["circuit_key"]) else None

        wfeat = calcular_features_meteo(r)
        pit_loss = calcular_perdida_pit(r)
        feats_neu = calcular_features_neumaticos(r)

        sr = calcular_vueltas_y_tiempos_finales(r).copy()
        sr["finish_time_s"] = pd.to_numeric(sr["finish_time_s"], errors="coerce")
        sr["n_laps_driver"] = pd.to_numeric(sr["n_laps_driver"], errors="coerce")

        sr["s_per_lap"] = sr["finish_time_s"] / sr["n_laps_driver"]

        # filtros “sanity”
        sr = sr[
            (sr["finish_time_s"].notna())
            & (sr["n_laps_driver"].notna())
            & (sr["finish_time_s"] > 2000)
            & (sr["s_per_lap"] > 50)
            & (sr["s_per_lap"] < 250)
        ].copy()
        sr = sr.drop(columns=["s_per_lap"], errors="ignore")

        if sr.empty:
            continue

        n_laps = float(pd.to_numeric(sr["n_laps_driver"], errors="coerce").max())

        act = calcular_acciones_pilotos(r, MAPA_INVERSO)

        # join resultados + acciones
        df = sr.merge(act, on="driver_number", how="left")
        df = df[df["action_id"].notna() & (df["action_id"] >= 0)].copy()
        if df.empty:
            continue

        if eliminar_dnfs:
            df = df[(~df["dnf"]) & (~df["dns"]) & (~df["dsq"])].copy()

        # constantes por carrera
        asignar_constante(df, "season", season)
        asignar_constante(df, "race_id", race_id)
        asignar_constante(df, "circuit_key", circuit_key)
        asignar_constante(
            df, "track_length_km", float(r.get("track_length_km")) if pd.notna(r.get("track_length_km")) else np.nan
        )
        asignar_constante(df, "n_laps", n_laps)

        for k, v in feats_neu.items():
            asignar_constante(df, k, v)

        asignar_constante(df, "wear_index_numeric", np.nan)
        asignar_constante(df, "pit_loss_s", pit_loss)

        asignar_constante(df, "track_temp_cat", wfeat.get("track_temp_cat", np.nan))
        asignar_constante(df, "weather_condition", wfeat.get("weather_condition", np.nan))

        dt = pd.to_datetime(r.get("date_start"), utc=True, errors="coerce")
        month = int(dt.month) if pd.notna(dt) else np.nan
        rp = prob_lluvia.get((circuit_key, month), np.nan) if circuit_key is not None and pd.notna(month) else np.nan

        asignar_constante(df, "rain_prob_value", rp)
        asignar_constante(df, "rain_prob_cat", categoria_por_cuantiles(rp, rp_q33, rp_q66))

        asignar_constante(
            df, "sc_prob", float(sc_prob_por_circuito.get(circuit_key, np.nan)) if circuit_key is not None else np.nan
        )

        filas_totales.append(df)

    out = pd.concat(filas_totales, ignore_index=True) if filas_totales else pd.DataFrame()
    if out.empty:
        return out

    # wear_index por cuantiles (media degradaciones)
    out["wear_index_numeric"] = out[["deg_soft", "deg_medium", "deg_hard"]].mean(axis=1)
    wvals = pd.to_numeric(out["wear_index_numeric"], errors="coerce").dropna().to_numpy(dtype=float)

    if len(wvals):
        w_q33 = np.quantile(wvals, 0.33)
        w_q66 = np.quantile(wvals, 0.66)
        out["wear_index"] = out["wear_index_numeric"].apply(lambda v: categoria_por_cuantiles(v, w_q33, w_q66))
    else:
        out["wear_index"] = np.nan

    out = out.drop(columns=["wear_index_numeric", "rain_prob_value"], errors="ignore")

    columnas = [
        "season", "race_id", "circuit_key",
        "track_length_km", "n_laps", "wear_index", "pit_loss_s",
        "track_temp_cat", "weather_condition", "rain_prob_cat", "sc_prob",
        "life_soft", "life_medium", "life_hard",
        "pace_soft", "pace_medium", "pace_hard",
        "deg_soft", "deg_medium", "deg_hard",
        "action_id", "strategy_compounds", "n_stints",
        "finish_time_s",
        "dnf", "dns", "dsq",
    ]
    columnas = [c for c in columnas if c in out.columns]
    return out[columnas]


# Derivados RL / ML / simulador-----------------------------------------------------------------------------------------
def construir_datasets_derivados(df: pd.DataFrame, *, id_cols: list[str], estado_cols: list[str], accion_cols: list[str],
    tiempo_col: list[str], filter_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Devuelve (dataset_simulador, dataset_RL, dataset_ML) a partir del df base.
    """
    id_cols = existen_columnas(id_cols, df)
    estado_cols = existen_columnas(estado_cols, df)
    accion_cols = existen_columnas(accion_cols, df)
    tiempo_col = existen_columnas(tiempo_col, df)
    filter_cols = existen_columnas(filter_cols, df)

    # SIM
    dataset_simulador = df.copy()
    dataset_simulador = dataset_simulador[
        dataset_simulador["finish_time_s"].notna()
        & dataset_simulador["action_id"].notna()
        & (dataset_simulador["action_id"] >= 0)
    ].copy()
    sim_cols = id_cols + estado_cols + accion_cols + tiempo_col + filter_cols
    sim_cols = existen_columnas(sim_cols, dataset_simulador)
    dataset_simulador = dataset_simulador[sim_cols].copy()

    # RL (solo estado, sin DNF/DNS/DSQ)
    dataset_RL = df.copy()
    dataset_RL = dataset_RL[(~dataset_RL["dnf"]) & (~dataset_RL["dns"]) & (~dataset_RL["dsq"])].copy()
    rl_cols = id_cols + estado_cols + filter_cols
    rl_cols = existen_columnas(rl_cols, dataset_RL)
    dataset_RL = dataset_RL[rl_cols].copy()

    # ML (estado + action_id, sin DNF/DNS/DSQ)
    dataset_ML = df.copy()
    dataset_ML = dataset_ML[df["action_id"].notna() & (df["action_id"] >= 0)].copy()
    dataset_ML = dataset_ML[(~dataset_ML["dnf"]) & (~dataset_ML["dns"]) & (~dataset_ML["dsq"])].copy()
    ml_cols = id_cols + estado_cols + ["action_id"] + filter_cols
    ml_cols = existen_columnas(ml_cols, dataset_ML)
    dataset_ML = dataset_ML[ml_cols].copy()

    return dataset_simulador, dataset_RL, dataset_ML
