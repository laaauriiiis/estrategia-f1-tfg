"""
dataset_features.py
TODO
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from estrategia_f1.data.dataset_builder import convertir_a_datetime
from estrategia_f1.data.openf1_client import (
    openf1_descargar,
)
from estrategia_f1.acciones import (
    normalizar_estrategia,
    accion_id_desde_estrategia,
)

# Helpers---------------------------------------------------------------------------------------------------------------
def mediana_segura(x) -> float:
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    return float(x.median()) if len(x) else np.nan


def media_segura(x) -> float:
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    return float(x.mean()) if len(x) else np.nan

def categoria_temp_pista(temp_pista_c):
    if pd.isna(temp_pista_c):
        return np.nan
    if temp_pista_c < 25:
        return "baja"
    elif temp_pista_c <= 40:
        return "media"
    else:
        return "alta"

def condicion_meteo_desde_lluvia(valor_lluvia):
    if pd.isna(valor_lluvia):
        return np.nan
    return "lluvia" if valor_lluvia > 0 else "seco"

def categoria_por_cuantiles(valor, q33, q66):
    if pd.isna(valor):
        return np.nan
    if valor <= q33:
        return "baja"
    elif valor <= q66:
        return "media"
    else:
        return "alta"

# Features meteo--------------------------------------------------------------------------------------------------------
def calcular_features_meteo(fila_sesion_carrera: pd.Series) -> dict:
    """
    Estima:
    - track_temp_est (media últimos ~20 min)
    - track_temp_cat
    - weather_condition (lluvia/seco)
    - rainfall_est
    """
    sk = int(fila_sesion_carrera["session_key"])
    date_start = pd.to_datetime(fila_sesion_carrera.get("date_start"), utc=True, errors="coerce")

    w = openf1_descargar("weather", {"session_key": sk})
    if w.empty:
        return dict(track_temp_est=np.nan, track_temp_cat=np.nan, weather_condition=np.nan, rainfall_est=np.nan)

    w = convertir_a_datetime(w, ["date"])

    if pd.notna(date_start) and "date" in w.columns:
        w_prev = w[(w["date"] <= date_start) & (w["date"] >= date_start - pd.Timedelta(minutes=20))]
        if w_prev.empty:
            w_prev = w
    else:
        w_prev = w

    tt = media_segura(w_prev.get("track_temperature"))
    rf = media_segura(w_prev.get("rainfall"))

    return dict(
        track_temp_est=tt,
        track_temp_cat=categoria_temp_pista(tt),
        weather_condition=condicion_meteo_desde_lluvia(rf),
        rainfall_est=rf,
    )

# Pit loss--------------------------------------------------------------------------------------------------------------
def calcular_perdida_pit(fila_sesion_carrera: pd.Series) -> float:
    """
    Estima pit_loss (s):
    1) mediana lane_duration / pit_duration si existe y es razonable
    2) fallback usando diferencia pit_out_lap vs mediana laps normales por piloto
    """
    sk = int(fila_sesion_carrera["session_key"])

    p = openf1_descargar("pit", {"session_key": sk})
    if not p.empty:
        col = "lane_duration" if "lane_duration" in p.columns else ("pit_duration" if "pit_duration" in p.columns else None)
        if col:
            x = pd.to_numeric(p[col], errors="coerce").dropna()
            x = x[(x >= 10) & (x <= 80)]
            if len(x):
                return float(x.median())

    laps = openf1_descargar("laps", {"session_key": sk})
    if laps.empty:
        return np.nan

    if "is_pit_out_lap" not in laps.columns or "lap_duration" not in laps.columns:
        return np.nan

    laps = laps.copy()
    laps["lap_duration"] = pd.to_numeric(laps["lap_duration"], errors="coerce")
    laps = laps.dropna(subset=["driver_number", "lap_duration"])

    pit_out = laps[laps["is_pit_out_lap"].fillna(False)].copy()
    normal = laps[~laps["is_pit_out_lap"].fillna(False)].copy()

    if pit_out.empty or normal.empty:
        return np.nan

    deltas = []
    for drv, g_out in pit_out.groupby("driver_number"):
        base = normal.loc[normal["driver_number"] == drv, "lap_duration"].dropna()
        if len(base) < 5:
            continue
        base_med = float(base.median())
        d = (pd.to_numeric(g_out["lap_duration"], errors="coerce") - base_med).dropna()
        deltas.extend(d.tolist())

    if not deltas:
        return np.nan

    x = pd.Series(deltas, dtype=float)
    x = x[(x >= 5) & (x <= 90)]
    return float(x.median()) if len(x) else np.nan


# SC / VSC--------------------------------------------------------------------------------------------------------------
def calcular_flag_sc(fila_sesion_carrera: pd.Series) -> int:
    """
    1 si hubo Safety Car o VSC, si no 0.
    """
    sk = int(fila_sesion_carrera["session_key"])
    rc = openf1_descargar("race_control", {"session_key": sk})
    if rc.empty:
        return 0

    cat = rc.get("category", pd.Series([], dtype=str)).astype(str)
    msg = rc.get("message", pd.Series([], dtype=str)).astype(str)

    hay_sc = (cat.str.contains("SafetyCar", case=False, na=False)).any()
    hay_vsc = (msg.str.contains("VIRTUAL SAFETY CAR", case=False, na=False)).any()
    return int(hay_sc or hay_vsc)


# Neumáticos (life/pace/deg)--------------------------------------------------------------------------------------------
def calcular_features_neumaticos(fila_sesion_carrera: pd.Series) -> dict:
    """
    Devuelve life/pace/deg por compuesto (SOFT/MEDIUM/HARD), agregados a nivel de carrera.
    """
    sk = int(fila_sesion_carrera["session_key"])

    st = openf1_descargar("stints", {"session_key": sk})
    laps = openf1_descargar("laps", {"session_key": sk})

    out = {
        "life_soft": np.nan, "life_medium": np.nan, "life_hard": np.nan,
        "pace_soft": np.nan, "pace_medium": np.nan, "pace_hard": np.nan,
        "deg_soft": np.nan, "deg_medium": np.nan, "deg_hard": np.nan,
    }
    if st.empty or laps.empty:
        return out

    st = st.copy()
    laps = laps.copy()

    st["lap_start"] = pd.to_numeric(st["lap_start"], errors="coerce")
    st["lap_end"] = pd.to_numeric(st["lap_end"], errors="coerce")
    st["stint_length"] = st["lap_end"] - st["lap_start"] + 1

    st["compound"] = st["compound"].astype(str)
    st = st[st["compound"].isin(["SOFT", "MEDIUM", "HARD"])].copy()
    if st.empty:
        return out

    laps["lap_number"] = pd.to_numeric(laps.get("lap_number"), errors="coerce")
    laps["lap_duration"] = pd.to_numeric(laps.get("lap_duration"), errors="coerce")
    if "is_pit_out_lap" in laps.columns:
        laps = laps[~laps["is_pit_out_lap"].fillna(False)].copy()
    laps = laps.dropna(subset=["lap_number", "lap_duration", "driver_number"])

    # LIFE (mediana stint_length por compuesto)
    for comp in ["SOFT", "MEDIUM", "HARD"]:
        med = mediana_segura(st.loc[st["compound"] == comp, "stint_length"])
        if comp == "SOFT":
            out["life_soft"] = med
        elif comp == "MEDIUM":
            out["life_medium"] = med
        else:
            out["life_hard"] = med

    muestras_ritmo = {"SOFT": [], "MEDIUM": [], "HARD": []}
    betas = {"SOFT": [], "MEDIUM": [], "HARD": []}

    for drv, st_drv in st.groupby("driver_number"):
        laps_drv = laps[laps["driver_number"] == drv].copy()
        if laps_drv.empty:
            continue

        for _, fila_st in st_drv.iterrows():
            comp = fila_st["compound"]
            ls, le = fila_st["lap_start"], fila_st["lap_end"]
            if pd.isna(ls) or pd.isna(le):
                continue

            stint_laps = laps_drv[(laps_drv["lap_number"] >= ls) & (laps_drv["lap_number"] <= le)].copy()
            stint_laps = stint_laps.dropna(subset=["lap_duration", "lap_number"])
            if stint_laps.empty:
                continue

            # muestras de ritmo
            muestras_ritmo[comp].extend(stint_laps["lap_duration"].tolist())

            # degradación por stint: lap_duration ~ lap_in_stint
            stint_laps["lap_in_stint"] = stint_laps["lap_number"] - ls + 1
            if len(stint_laps) < 5:
                continue

            x = stint_laps["lap_in_stint"].to_numpy(dtype=float)
            y = stint_laps["lap_duration"].to_numpy(dtype=float)
            vx = np.var(x)
            if vx <= 1e-9:
                continue

            beta = float(np.cov(x, y, bias=True)[0, 1] / vx)
            if np.isfinite(beta):
                betas[comp].append(beta)

    for comp in ["SOFT", "MEDIUM", "HARD"]:
        pace = mediana_segura(muestras_ritmo[comp])
        deg = mediana_segura(betas[comp])

        if comp == "SOFT":
            out["pace_soft"] = pace
            out["deg_soft"] = deg
        elif comp == "MEDIUM":
            out["pace_medium"] = pace
            out["deg_medium"] = deg
        else:
            out["pace_hard"] = pace
            out["deg_hard"] = deg

    return out


# Resultados finales (finish_time, laps, dnf/dns/dsq)-------------------------------------------------------------------
def calcular_vueltas_y_tiempos_finales(fila_sesion_carrera: pd.Series) -> pd.DataFrame:
    """
    Devuelve df con: driver_number, finish_time_s, n_laps_driver, dnf, dns, dsq.
    """
    sk = int(fila_sesion_carrera["session_key"])
    sr = openf1_descargar("session_result", {"session_key": sk})
    if sr.empty:
        return pd.DataFrame(columns=["driver_number", "finish_time_s", "n_laps_driver", "dnf", "dns", "dsq"])

    sr = sr.copy()
    sr["driver_number"] = pd.to_numeric(sr["driver_number"], errors="coerce")
    sr["finish_time_s"] = pd.to_numeric(sr.get("duration"), errors="coerce")
    sr["n_laps_driver"] = pd.to_numeric(sr.get("number_of_laps"), errors="coerce")

    for f in ["dnf", "dns", "dsq"]:
        if f not in sr.columns:
            sr[f] = False

    return sr[["driver_number", "finish_time_s", "n_laps_driver", "dnf", "dns", "dsq"]]


# Acciones reales de pilotos--------------------------------------------------------------------------------------------
def calcular_acciones_pilotos(fila_sesion_carrera: pd.Series, mapa_inverso) -> pd.DataFrame:
    """
    Devuelve df: driver_number -> strategy_compounds, n_stints, action_id.

    mapa_inverso: dict[tuple[str, ...], int]
    """
    sk = int(fila_sesion_carrera["session_key"])
    st = openf1_descargar("stints", {"session_key": sk})
    if st.empty:
        return pd.DataFrame(columns=["driver_number", "strategy_compounds", "n_stints", "action_id"])

    st = st.copy()
    st = st[st["compound"].isin(["SOFT", "MEDIUM", "HARD"])].copy()
    if st.empty:
        return pd.DataFrame(columns=["driver_number", "strategy_compounds", "n_stints", "action_id"])

    st["driver_number"] = pd.to_numeric(st["driver_number"], errors="coerce")
    st["stint_number"] = pd.to_numeric(st["stint_number"], errors="coerce")

    filas = []
    for drv, g in st.groupby("driver_number"):
        g = g.sort_values("stint_number")
        seq = g["compound"].astype(str).tolist()

        seq_norm = normalizar_estrategia(seq)
        if not seq_norm:
            filas.append({"driver_number": drv, "strategy_compounds": None, "n_stints": len(seq), "action_id": -1})
            continue

        aid = accion_id_desde_estrategia(seq_norm, mapa_inverso)  # -> int (o -1)
        filas.append(
            {
                "driver_number": drv,
                "strategy_compounds": seq_norm,  # ya limpio y en mayúsculas
                "n_stints": len(seq_norm),
                "action_id": aid,
            }
        )
    return pd.DataFrame(filas)
