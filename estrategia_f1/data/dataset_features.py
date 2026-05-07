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
    - track_temp_est (media últimos ~20 min antes de carrera)
    - track_temp_cat
    - weather_condition (lluvia/seco)
    - rainfall_est

    Evita fuga temporal usando solo registros meteorológicos anteriores
    o iguales al inicio de la carrera.
    """
    sk = int(fila_sesion_carrera["session_key"])
    date_start = pd.to_datetime(
        fila_sesion_carrera.get("date_start"),
        utc=True,
        errors="coerce",
    )

    try:
        w = openf1_descargar("weather", {"session_key": sk})
    except RuntimeError:
        return dict(
            track_temp_est=np.nan,
            track_temp_cat=np.nan,
            weather_condition=np.nan,
            rainfall_est=np.nan,
        )

    if w.empty:
        return dict(
            track_temp_est=np.nan,
            track_temp_cat=np.nan,
            weather_condition=np.nan,
            rainfall_est=np.nan,
        )

    w = convertir_a_datetime(w, ["date"])

    if pd.notna(date_start) and "date" in w.columns:
        w_prev = w[w["date"] <= date_start].copy()

        if w_prev.empty:
            return dict(
                track_temp_est=np.nan,
                track_temp_cat=np.nan,
                weather_condition=np.nan,
                rainfall_est=np.nan,
            )

        w_ultimos_20 = w_prev[
            w_prev["date"] >= date_start - pd.Timedelta(minutes=20)
        ].copy()

        if not w_ultimos_20.empty:
            w_prev = w_ultimos_20
    else:
        # Si no hay fecha válida, no podemos garantizar ausencia de fuga temporal.
        # Mejor devolver NaN antes que usar toda la sesión.
        return dict(
            track_temp_est=np.nan,
            track_temp_cat=np.nan,
            weather_condition=np.nan,
            rainfall_est=np.nan,
        )

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
    Calcula la pérdida media de tiempo en pit lane para una carrera.
    Si OpenF1 no devuelve datos de pit, devuelve NaN.
    """
    sk = int(fila_sesion_carrera["session_key"])

    try:
        p = openf1_descargar("pit", {"session_key": sk})
    except RuntimeError:
        return np.nan

    if p.empty:
        return np.nan

    if "pit_duration" in p.columns:
        vals = pd.to_numeric(p["pit_duration"], errors="coerce")
    elif "duration" in p.columns:
        vals = pd.to_numeric(p["duration"], errors="coerce")
    else:
        return np.nan

    vals = vals.dropna()
    vals = vals[(vals > 0) & (vals < 120)]

    return float(vals.median()) if len(vals) else np.nan


# SC / VSC--------------------------------------------------------------------------------------------------------------
def calcular_flag_sc(fila_sesion_carrera: pd.Series) -> int:
    """
    1 si hubo Safety Car o VSC, si no 0.

    Si OpenF1 no devuelve datos de race_control para una sesión concreta,
    se asume 0 para no interrumpir la construcción del dataset.
    """
    sk = int(fila_sesion_carrera["session_key"])

    try:
        rc = openf1_descargar("race_control", {"session_key": sk})
    except RuntimeError:
        return 0

    if rc.empty:
        return 0

    cat = rc.get("category", pd.Series([], dtype=str)).astype(str)
    msg = rc.get("message", pd.Series([], dtype=str)).astype(str)

    hay_sc = cat.str.contains("SafetyCar", case=False, na=False).any()
    hay_vsc = msg.str.contains("VIRTUAL SAFETY CAR", case=False, na=False).any()

    return int(hay_sc or hay_vsc)


# Neumáticos (life/pace/deg)--------------------------------------------------------------------------------------------
def calcular_features_neumaticos(
    fila_sesion_carrera: pd.Series,
    stints_df: pd.DataFrame | None = None,
) -> dict:
    """
    Calcula para SOFT / MEDIUM / HARD:

    - life_*   : vida útil media (vueltas)
    - pace_*   : ritmo medio por vuelta
    - deg_*    : degradación media (pendiente intra-stint)

    Si OpenF1 falla (404/429/etc.) o faltan datos,
    devuelve NaN.
    """
    sk = int(fila_sesion_carrera["session_key"])

    resultado_vacio = {
        "life_soft": np.nan,
        "life_medium": np.nan,
        "life_hard": np.nan,
        "pace_soft": np.nan,
        "pace_medium": np.nan,
        "pace_hard": np.nan,
        "deg_soft": np.nan,
        "deg_medium": np.nan,
        "deg_hard": np.nan,
    }

    try:
        if stints_df is not None:
            st = stints_df.copy()
        else:
            st = openf1_descargar("stints", {"session_key": sk})

        laps = openf1_descargar(
            "laps",
            {"session_key": sk},
        )

    except RuntimeError:
        return resultado_vacio

    if st.empty or laps.empty:
        return resultado_vacio

    st = st.copy()
    laps = laps.copy()

    if "compound" not in st.columns:
        return resultado_vacio

    if "lap_duration" not in laps.columns:
        return resultado_vacio

    st["compound"] = (
        st["compound"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    laps["lap_duration"] = pd.to_numeric(
        laps["lap_duration"],
        errors="coerce",
    )

    compuestos_validos = [
        "SOFT",
        "MEDIUM",
        "HARD",
    ]

    out = {}

    for compound in compuestos_validos:

        st_comp = st[
            st["compound"] == compound
        ].copy()

        if st_comp.empty:

            out[f"life_{compound.lower()}"] = np.nan
            out[f"pace_{compound.lower()}"] = np.nan
            out[f"deg_{compound.lower()}"] = np.nan

            continue

        # VIDA ÚTIL
        stint_lengths = []

        for c in [
            "lap_end",
            "lap_start",
        ]:
            if c not in st_comp.columns:
                break

        if (
            "lap_start" in st_comp.columns
            and "lap_end" in st_comp.columns
        ):

            lap_start = pd.to_numeric(
                st_comp["lap_start"],
                errors="coerce",
            )

            lap_end = pd.to_numeric(
                st_comp["lap_end"],
                errors="coerce",
            )

            stint_lengths = (
                lap_end - lap_start + 1
            ).dropna()

        life = (
            float(stint_lengths.median())
            if len(stint_lengths)
            else np.nan
        )

        # PACE
        pace = np.nan

        if "driver_number" in st_comp.columns:

            drivers = set(
                pd.to_numeric(
                    st_comp["driver_number"],
                    errors="coerce",
                ).dropna()
            )

            laps_comp = laps[
                pd.to_numeric(
                    laps.get("driver_number"),
                    errors="coerce",
                ).isin(drivers)
            ].copy()

            lap_times = pd.to_numeric(
                laps_comp["lap_duration"],
                errors="coerce",
            )

            lap_times = lap_times[
                (lap_times > 50)
                & (lap_times < 250)
            ]

            if len(lap_times):
                pace = float(
                    lap_times.median()
                )

        # DEGRADACIÓN
        degs = []

        if (
            "driver_number" in st_comp.columns
            and "lap_start" in st_comp.columns
        ):

            for _, stint_row in st_comp.iterrows():

                driver = stint_row.get(
                    "driver_number"
                )

                lap_start = stint_row.get(
                    "lap_start"
                )

                if pd.isna(driver) or pd.isna(lap_start):
                    continue

                laps_driver = laps[
                    pd.to_numeric(
                        laps.get("driver_number"),
                        errors="coerce",
                    ) == float(driver)
                ].copy()

                if (
                    "lap_number"
                    not in laps_driver.columns
                ):
                    continue

                laps_driver["lap_number"] = pd.to_numeric(
                    laps_driver["lap_number"],
                    errors="coerce",
                )

                laps_driver["lap_duration"] = pd.to_numeric(
                    laps_driver["lap_duration"],
                    errors="coerce",
                )

                laps_driver = laps_driver.dropna(
                    subset=[
                        "lap_number",
                        "lap_duration",
                    ]
                )

                laps_driver = laps_driver[
                    (laps_driver["lap_duration"] > 50)
                    & (laps_driver["lap_duration"] < 250)
                ]

                if len(laps_driver) < 2:
                    continue

                x = (
                    laps_driver["lap_number"]
                    - float(lap_start)
                    + 1
                )

                y = laps_driver[
                    "lap_duration"
                ]

                if len(x) < 2:
                    continue

                try:
                    beta = np.polyfit(
                        x,
                        y,
                        deg=1,
                    )[0]

                    if np.isfinite(beta):
                        degs.append(
                            float(beta)
                        )

                except Exception:
                    continue

        deg = (
            float(np.median(degs))
            if degs
            else np.nan
        )

        out[f"life_{compound.lower()}"] = life
        out[f"pace_{compound.lower()}"] = pace
        out[f"deg_{compound.lower()}"] = deg

    return out

# Resultados finales (finish_time, laps, dnf/dns/dsq)-------------------------------------------------------------------
def calcular_vueltas_y_tiempos_finales(fila_sesion_carrera: pd.Series) -> pd.DataFrame:
    """
    Obtiene vueltas completadas y tiempo final por piloto.
    Si OpenF1 falla por 404/429/etc., devuelve DataFrame vacío.
    """
    sk = int(fila_sesion_carrera["session_key"])

    try:
        sr = openf1_descargar("session_result", {"session_key": sk})
    except RuntimeError:
        return pd.DataFrame(
            columns=[
                "driver_number",
                "finish_time_s",
                "n_laps_driver",
                "dnf",
                "dns",
                "dsq",
            ]
        )

    if sr.empty:
        return pd.DataFrame(
            columns=[
                "driver_number",
                "finish_time_s",
                "n_laps_driver",
                "dnf",
                "dns",
                "dsq",
            ]
        )

    sr = sr.copy()

    if "driver_number" not in sr.columns:
        return pd.DataFrame(
            columns=[
                "driver_number",
                "finish_time_s",
                "n_laps_driver",
                "dnf",
                "dns",
                "dsq",
            ]
        )

    sr["driver_number"] = pd.to_numeric(sr["driver_number"], errors="coerce")

    if "duration" in sr.columns:
        sr["finish_time_s"] = pd.to_numeric(sr["duration"], errors="coerce")
    elif "finish_time" in sr.columns:
        sr["finish_time_s"] = pd.to_numeric(sr["finish_time"], errors="coerce")
    else:
        sr["finish_time_s"] = np.nan

    if "number_of_laps" in sr.columns:
        sr["n_laps_driver"] = pd.to_numeric(sr["number_of_laps"], errors="coerce")
    elif "laps" in sr.columns:
        sr["n_laps_driver"] = pd.to_numeric(sr["laps"], errors="coerce")
    else:
        sr["n_laps_driver"] = np.nan

    status_col = None
    for c in ["status", "classified", "result_status"]:
        if c in sr.columns:
            status_col = c
            break

    if status_col is not None:
        status = sr[status_col].astype(str).str.upper()
        sr["dnf"] = status.str.contains("DNF|RETIRED|WITHDRAWN", regex=True, na=False)
        sr["dns"] = status.str.contains("DNS|DID NOT START", regex=True, na=False)
        sr["dsq"] = status.str.contains("DSQ|DISQUALIFIED", regex=True, na=False)
    else:
        sr["dnf"] = False
        sr["dns"] = False
        sr["dsq"] = False

    return sr[
        [
            "driver_number",
            "finish_time_s",
            "n_laps_driver",
            "dnf",
            "dns",
            "dsq",
        ]
    ].copy()


# Acciones reales de pilotos--------------------------------------------------------------------------------------------
def calcular_acciones_pilotos(
    fila_sesion_carrera: pd.Series,
    mapa_inverso: dict,
    stints_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Reconstruye la estrategia real de neumáticos por piloto a partir de los stints.

    Si se pasa stints_df, reutiliza ese DataFrame y no vuelve a llamar a OpenF1.
    Esto evita duplicar llamadas al endpoint stints y reduce errores 429.
    """
    sk = int(fila_sesion_carrera["session_key"])

    if stints_df is not None:
        st = stints_df.copy()
    else:
        try:
            st = openf1_descargar("stints", {"session_key": sk})
        except RuntimeError:
            return pd.DataFrame(
                columns=[
                    "driver_number",
                    "action_id",
                    "strategy_compounds",
                    "n_stints",
                ]
            )

    if st.empty:
        return pd.DataFrame(
            columns=[
                "driver_number",
                "action_id",
                "strategy_compounds",
                "n_stints",
            ]
        )

    columnas_necesarias = {
        "driver_number",
        "compound",
        "stint_number",
    }

    if not columnas_necesarias.issubset(set(st.columns)):
        return pd.DataFrame(
            columns=[
                "driver_number",
                "action_id",
                "strategy_compounds",
                "n_stints",
            ]
        )

    st = st.copy()

    st["driver_number"] = pd.to_numeric(
        st["driver_number"],
        errors="coerce",
    )

    st["stint_number"] = pd.to_numeric(
        st["stint_number"],
        errors="coerce",
    )

    st["compound"] = (
        st["compound"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    st = st.dropna(
        subset=[
            "driver_number",
            "stint_number",
            "compound",
        ]
    ).copy()

    st = st[
        st["compound"].isin(["SOFT", "MEDIUM", "HARD"])
    ].copy()

    if st.empty:
        return pd.DataFrame(
            columns=[
                "driver_number",
                "action_id",
                "strategy_compounds",
                "n_stints",
            ]
        )

    filas = []

    for driver_number, grupo in st.groupby("driver_number"):
        grupo = grupo.sort_values("stint_number").copy()

        compuestos = grupo["compound"].tolist()

        # Eliminamos repeticiones consecutivas del mismo compuesto.
        # Ejemplo: SOFT, SOFT, MEDIUM -> SOFT, MEDIUM
        estrategia = []
        for c in compuestos:
            if not estrategia or estrategia[-1] != c:
                estrategia.append(c)

        n_stints = len(estrategia)

        if n_stints < 2 or n_stints > 4:
            action_id = np.nan
        else:
            action_id = mapa_inverso.get(tuple(estrategia), np.nan)

        filas.append(
            {
                "driver_number": int(driver_number),
                "action_id": action_id,
                "strategy_compounds": tuple(estrategia),
                "n_stints": n_stints,
            }
        )

    return pd.DataFrame(filas)

def calcular_perdida_pit_historica(sesiones_previas: pd.DataFrame) -> float:
    valores = []

    for _, r_prev in sesiones_previas.iterrows():
        try:
            v = calcular_perdida_pit(r_prev)
        except Exception:
            v = np.nan

        if pd.notna(v) and np.isfinite(v):
            valores.append(float(v))

    return float(np.median(valores)) if valores else np.nan

def calcular_features_neumaticos_historicas(sesiones_previas: pd.DataFrame) -> dict:
    acumulado = {
        "life_soft": [], "life_medium": [], "life_hard": [],
        "pace_soft": [], "pace_medium": [], "pace_hard": [],
        "deg_soft": [], "deg_medium": [], "deg_hard": [],
    }

    for _, r_prev in sesiones_previas.iterrows():
        try:
            feats = calcular_features_neumaticos(r_prev)
        except Exception:
            continue

        for k, v in feats.items():
            if v is not None and pd.notna(v) and np.isfinite(v):
                acumulado[k].append(float(v))

    return {
        k: float(np.median(vs)) if vs else np.nan
        for k, vs in acumulado.items()
    }
