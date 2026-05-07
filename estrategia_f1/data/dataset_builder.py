"""
dataset_builder.py
Construcción del dataset evitando fuga temporal sin repetir llamadas históricas a OpenF1.

Idea:
1. Se calculan las variables observadas de cada carrera una sola vez.
2. Se crean columnas históricas mediante shift + expanding, usando solo carreras anteriores.
3. El modelo recibe las columnas históricas, no los valores observados de la propia carrera.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from estrategia_f1.config import CIRCUITOS_CSV
from estrategia_f1.data.openf1_client import openf1_descargar

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


# Importamos aquí para evitar la importación circular con dataset_features.py,
# que importa convertir_a_datetime desde este módulo.
from estrategia_f1.data.dataset_features import (  # noqa: E402
    calcular_features_meteo,
    calcular_perdida_pit,
    calcular_flag_sc,
    calcular_features_neumaticos,
    calcular_vueltas_y_tiempos_finales,
    calcular_acciones_pilotos,
)


CLAVES_NEUMATICOS = [
    "life_soft", "life_medium", "life_hard",
    "pace_soft", "pace_medium", "pace_hard",
    "deg_soft", "deg_medium", "deg_hard",
]


def es_finito(valor) -> bool:
    """
    Comprueba si un valor numérico es finito, aceptando None/NaN sin romper.
    """
    if valor is None or pd.isna(valor):
        return False
    try:
        return bool(np.isfinite(float(valor)))
    except (TypeError, ValueError):
        return False


def categoria_lluvia(rp: float):
    """
    Categoriza la probabilidad histórica de lluvia con umbrales fijos.
    Evita cuantiles calculados con carreras futuras.
    """
    if not es_finito(rp):
        return np.nan
    if rp < 0.10:
        return "baja"
    if rp < 0.30:
        return "media"
    return "alta"


def categoria_wear(v: float):
    """
    Categoriza desgaste con umbrales fijos.
    Evita cuantiles calculados con carreras futuras.
    """
    if not es_finito(v):
        return np.nan
    if v < 0.04:
        return "bajo"
    if v < 0.08:
        return "medio"
    return "alto"


def _expanding_median_shift(s: pd.Series) -> pd.Series:
    """
    Para cada fila devuelve la mediana de valores anteriores, nunca el valor actual.
    """
    return s.shift().expanding(min_periods=1).median()


def _expanding_mean_shift(s: pd.Series) -> pd.Series:
    """
    Para cada fila devuelve la media de valores anteriores, nunca el valor actual.
    """
    return s.shift().expanding(min_periods=1).mean()


def _historico_mediana_con_fallback_global(
    sesiones: pd.DataFrame,
    col_real: str,
    col_salida: str,
    *,
    by: str = "circuit_key",
) -> None:
    """
    Crea una columna histórica por circuito usando solo carreras anteriores.
    Si no hay histórico de ese circuito, usa el histórico global anterior.
    """
    hist_circuito = sesiones.groupby(by, sort=False)[col_real].transform(_expanding_median_shift)
    hist_global = sesiones[col_real].shift().expanding(min_periods=1).median()
    sesiones[col_salida] = hist_circuito.fillna(hist_global)


def _historico_media_con_fallback_global(
    sesiones: pd.DataFrame,
    col_real: str,
    col_salida: str,
    *,
    by: str = "circuit_key",
) -> None:
    """
    Crea una columna histórica por circuito usando solo carreras anteriores.
    Si no hay histórico de ese circuito, usa el histórico global anterior.
    """
    hist_circuito = sesiones.groupby(by, sort=False)[col_real].transform(_expanding_mean_shift)
    hist_global = sesiones[col_real].shift().expanding(min_periods=1).mean()
    sesiones[col_salida] = hist_circuito.fillna(hist_global)


def _calcular_rain_prob_historica(sesiones: pd.DataFrame) -> pd.Series:
    """
    Probabilidad de lluvia histórica.
    Prioridad:
    1. mismo circuito + mismo mes, usando carreras anteriores;
    2. mismo mes global, usando carreras anteriores;
    3. histórico global anterior.
    """
    hist_circuito_mes = sesiones.groupby(["circuit_key", "month"], sort=False)["rain_event_real"].transform(
        _expanding_mean_shift
    )
    hist_mes_global = sesiones.groupby("month", sort=False)["rain_event_real"].transform(_expanding_mean_shift)
    hist_global = sesiones["rain_event_real"].shift().expanding(min_periods=1).mean()
    return hist_circuito_mes.fillna(hist_mes_global).fillna(hist_global)


def _calcular_features_observadas_por_carrera(
    sessions_all: pd.DataFrame,
    cache_stints: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """
    Calcula una vez por carrera las variables observadas que servirán como materia prima
    de las variables históricas. Estas columnas *_real no se entregan al modelo.
    """
    registros: list[dict] = []

    for _, r in tqdm(sessions_all.iterrows(), total=len(sessions_all), desc="Features observadas por carrera"):
        sk = int(r["session_key"])
        try:
            stints_df = openf1_descargar("stints", {"session_key": sk})
        except RuntimeError:
            stints_df = pd.DataFrame()

        cache_stints[sk] = stints_df
        dt = pd.to_datetime(r.get("date_start"), utc=True, errors="coerce")
        month = int(dt.month) if pd.notna(dt) else np.nan

        wfeat = calcular_features_meteo(r)
        feats_neu = calcular_features_neumaticos(r, stints_df=stints_df)

        registro = {
            "session_key": sk,
            "month": month,
            "track_temp_cat": wfeat.get("track_temp_cat", np.nan),
            "weather_condition": wfeat.get("weather_condition", np.nan),
            "rainfall_est_real": wfeat.get("rainfall_est", np.nan),
            "rain_event_real": 1 if es_finito(wfeat.get("rainfall_est", np.nan)) and wfeat.get("rainfall_est", 0) > 0 else 0,
            "sc_flag_real": calcular_flag_sc(r),
            "pit_loss_real": calcular_perdida_pit(r),
        }

        for k in CLAVES_NEUMATICOS:
            registro[f"{k}_real"] = feats_neu.get(k, np.nan)

        registros.append(registro)

    features_race = pd.DataFrame(registros)
    return sessions_all.merge(features_race, on="session_key", how="left")


def _aplicar_historicos_sin_fuga(sessions_all: pd.DataFrame) -> pd.DataFrame:
    """
    Sustituye las variables que no deberían usar la carrera actual por agregados históricos.
    Todas se calculan con shift(), por tanto usan solo carreras anteriores.
    """
    sesiones = sessions_all.sort_values(["date_start", "year", "meeting_key"]).reset_index(drop=True).copy()

    _historico_mediana_con_fallback_global(sesiones, "pit_loss_real", "pit_loss_s")

    for k in CLAVES_NEUMATICOS:
        _historico_mediana_con_fallback_global(sesiones, f"{k}_real", k)

    _historico_media_con_fallback_global(sesiones, "sc_flag_real", "sc_prob")

    sesiones["rain_prob_value"] = _calcular_rain_prob_historica(sesiones)
    sesiones["rain_prob_cat"] = sesiones["rain_prob_value"].apply(categoria_lluvia)

    sesiones["wear_index_numeric"] = sesiones[["deg_soft", "deg_medium", "deg_hard"]].mean(axis=1)
    sesiones["wear_index"] = sesiones["wear_index_numeric"].apply(categoria_wear)

    return sesiones


# Construcción dataset base---------------------------------------------------------------------------------------------
def construir_dataset(temporadas: list[int], eliminar_dnfs: bool = False) -> pd.DataFrame:
    """
    Construye el dataset base a nivel piloto-carrera:
    - une sessions/meetings;
    - calcula una vez por carrera las variables observadas;
    - sustituye las variables históricas por agregados con carreras anteriores;
    - une resultados (session_result) + acciones reales (stints -> action_id).
    """
    circuitos = cargar_circuitos()
    cache_stints = {}

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
    sessions_all = sessions_all.sort_values(["date_start", "year", "meeting_key"]).reset_index(drop=True)

    # 1) Calculamos cada carrera UNA sola vez.
    sessions_all = _calcular_features_observadas_por_carrera(sessions_all, cache_stints)

    # 2) Creamos las columnas históricas sin fuga temporal.
    sessions_all = _aplicar_historicos_sin_fuga(sessions_all)

    # 3) Construcción filas piloto-carrera.
    filas_totales = []
    for _, r in tqdm(sessions_all.iterrows(), total=len(sessions_all), desc="Construyendo piloto-carrera"):
        season = int(r["year"])
        race_id = int(r["meeting_key"])
        race_date = pd.to_datetime(r.get("date_start"), utc=True, errors="coerce")
        circuit_key = int(r["circuit_key"]) if pd.notna(r["circuit_key"]) else None

        sr = calcular_vueltas_y_tiempos_finales(r).copy()
        if sr.empty:
            continue

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

        # El número de vueltas del GP es conocido antes de la carrera, aunque aquí se obtiene del resultado.
        n_laps = float(pd.to_numeric(sr["n_laps_driver"], errors="coerce").max())

        act = calcular_acciones_pilotos(
            r,
            MAPA_INVERSO,
            stints_df=cache_stints.get(int(r["session_key"])),
        )

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
        asignar_constante(df, "race_date", race_date)
        asignar_constante(df, "circuit_key", circuit_key)
        asignar_constante(
            df,
            "track_length_km",
            float(r.get("track_length_km")) if pd.notna(r.get("track_length_km")) else np.nan,
        )
        asignar_constante(df, "n_laps", n_laps)

        asignar_constante(df, "wear_index", r.get("wear_index", np.nan))
        asignar_constante(df, "pit_loss_s", r.get("pit_loss_s", np.nan))
        asignar_constante(df, "track_temp_cat", r.get("track_temp_cat", np.nan))
        asignar_constante(df, "weather_condition", r.get("weather_condition", np.nan))
        asignar_constante(df, "rain_prob_cat", r.get("rain_prob_cat", np.nan))
        asignar_constante(df, "sc_prob", r.get("sc_prob", np.nan))

        for k in CLAVES_NEUMATICOS:
            asignar_constante(df, k, r.get(k, np.nan))

        filas_totales.append(df)

    out = pd.concat(filas_totales, ignore_index=True) if filas_totales else pd.DataFrame()
    if out.empty:
        return out

    columnas = [
        "season", "race_id", "race_date", "circuit_key",
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
def construir_datasets_derivados(
    df: pd.DataFrame,
    *,
    id_cols: list[str],
    estado_cols: list[str],
    accion_cols: list[str],
    tiempo_col: list[str],
    filter_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    dataset_ML = dataset_ML[dataset_ML["action_id"].notna() & (dataset_ML["action_id"] >= 0)].copy()
    dataset_ML = dataset_ML[(~dataset_ML["dnf"]) & (~dataset_ML["dns"]) & (~dataset_ML["dsq"])].copy()
    ml_cols = id_cols + estado_cols + ["action_id"] + filter_cols
    ml_cols = existen_columnas(ml_cols, dataset_ML)
    dataset_ML = dataset_ML[ml_cols].copy()

    return dataset_simulador, dataset_RL, dataset_ML
