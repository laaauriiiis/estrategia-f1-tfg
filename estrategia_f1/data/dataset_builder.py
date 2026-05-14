"""
dataset_builder.py

Construcción del dataset experimental.

Este módulo contiene la lógica necesaria para:
- Descargar y combinar datos históricos de carreras, sesiones, resultados y stints procedentes de OpenF1.
- Calcular variables observadas por carrera y transformarlas en variables históricas usando solo carreras anteriores.
- Construir los datasets derivados utilizados por el simulador, el enfoque supervisado y el enfoque basado en valor.
"""

# IMPORTS
from __future__ import annotations
import numpy as np
import pandas as pd
from tqdm import tqdm
from estrategia_f1.config import (
    CIRCUITOS_CSV,
    RAIN_THRESHOLDS,
    WEAR_THRESHOLDS,
    CLAVES_NEUMATICOS
)
from estrategia_f1.data.openf1_client import openf1_descargar
from estrategia_f1.acciones import (
    MAPA_ACCIONES,
    MAPA_INVERSO
)

# CIRCUITOS ------------------------------------------------------------------------------------------------------------
def cargar_circuitos() -> pd.DataFrame:
    """
    Carga y valida la información estática de los circuitos.

    Parámetros
    ----------
    None

    Returns
    -------
    pd.DataFrame
        DataFrame con la información de los circuitos utilizada
        durante la construcción del dataset, incluyendo al menos
        el identificador del circuito y su longitud en kilómetros.
    """
    if not CIRCUITOS_CSV.exists():
        raise FileNotFoundError(
            f"No existe circuitos.csv en {CIRCUITOS_CSV}. "
            f"Colócalo en esa ruta o cambia CIRCUITOS_CSV."
        )

    circuitos = pd.read_csv(CIRCUITOS_CSV, sep=None, engine="python")
    if "track_length_km" not in circuitos.columns:
        raise ValueError("El dataset circuitos.csv debe incluir track_length_km.")

    # Se normalizan los tipos para garantizar consistencia durante los joins posteriores
    circuitos["circuit_key"] = pd.to_numeric(circuitos["circuit_key"], errors="coerce").astype("Int64")
    circuitos["track_length_km"] = pd.to_numeric(circuitos["track_length_km"], errors="coerce")
    return circuitos


# HELPERS GENERALES ----------------------------------------------------------------------------------------------------
def convertir_a_datetime(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Convierte columnas temporales a formato datetime con zona UTC.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame que contiene las columnas temporales
        a convertir.
    cols : list[str]
        Lista con los nombres de las columnas que deben
        transformarse a formato datetime.

    Returns
    -------
    pd.DataFrame
        DataFrame con las columnas existentes convertidas
        a tipo datetime con zona horaria UTC.

        Las columnas no presentes en el DataFrame se ignoran.
    """
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    return df


def asignar_constante(df: pd.DataFrame, col: str, valor):
    """
    Asigna a todas las observaciones de una carrera un mismo
    valor compartido.

    Durante la construcción del dataset final, algunas variables
    se calculan una única vez por Gran Premio (por ejemplo,
    la longitud del circuito, el número total de vueltas o la
    pérdida estimada en boxes), pero deben estar presentes en
    cada fila del dataset. Esta función replica dichos valores
    sobre todas las observaciones asociadas a la carrera,
    adaptándose automáticamente al formato de entrada cuando el
    valor se recibe como escalar, serie o secuencia.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame sobre el que se desea asignar la columna.
    col : str
        Nombre de la columna de salida.
    valor : Any
        Valor que se desea asignar. Puede ser un escalar,
        una secuencia, un array o una serie.

    Returns
    -------
    None
        La función modifica el DataFrame de entrada
        directamente.
    """

    # Los valores escalares simples pueden asignarse directamente a todas las filas
    if isinstance(valor, str) or valor is None:
        df[col] = valor
        return
    try:
        # Se preservan explícitamente los valores NaN sin intentar expandirlos
        if isinstance(valor, float) and np.isnan(valor):
            df[col] = valor
            return
    except Exception:
        pass

    # Si el valor es una secuencia, se adapta su longitud al número de observaciones de la carrera
    if isinstance(valor, (pd.Series, np.ndarray, list, tuple)):
        try:
            # Secuencia vacía: no hay información disponible
            if len(valor) == 0:
                df[col] = np.nan
            # Un único valor se replica para todas las filas
            elif len(valor) == 1:
                df[col] = valor[0]
            # Si la longitud coincide, se conserva la correspondencia fila a fila
            elif len(valor) == len(df):
                df[col] = list(valor)
            # En caso ambiguo, se utiliza el primer valor como representación de la carrera
            else:
                df[col] = valor[0]
        except TypeError:
            df[col] = valor
        return

    df[col] = valor


def existen_columnas(cols: list[str], frame: pd.DataFrame) -> list[str]:
    """
    Filtra una lista de columnas conservando únicamente
    aquellas presentes en un DataFrame.

    Parámetros
    ----------
    cols : list[str]
        Lista de nombres de columnas candidatas.
    frame : pd.DataFrame
        DataFrame sobre el que se desea comprobar
        la existencia de dichas columnas.

    Returns
    -------
    list[str]
        Lista con las columnas que existen realmente
        en el DataFrame.
    """
    return [c for c in cols if c in frame.columns]

def es_finito(valor) -> bool:
    """
    Comprueba si un valor puede interpretarse como un
    número finito.

    Parámetros
    ----------
    valor : Any
        Valor que se desea validar. Puede ser un número,
        una cadena numérica, None o un valor ausente.

    Returns
    -------
    bool
        True si el valor puede convertirse a tipo float
        y representa un número finito.

        False si el valor es nulo, ausente o no puede
        interpretarse como un número válido.
    """
    if valor is None or pd.isna(valor):
        return False
    try:
        return bool(np.isfinite(float(valor)))
    except (TypeError, ValueError):
        return False

# IMPORTS DIFERIDOS ----------------------------------------------------------------------------------------------------
# Importamos aquí para evitar la importación circular con dataset_features.py,
# que importa convertir_a_datetime desde este módulo
from estrategia_f1.data.dataset_features import (  # noqa: E402
    calcular_features_meteo,
    calcular_perdida_pit,
    calcular_flag_sc,
    calcular_features_neumaticos,
    calcular_vueltas_y_tiempos_finales,
    calcular_acciones_pilotos,
)

# CATEGORIZACIÓN -------------------------------------------------------------------------------------------------------
def categoria_lluvia(rp: float):
    """
    Discretiza la probabilidad histórica de lluvia en
    categorías ordinales.

    Parámetros
    ----------
    rp : float
        Probabilidad histórica de lluvia asociada a una
        carrera, expresada como valor entre 0 y 1.

    Returns
    -------
    str | float
        Categoría cualitativa de lluvia. Si el valor no es
        numéricamente válido, devuelve np.nan.

        Se utilizan umbrales fijos para evitar
        dependencias de estadísticas calculadas
        con carreras futuras.
    """
    if not es_finito(rp):
        return np.nan

    if rp < RAIN_THRESHOLDS["baja"]:
        return "baja"

    if rp < RAIN_THRESHOLDS["media"]:
        return "media"

    return "alta"


def categoria_wear(v: float):
    """
    Discretiza el desgaste histórico del circuito en
    categorías ordinales.

    Parámetros
    ----------
    v : float
        Índice numérico de desgaste calculado a partir
        de la degradación histórica de los compuestos.

    Returns
    -------
    str | float
        Categoría cualitativa de desgaste. Si el valor no
        es numéricamente válido, devuelve np.nan.

        Se utilizan umbrales fijos para evitar
        dependencias de estadísticas calculadas
        con carreras futuras.
    """
    if not es_finito(v):
        return np.nan

    if v < WEAR_THRESHOLDS["baja"]:
        return "baja"

    if v < WEAR_THRESHOLDS["media"]:
        return "media"

    return "alta"

# HISTÓRICOS SIN FUGA TEMPORAL -----------------------------------------------------------------------------------------
def _expanding_median_shift(s: pd.Series) -> pd.Series:
    """
    Calcula la mediana acumulada de observaciones pasadas.

    Parámetros
    ----------
    s : pd.Series
        Serie temporal ordenada cronológicamente con los
        valores observados de una variable.

    Returns
    -------
    pd.Series
        Serie donde cada fila contiene la mediana de todos
        los valores anteriores disponibles, excluyendo
        explícitamente el valor de la observación actual.

        Esta transformación permite construir variables
        históricas sin introducir fuga temporal.
    """
    return s.shift().expanding(min_periods=1).median()


def _expanding_mean_shift(s: pd.Series) -> pd.Series:
    """
    Calcula la media acumulada de observaciones pasadas.

    Parámetros
    ----------
    s : pd.Series
        Serie temporal ordenada cronológicamente con los
        valores observados de una variable.

    Returns
    -------
    pd.Series
        Serie donde cada fila contiene la media de todos
        los valores anteriores disponibles, excluyendo
        explícitamente el valor de la observación actual.

        Esta transformación permite construir variables
        históricas sin introducir fuga temporal.
    """
    return s.shift().expanding(min_periods=1).mean()


def _historico_mediana_con_fallback_global(sesiones: pd.DataFrame, col_real: str, col_salida: str, *,
    by: str = "circuit_key") -> None:
    """
    Construye una variable histórica mediante mediana acumulada
    con fallback global.

    Parámetros
    ----------
    sesiones : pd.DataFrame
        DataFrame de sesiones ordenado cronológicamente.
    col_real : str
        Nombre de la columna observada real a partir de la cual
        se calcula el histórico.
    col_salida : str
        Nombre de la columna histórica que se va a crear.
    by : str, optional
        Columna utilizada para agrupar el histórico específico,
        por defecto el circuito.

    Returns
    -------
    None
        La función modifica el DataFrame de entrada directamente,
        añadiendo la columna histórica indicada.

    """
    hist_circuito = sesiones.groupby(by, sort=False)[col_real].transform(_expanding_median_shift)
    hist_global = sesiones[col_real].shift().expanding(min_periods=1).median()
    sesiones[col_salida] = hist_circuito.fillna(hist_global)


def _historico_media_con_fallback_global(sesiones: pd.DataFrame, col_real: str, col_salida: str, *,
    by: str = "circuit_key") -> None:
    """
    Construye una variable histórica mediante media acumulada
    con fallback global.

    Parámetros
    ----------
    sesiones : pd.DataFrame
        DataFrame de sesiones ordenado cronológicamente.
    col_real : str
        Nombre de la columna observada real a partir de la cual
        se calcula el histórico.
    col_salida : str
        Nombre de la columna histórica que se va a crear.
    by : str, optional
        Columna utilizada para agrupar el histórico específico,
        por defecto el circuito.

    Returns
    -------
    None
        La función modifica el DataFrame de entrada directamente,
        añadiendo la columna histórica indicada.

        Se utiliza para construir variables probabilísticas
        o de frecuencia sin introducir fuga temporal.
    """
    hist_circuito = sesiones.groupby(by, sort=False)[col_real].transform(_expanding_mean_shift)
    hist_global = sesiones[col_real].shift().expanding(min_periods=1).mean()
    sesiones[col_salida] = hist_circuito.fillna(hist_global)


def _calcular_rain_prob_historica(sesiones: pd.DataFrame) -> pd.Series:
    """
    Calcula la probabilidad histórica de lluvia disponible
    antes de cada carrera.

    Parámetros
    ----------
    sesiones : pd.DataFrame
        DataFrame de sesiones ordenado cronológicamente, con
        información del circuito, mes y evento real de lluvia.

    Returns
    -------
    pd.Series
        Serie con la probabilidad histórica de lluvia estimada
        para cada carrera.

        La estimación se calcula siguiendo una jerarquía de
        fallback.
        En todos los casos se usan únicamente carreras anteriores,
        evitando que la carrera actual influya en su propia
        estimación.
    """
    # Prioridad 1: histórico del mismo circuito y mes.
    hist_circuito_mes = sesiones.groupby(["circuit_key", "month"], sort=False)["rain_event_real"].transform(
        _expanding_mean_shift
    )
    # Prioridad 2: histórico global del mismo mes.
    hist_mes_global = sesiones.groupby("month", sort=False)["rain_event_real"].transform(_expanding_mean_shift)

    # Prioridad 3: histórico global anterior a la carrera.
    hist_global = sesiones["rain_event_real"].shift().expanding(min_periods=1).mean()
    return hist_circuito_mes.fillna(hist_mes_global).fillna(hist_global)


def _aplicar_historicos_sin_fuga(sessions_all: pd.DataFrame) -> pd.DataFrame:
    """
    Construye las variables históricas del dataset evitando
    fuga temporal de información.

    Parámetros
    ----------
    sessions_all : pd.DataFrame
        DataFrame con las sesiones de carrera y las variables
        observadas reales calculadas para cada Gran Premio.

    Returns
    -------
    pd.DataFrame
        DataFrame ordenado cronológicamente con las variables
        históricas que se entregarán al modelo.

        Las columnas observadas de la carrera actual se sustituyen
        por agregados calculados únicamente con carreras anteriores.
    """
    # Se ordenan las carreras cronológicamente para que los acumulados históricos respeten la secuencia temporal
    sesiones = sessions_all.sort_values(["date_start", "year", "meeting_key"]).reset_index(drop=True).copy()

    # Variables continuas estimadas mediante mediana histórica
    _historico_mediana_con_fallback_global(sesiones, "pit_loss_real", "pit_loss_s")

    for k in CLAVES_NEUMATICOS:
        _historico_mediana_con_fallback_global(sesiones, f"{k}_real", k)

    # Variables probabilísticas estimadas mediante media histórica
    _historico_media_con_fallback_global(sesiones, "sc_flag_real", "sc_prob")

    # Probabilidad histórica de lluvia y discretización categórica
    sesiones["rain_prob_value"] = _calcular_rain_prob_historica(sesiones)
    sesiones["rain_prob_cat"] = sesiones["rain_prob_value"].apply(categoria_lluvia)

    # Índice de desgaste derivado de la degradación histórica estimada para los tres compuestos
    sesiones["wear_index_numeric"] = sesiones[["deg_soft", "deg_medium", "deg_hard"]].mean(axis=1)
    sesiones["wear_index"] = sesiones["wear_index_numeric"].apply(categoria_wear)

    return sesiones

# FEATURES OBSERVADAS POR CARRERA --------------------------------------------------------------------------------------
def _calcular_features_observadas_por_carrera(sessions_all: pd.DataFrame, cache_stints: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """
    Calcula las variables observadas reales de cada carrera.

    Parámetros
    ----------
    sessions_all : pd.DataFrame
        DataFrame con las sesiones de carrera a partir de las
        cuales se calculan las variables observadas.
    cache_stints : dict[int, pd.DataFrame]
        Diccionario utilizado para almacenar los stints descargados
        por session_key y reutilizarlos posteriormente.

    Returns
    -------
    pd.DataFrame
        DataFrame de sesiones con columnas observadas
        reales, identificadas con el sufijo *_real.

        Estas columnas sirven como materia prima para construir
        variables históricas sin fuga temporal, pero no se entregan
        directamente a los modelos.
    """
    registros: list[dict] = []

    for _, r in tqdm(sessions_all.iterrows(), total=len(sessions_all), desc="Features observadas por carrera"):
        sk = int(r["session_key"])
        # Los stints se descargan una sola vez por carrera y se guardan para reutilizarlos después en la reconstrucción
        # de acciones
        try:
            stints_df = openf1_descargar("stints", {"session_key": sk})
        except RuntimeError:
            stints_df = pd.DataFrame()

        cache_stints[sk] = stints_df
        dt = pd.to_datetime(r.get("date_start"), utc=True, errors="coerce")
        month = int(dt.month) if pd.notna(dt) else np.nan

        # Cálculo de variables observadas de meteorología y neumáticos
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

        # Se añaden los parámetros reales observados de cada compuesto
        for k in CLAVES_NEUMATICOS:
            registro[f"{k}_real"] = feats_neu.get(k, np.nan)

        registros.append(registro)

    features_race = pd.DataFrame(registros)

    # Se incorporan las variables observadas al DataFrame de sesiones
    return sessions_all.merge(features_race, on="session_key", how="left")

# CONSTRUCCIÓN DATASET BASE --------------------------------------------------------------------------------------------
def construir_dataset(temporadas: list[int], eliminar_dnfs: bool = False) -> pd.DataFrame:
    """
       Construye el dataset base a nivel piloto-carrera.

       Parámetros
       ----------
       temporadas : list[int]
           Lista de temporadas de Fórmula 1 que se desean incluir
           en la construcción del dataset.
       eliminar_dnfs : bool, optional
           Indica si deben eliminarse las observaciones correspondientes
           a abandonos, no salidas o descalificaciones.

       Returns
       -------
       pd.DataFrame
           Dataset base con una fila por piloto y carrera, incluyendo
           identificadores, variables del estado previo, estrategia real
           observada, tiempo final de carrera y flags de filtrado.

           Las variables históricas se calculan usando únicamente
           carreras anteriores para evitar fuga temporal de información.
       """
    circuitos = cargar_circuitos()
    cache_stints = {}

    # Descarga y normalización de meetings por temporada
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

    # Descarga y normalización de sesiones de carrera por temporada
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

    # Integración de información complementaria del meeting cuando no está disponible directamente en la sesión
    if not meetings_all.empty:
        sessions_all = sessions_all.merge(meetings_all, on=["year", "meeting_key"], how="left", suffixes=("", "_m"))
        sessions_all["circuit_key"] = sessions_all["circuit_key"].fillna(sessions_all["circuit_key_m"])
        sessions_all["date_start"] = sessions_all["date_start"].fillna(sessions_all["date_start_m"])
        sessions_all = sessions_all.drop(columns=[c for c in sessions_all.columns if c.endswith("_m")], errors="ignore")

    # Incorporación de información estática del circuito
    sessions_all = sessions_all.merge(circuitos[["circuit_key", "track_length_km"]], on="circuit_key", how="left")
    sessions_all = sessions_all.sort_values(["date_start", "year", "meeting_key"]).reset_index(drop=True)

    # 1) Se calculan variables observadas de la propia carrera (meteorología, SC, pit loss, degradación, etc.) como
    # materia prima para construir históricos
    # IMPORTANTE: estas columnas *_real nunca se entregan directamente a los modelos
    sessions_all = _calcular_features_observadas_por_carrera(sessions_all, cache_stints)

    # 2) Las variables *_real se transforman en agregados históricos mediante shift(), por lo que cada carrera
    # solo utiliza información disponible antes de su inicio
    sessions_all = _aplicar_historicos_sin_fuga(sessions_all)

    # 3) Construcción de observaciones piloto-carrera
    filas_totales = []
    for _, r in tqdm(sessions_all.iterrows(), total=len(sessions_all), desc="Construyendo piloto-carrera"):
        season = int(r["year"])
        race_id = int(r["meeting_key"])
        race_date = pd.to_datetime(r.get("date_start"), utc=True, errors="coerce")
        circuit_key = int(r["circuit_key"]) if pd.notna(r["circuit_key"]) else None

        sr = calcular_vueltas_y_tiempos_finales(r).copy()
        if sr.empty:
            continue

        # Recuperación y normalización de resultados finales para cada piloto de la carrera
        sr["finish_time_s"] = pd.to_numeric(sr["finish_time_s"], errors="coerce")
        sr["n_laps_driver"] = pd.to_numeric(sr["n_laps_driver"], errors="coerce")
        sr["s_per_lap"] = sr["finish_time_s"] / sr["n_laps_driver"]

        # Eliminación de resultados incompletos o con tiempos no plausibles para entrenamiento
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

        # El número total de vueltas se considera una variable conocida antes del inicio de la carrera
        n_laps = float(pd.to_numeric(sr["n_laps_driver"], errors="coerce").max())

        # Reconstrucción de la estrategia real observada a partir de los stints del piloto
        act = calcular_acciones_pilotos(
            r,
            MAPA_INVERSO,
            stints_df=cache_stints.get(int(r["session_key"])),
        )

        # Integración de resultados finales y estrategias reales
        df = sr.merge(act, on="driver_number", how="left")
        df = df[df["action_id"].notna() & (df["action_id"] >= 0)].copy()
        if df.empty:
            continue

        # Filtrado de abandonos, no salidas y descalificaciones
        if eliminar_dnfs:
            df = df[(~df["dnf"]) & (~df["dns"]) & (~df["dsq"])].copy()

        # Replicación de variables comunes de carrera sobre todas las observaciones piloto-carrera
        asignar_constante(df, "season", season)
        asignar_constante(df, "race_id", race_id)
        asignar_constante(df, "race_date", race_date)
        asignar_constante(df, "circuit_key", circuit_key)
        asignar_constante(df, "track_length_km",
                          float(r.get("track_length_km")) if pd.notna(r.get("track_length_km")) else np.nan)
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

    # Concatenación de todas las observaciones válidas
    out = pd.concat(filas_totales, ignore_index=True) if filas_totales else pd.DataFrame()
    if out.empty:
        return out

    # Selección final de columnas del dataset base
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


# DATASET DERIVADO ---------------------------------------------------------------------------------------------------
def preparar_dataset_experimental(df: pd.DataFrame, *, id_cols: list[str], estado_cols: list[str], accion_cols: list[str],
    tiempo_col: list[str], filter_cols: list[str]) -> pd.DataFrame:
    """
    Prepara el dataset final utilizado en los experimentos.

    Parámetros
    ----------
    df : pd.DataFrame
        Dataset base construido a nivel piloto-carrera.
    id_cols : list[str]
        Columnas identificadoras que deben conservarse.
    estado_cols : list[str]
        Columnas que representan el estado previo a la carrera.
    accion_cols : list[str]
        Columnas asociadas a la estrategia real observada.
    tiempo_col : list[str]
        Columna con el tiempo final real de carrera.
    filter_cols : list[str]
        Columnas auxiliares de filtrado, como DNF, DNS o DSQ.

    Returns
    -------
    pd.DataFrame
        Dataset final con las observaciones válidas para entrenamiento,
        simulación, validación y evaluación experimental.
    """
    id_cols = existen_columnas(id_cols, df)
    estado_cols = existen_columnas(estado_cols, df)
    accion_cols = existen_columnas(accion_cols, df)
    tiempo_col = existen_columnas(tiempo_col, df)
    filter_cols = existen_columnas(filter_cols, df)

    dataset = df.copy()

    # Se conservan únicamente observaciones completas y evaluables
    dataset = dataset[
        dataset["finish_time_s"].notna()
        & dataset["action_id"].notna()
        & (dataset["action_id"] >= 0)
        ].copy()

    # Selección final de columnas necesarias para los experimentos
    columnas = id_cols + estado_cols + accion_cols + tiempo_col + filter_cols
    columnas = existen_columnas(columnas, dataset)

    return dataset[columnas].copy()