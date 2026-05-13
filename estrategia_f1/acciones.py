"""
acciones.py

Definición y manipulación del espacio discreto de acciones estratégicas.

Este módulo contiene la lógica necesaria para:
- Generar y validar estrategias de neumáticos compatibles con las restricciones del dominio.
- Convertir entre representaciones de estrategias y sus identificadores discretos (action_id).
- Filtrar acciones válidas para una observación concreta y construir estructuras auxiliares para entrenamiento,
    simulación y evaluación.
"""

# IMPORTS
from __future__ import annotations
import ast
from itertools import product
from typing import Iterable
import numpy as np
import pandas as pd
import estrategia_f1.config as cfg

# HELPERS INTERNOS -----------------------------------------------------------------------------------------------------
def a_float_o_nan(x) -> float:
    """
    Convierte a float si puede, si no, devuelve np.nan.

    Parámetros
    ----------
    x : Any
        Valor de entrada que se quiere convertir a tipo numérico.
        Puede ser un número, una cadena numérica, None o NaN.

    Returns
    -------
    float
        Valor convertido a float si la conversión es válida.
        En caso de que el valor sea None, NaN o no pueda convertirse,
        devuelve np.nan.
    """
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def limpiar_compuestos(secuencia: Iterable) -> list[str]:
    """
    Normaliza una secuencia de compuestos de neumáticos.

    Parámetros
    ----------
    secuencia : Iterable
        Secuencia de entrada con los compuestos de neumáticos.
        Puede recibirse como lista, tupla o array de NumPy.

    Returns
    -------
    list[str]
        Lista de compuestos convertidos a cadenas en mayúsculas,
        sin espacios en blanco adicionales.

        Si la entrada no corresponde a una secuencia válida,
        devuelve una lista vacía.
    """
    if isinstance(secuencia, np.ndarray):
        secuencia = secuencia.tolist()

    if not isinstance(secuencia, (list, tuple)):
        return []

    out: list[str] = []
    for x in secuencia:
        c = str(x).strip().upper()
        if c:
            out.append(c)
    return out

# NORMALIZACIÓN DE ESTRATEGIAS -----------------------------------------------------------------------------------------
def estrategia_valida(estrategia: list[str]) -> bool:
    """
    Comprueba si una estrategia de neumáticos cumple las restricciones
    definidas en el dominio del problema.

    Parámetros
    ----------
    estrategia : list[str]
        Secuencia ordenada de compuestos de neumáticos que representa
        una estrategia candidata.

    Returns
    -------
    bool
        True si la estrategia cumple todas las restricciones del dominio:
        número permitido de stints, uso exclusivo de compuestos válidos
        y presencia de al menos dos compuestos distintos.

        False en caso contrario.
    """
    estrategia = limpiar_compuestos(estrategia)

    if not (cfg.MIN_STINTS <= len(estrategia) <= cfg.MAX_STINTS):
        return False
    if any(c not in cfg.COMPUESTOS for c in estrategia):
        return False
    if len(set(estrategia)) < 2:
        return False

    return True

def normalizar_estrategia(x) -> list[str] | None:
    """
    Convierte una estrategia a una representación normalizada y válida.

    Parámetros
    ----------
    x : Any
        Estrategia de entrada en cualquiera de los formatos soportados.
        Puede recibirse como lista, tupla, array de NumPy, cadena con
        representación de lista o valor nulo.

    Returns
    -------
    list[str] | None
        Estrategia convertida a una lista de compuestos normalizados
        en mayúsculas si la entrada es válida y cumple las restricciones
        del dominio.

        En caso contrario, devuelve None.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None

    if isinstance(x, (list, tuple, np.ndarray)):
        seq = list(x)

    elif isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() == "nan":
            return None
        try:
            seq = ast.literal_eval(s)
            if not isinstance(seq, (list, tuple, np.ndarray)):
                return None
        except Exception:
            return None
    else:
        return None

    estrategia = limpiar_compuestos(seq)
    if estrategia_valida(estrategia):
        return estrategia
    return None

# ESPACIO DE ACCIONES --------------------------------------------------------------------------------------------------
def construir_mapa_acciones() -> dict[int, list[str]]:
    """
    Genera el espacio discreto de acciones válidas del problema.

    Parámetros
    ----------
    None

    Returns
    -------
    dict[int, list[str]]
        Diccionario que asocia cada identificador de acción
        (action_id) con una estrategia de neumáticos representada
        como una secuencia ordenada de compuestos.

        Las estrategias generadas cumplen las restricciones del
        dominio.
    """
    mapa: dict[int, list[str]] = {}
    accion_id = 0

    # Recorremos el número de stints permitidos
    for n_stints in range(cfg.MIN_STINTS, cfg.MAX_STINTS + 1):
        # Todas las secuencias posibles con repetición (para mantener el orden)
        for secuencia in product(cfg.COMPUESTOS, repeat=n_stints):
            if len(set(secuencia)) < 2:
                continue
            mapa[accion_id] = list(secuencia)
            accion_id += 1

    return mapa

MAPA_ACCIONES = construir_mapa_acciones()

def construir_mapa_acciones_inverso(mapa_acciones: dict[int, list[str]]) -> dict[tuple[str, ...], int]:
    """
    Construye el mapa inverso del espacio discreto de acciones.

    Parámetros
    ----------
    mapa_acciones : dict[int, list[str]]
        Diccionario que asocia cada identificador de acción
        (action_id) con su estrategia de neumáticos correspondiente.

    Returns
    -------
    dict[tuple[str, ...], int]
        Diccionario que asocia cada estrategia normalizada,
        representada como una tupla de compuestos, con su
        identificador de acción correspondiente.
    """
    mapa_inverso: dict[tuple[str, ...], int] = {}
    for accion_id, lista_compuestos in mapa_acciones.items():
        mapa_inverso[tuple(limpiar_compuestos(lista_compuestos))] = accion_id
    return mapa_inverso

def estrategia_desde_accion_id(accion_id: int, mapa_acciones: dict[int, list[str]]) -> list[str]:
    """
    Obtiene la estrategia asociada a un identificador de acción.

    Parámetros
    ----------
    accion_id : int
        Identificador entero de la acción dentro del espacio discreto
        de estrategias.
    mapa_acciones : dict[int, list[str]]
        Diccionario que asocia cada action_id con su estrategia
        de neumáticos correspondiente.

    Returns
    -------
    list[str]
        Secuencia ordenada de compuestos que representa la estrategia
        asociada al identificador proporcionado.
    """
    if accion_id not in mapa_acciones:
        raise KeyError(f"La accion_id = {accion_id} no existe en mapa_acciones")

    estrategia = mapa_acciones[accion_id]
    if isinstance(estrategia, str):
        estrategia = ast.literal_eval(estrategia)

    if not isinstance(estrategia, (list, tuple)):
        raise TypeError(f"Estrategia inválida para accion_id = {accion_id}: {estrategia}")

    return limpiar_compuestos(estrategia)

def accion_id_desde_estrategia(estrategia: list[str], mapa_inverso: dict[tuple[str, ...], int]) -> int:
    """
    Obtiene el identificador de acción asociado a una estrategia.

    Parámetros
    ----------
    estrategia : list[str]
        Secuencia ordenada de compuestos que representa una
        estrategia de neumáticos.
    mapa_inverso : dict[tuple[str, ...], int]
        Diccionario que asocia cada estrategia normalizada con
        su identificador de acción correspondiente.

    Returns
    -------
    int
        Identificador entero de la estrategia dentro del espacio
        discreto de acciones.

        Si la estrategia no es válida o no existe en el mapa,
        devuelve -1.
    """
    estrategia = limpiar_compuestos(estrategia)
    if not estrategia_valida(estrategia):
        return -1
    return mapa_inverso.get(tuple(estrategia), -1)


# VALIDACIÓN DE ACCIONES -----------------------------------------------------------------------------------------------
def compuestos_disponibles(fila: pd.Series) -> set[str]:
    """
    Determina los compuestos de neumáticos disponibles para una carrera.

    Parámetros
    ----------
    fila : pd.Series
        Observación del dataset que contiene los parámetros estimados
        de ritmo (pace), degradación (deg) y vida útil (life) para
        cada compuesto.

    Returns
    -------
    set[str]
        Conjunto de compuestos disponibles para la observación
        analizada.

        Un compuesto se considera disponible únicamente si dispone
        de valores numéricos válidos para sus parámetros de ritmo,
        degradación y vida útil.
    """
    disponibles: set[str] = set()
    for comp in ("soft", "medium", "hard"):
        pace = pd.to_numeric(fila.get(f"pace_{comp}"), errors="coerce")
        deg  = pd.to_numeric(fila.get(f"deg_{comp}"), errors="coerce")
        life = pd.to_numeric(fila.get(f"life_{comp}"), errors="coerce")

        if np.isfinite(pace) and np.isfinite(deg) and np.isfinite(life):
            disponibles.add(comp.upper())

    return disponibles

def acciones_validas_para_fila(fila: pd.Series, mapa_acciones: dict[int, list[str]]) -> list[int]:
    """
    Filtra las acciones compatibles con una observación concreta.

    Parámetros
    ----------
    fila : pd.Series
        Observación del dataset con los parámetros disponibles
        para la carrera analizada.
    mapa_acciones : dict[int, list[str]]
        Diccionario que asocia cada identificador de acción
        (action_id) con su estrategia correspondiente.

    Returns
    -------
    list[int]
        Lista de identificadores de acción cuyas estrategias
        utilizan únicamente compuestos disponibles en la
        observación analizada.
    """
    disponibles = compuestos_disponibles(fila)

    validas: list[int] = []
    for accion_id, lista_compuestos in mapa_acciones.items():
        lista_compuestos_limpia = limpiar_compuestos(lista_compuestos)
        if all(c in disponibles for c in lista_compuestos_limpia):
            validas.append(accion_id)

    return validas


# HELPERS DE SIMULACIÓN ------------------------------------------------------------------------------------------------
def multiplicador_desgaste(wear_categoria: str | None) -> float:
    """
    Obtiene el multiplicador de degradación asociado al desgaste
    del circuito.

    Parámetros
    ----------
    wear_categoria : str | None
        Categoría de desgaste del circuito para la observación
        analizada. Puede tomar valores como "baja", "media"
        o "alta".

    Returns
    -------
    float
        Factor multiplicativo utilizado para ajustar la
        degradación de los neumáticos en el simulador.

        Si no se proporciona una categoría válida,
        devuelve 1.0.
    """
    if wear_categoria is None:
        return 1.0
    key = str(wear_categoria).strip().lower()
    return float(cfg.WEAR_MAP.get(key, 1.0))

def obtener_parametros_compuesto(fila: pd.Series, compuesto: str) -> tuple[float, float, float]:
    """
    Recupera los parámetros asociados a un compuesto de neumáticos.

    Parámetros
    ----------
    fila : pd.Series
        Observación del dataset que contiene los parámetros
        estimados de cada compuesto.
    compuesto : str
        Nombre del compuesto de neumáticos que se desea consultar.

    Returns
    -------
    tuple[float, float, float]
        Tupla con los parámetros del compuesto en el siguiente orden:
        ritmo medio por vuelta (pace), degradación por vuelta (deg)
        y vida útil estimada (life).
    """
    c = str(compuesto).strip().lower()
    if c not in ("soft", "medium", "hard"):
        raise ValueError(f"Compuesto inválido: {compuesto}")

    pace = pd.to_numeric(fila.get(f"pace_{c}"), errors="coerce")
    deg  = pd.to_numeric(fila.get(f"deg_{c}"), errors="coerce")
    life = pd.to_numeric(fila.get(f"life_{c}"), errors="coerce")

    if not (np.isfinite(pace) and np.isfinite(deg) and np.isfinite(life)):
        raise ValueError(f"Parámetros no disponibles para {compuesto}: pace/deg/life inválidos")

    deg = float(np.clip(float(deg), float(cfg.DEG_MIN), float(cfg.DEG_MAX)))
    return float(pace), float(deg), float(life)

def elegir_estrategia_baseline(fila: pd.Series) -> list[str] | None:
    """
    Selecciona la estrategia base utilizada como referencia
    durante la evaluación.

    Parámetros
    ----------
    fila : pd.Series
        Observación del dataset con la información de la carrera
        y los parámetros disponibles para cada compuesto.

    Returns
    -------
    list[str] | None
        Estrategia de neumáticos utilizada como baseline para
        la observación analizada.

        Se prioriza la estrategia real observada en los datos
        históricos si es válida y compatible con los compuestos
        disponibles. En caso contrario, se selecciona una
        estrategia alternativa siguiendo un orden de prioridad
        predefinido.

        Si no es posible construir una estrategia válida,
        devuelve None.
    """
    # Se prioriza la estrategia real observada en el dataset
    action_id = fila.get("action_id", None)

    if action_id is not None and not pd.isna(action_id):
        try:
            action_id_int = int(action_id)
        except (TypeError, ValueError):
            action_id_int = None

        if action_id_int is not None:
            estrategia_real = MAPA_ACCIONES.get(action_id_int)
            if estrategia_real is not None:
                # La estrategia histórica solo se utiliza si todos sus compuestos disponen de parámetros válidos
                disponibles = compuestos_disponibles(fila)
                if disponibles and all(c in disponibles for c in estrategia_real):
                    return estrategia_real
                # Si disponibles es vacío o no compatible, cae al fallback

    # Fallback: Selección determinista basada en prioridades predefinidas
    disponibles = compuestos_disponibles(fila)

    for estrategia in cfg.BASELINE_PRIORIDAD:
        if all(c in disponibles for c in estrategia):
            return estrategia

    # Como último recurso, se construye una estrategia mínima a partir de los compuestos disponibles en la observación
    disp = sorted(list(disponibles))
    if len(disp) >= 2:
        return disp[:2]
    if len(disp) == 1:
        return [disp[0], disp[0]]

    # Si no existen compuestos utilizables, la observación no puede evaluarse bajo el simulador
    return None

# HELPERS DE ENTRENAMIENTO ---------------------------------------------------------------------------------------------
def construir_grupos(df: pd.DataFrame) -> np.ndarray:
    """
    Construye los identificadores de agrupación por carrera (race_id).

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con las observaciones del dataset.

    Returns
    -------
    np.ndarray
        Array con los identificadores de carrera (race_id)
        asociados a cada observación.

        Esta estructura se utiliza para realizar particiones
        entrenamiento-test agrupadas por carrera, evitando
        fugas de información entre observaciones del mismo
        Gran Premio.
    """
    if "race_id" not in df.columns:
        raise KeyError("El dataframe no tiene la columna 'race_id' para agrupar.")
    return df["race_id"].astype(str).to_numpy()

def construir_estado_df(df: pd.DataFrame, *, columnas: list[str], columnas_excluir: list[str] | None = None,
        imputar_numericas: bool = True,) -> pd.DataFrame:
    """
    Construye la representación tabular del estado utilizada por los modelos de aprendizaje.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original con las observaciones del dataset.
    columnas : list[str]
        Lista de columnas que deben formar parte del estado.
    columnas_excluir : list[str] | None, optional
        Columnas que deben eliminarse de la representación final del estado.
    imputar_numericas : bool, optional
        Indica si los valores numéricos ausentes deben
        imputarse mediante la mediana de cada variable.

    Returns
    -------
    pd.DataFrame
        DataFrame con la representación final del estado,
        incluyendo únicamente las variables seleccionadas
        y aplicando las transformaciones necesarias para
        su uso en entrenamiento y evaluación.
    """
    if columnas_excluir is None:
        columnas_excluir = []

    faltan = set(columnas) - set(df.columns)
    if faltan:
        raise KeyError(f"Faltan las columnas {sorted(faltan)} en el DataFrame.")

    X = df.loc[:, columnas].copy()
    cols_a_eliminar = [c for c in columnas_excluir if c in X.columns]
    if cols_a_eliminar:
        X = X.drop(columns=cols_a_eliminar)

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        # Los valores infinitos se tratan como ausentes para mantener consistencia numérica
        X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan)

        if imputar_numericas:
            # La imputación por mediana reduce el impacto de valores extremos sobre la distribución
            medianas = X[num_cols].median(numeric_only=True)
            X[num_cols] = X[num_cols].fillna(medianas)

    return X

def columnas_numericas(df: pd.DataFrame) -> list[str]:
    """
    Identifica las columnas numéricas de un DataFrame.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame sobre el que se desea identificar
        las variables numéricas.

    Returns
    -------
    list[str]
        Lista con los nombres de las columnas cuyo
        tipo de dato es numérico.
    """
    return list(df.select_dtypes(include=[np.number]).columns)


# OUTPUT ---------------------------------------------------------------------------------------------------------------
def imprimir_resumen_evaluacion(resultados: pd.DataFrame) -> None:
    """
    Muestra por consola un resumen agregado de las métricas
    obtenidas durante la evaluación.

    Parámetros
    ----------
    resultados : pd.DataFrame
        DataFrame con los resultados de evaluación de una
        política, incluyendo métricas de mejora relativa,
        proximidad al Oracle y rendimiento Top-k.

    Returns
    -------
    None
        Esta función no devuelve ningún valor. Su objetivo
        es presentar un resumen formateado de los resultados
        experimentales.
    """
    if resultados.empty:
        print("No hay resultados evaluables.")
        return

    print("\n-------------------------- EVALUACIÓN -------------------------")

    print("\n---------------------- Baseline vs Policy ---------------------")
    print(f"ΔT(policy - baseline) media   : {resultados['delta_policy_vs_baseline'].mean():.3f} s")
    print(f"ΔT(policy - baseline) mediana : {resultados['delta_policy_vs_baseline'].median():.3f} s")
    print(f"% carreras policy < baseline  : {(resultados['delta_policy_vs_baseline'] < 0).mean() * 100:.1f}%")

    print("\n----------------------- Oracle y Regret -----------------------")
    print(f"Regret(policy) medio          : {resultados['regret_policy'].mean():.3f} s")
    print(f"Regret(policy) mediano        : {resultados['regret_policy'].median():.3f} s")
    print(f"% policy = oracle             : {(resultados['regret_policy'] <= 1e-9).mean() * 100:.1f}%")

    print("\n----------------------------- TopK ----------------------------")
    for k in cfg.TOPK:
        col_regret = f"regret@{k}"
        col_hit = f"hit@{k}"

        if col_regret in resultados.columns:
            print(f"Regret@{k} medio : {resultados[col_regret].mean():.3f} s")

        if col_hit in resultados.columns:
            print(f"Hit@{k}          : {resultados[col_hit].mean() * 100:.1f}%")

    print("\n===============================================================\n")
