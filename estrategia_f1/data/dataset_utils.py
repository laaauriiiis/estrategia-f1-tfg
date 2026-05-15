"""
dataset_utils.py

Funciones auxiliares para la construcción y validación
del dataset experimental.

Este módulo contiene la lógica necesaria para:
- Calcular estadísticas robustas sobre datos numéricos, como medias y medianas ignorando valores no válidos.
- Normalizar tipos de datos utilizados durante la construcción del dataset, como fechas y columnas numéricas.
- Aplicar operaciones auxiliares de validación, filtrado y asignación sobre DataFrames.
"""

# IMPORTS
import numpy as np
import pandas as pd

def mediana_segura(x) -> float:
    """
    Calcula la mediana de una colección de valores
    ignorando entradas no numéricas o ausentes.

    Parámetros
    ----------
    x : Any
        Colección de valores numéricos o convertibles
        a formato numérico.

    Returns
    -------
    float
        Mediana de los valores válidos.

        Si no existen valores numéricos válidos,
        devuelve np.nan.
    """
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    return float(x.median()) if len(x) else np.nan

def media_segura(x) -> float:
    """
    Calcula la media de una colección de valores
    ignorando entradas no numéricas o ausentes.

    Parámetros
    ----------
    x : Any
        Colección de valores numéricos o convertibles
        a formato numérico.

    Returns
    -------
    float
        Media aritmética de los valores válidos.

        Si no existen valores numéricos válidos,
        devuelve np.nan.
    """
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    return float(x.mean()) if len(x) else np.nan

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