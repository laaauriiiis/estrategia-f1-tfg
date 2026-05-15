"""
dataset_categorias.py

Discretización de variables continuas y observadas del dataset.

Este módulo contiene la lógica necesaria para:
- Convertir variables meteorológicas en categorías interpretables, como temperatura de pista baja, media o alta.
- Transformar valores de lluvia observada o probabilidad histórica de lluvia en variables categóricas.
- Clasificar el desgaste histórico del circuito a partir de umbrales fijos definidos en la configuración.
"""

# IMPORTS
import numpy as np
import pandas as pd
from estrategia_f1.config import (
    WEAR_THRESHOLDS,
    RAIN_THRESHOLDS
)
from estrategia_f1.data.dataset_utils import es_finito

# CATEGORIZACIÓN -------------------------------------------------------------------------------------------------------
def categoria_temp_pista(temp_pista_c):
    """
     Discretiza la temperatura observada de pista en
     categorías ordinales.

     Parámetros
     ----------
     temp_pista_c : float
         Temperatura de pista observada, expresada
         en grados Celsius.

     Returns
     -------
     str | float
         Categoría cualitativa de temperatura:
         - "baja"
         - "media"
         - "alta"

         Si el valor no es válido, devuelve np.nan.
     """
    if pd.isna(temp_pista_c):
        return np.nan
    if temp_pista_c < 25:
        return "baja"
    elif temp_pista_c <= 40:
        return "media"
    else:
        return "alta"

def condicion_meteo_desde_lluvia(valor_lluvia):
    """
    Convierte un valor observado de lluvia en una
    categoría meteorológica binaria.

    Parámetros
    ----------
    valor_lluvia : float
        Nivel observado de lluvia asociado a una
        carrera.

    Returns
    -------
    str | float
        Condición meteorológica cualitativa:

        - "lluvia" si el valor es mayor que cero.
        - "seco" en caso contrario.

        Si el valor no es válido, devuelve np.nan.
    """
    if pd.isna(valor_lluvia):
        return np.nan
    return "lluvia" if valor_lluvia > 0 else "seco"

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