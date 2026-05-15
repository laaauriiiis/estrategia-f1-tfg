"""
openf1_client.py

Cliente de acceso a la API de OpenF1.

Este módulo contiene la lógica necesaria para:
- Realizar consultas HTTP a los distintos endpoints de OpenF1.
- Gestionar pausas entre peticiones para reducir errores por limitación de tasa.
- Aplicar reintentos automáticos ante errores temporales de red o respuestas no válidas.
- Convertir las respuestas JSON de OpenF1 en estructuras tabulares de pandas.

Este cliente centraliza el acceso a datos externos durante
la construcción del dataset experimental y la extracción de
variables observadas del sistema de recomendación de
estrategias de neumáticos para Fórmula 1.
"""

# IMPORTS
from __future__ import annotations
import time
import requests
import pandas as pd
from estrategia_f1.config import BASE_API

_sesion_http = requests.Session()

def openf1_descargar(endpoint: str, params: dict, pausa: float = 0.15, reintentos: int = 4) -> pd.DataFrame:
    """
    Descarga información desde un endpoint de OpenF1
    y devuelve la respuesta en formato tabular.

    Parámetros
    ----------
    endpoint : str
        Nombre del endpoint de OpenF1 que se desea
        consultar.
    params : dict
        Diccionario con los parámetros de consulta
        enviados a la API.
    pausa : float, optional
        Tiempo de espera, en segundos, aplicado tras
        una petición exitosa para reducir la frecuencia
        de acceso a la API.
    reintentos : int, optional
        Número máximo de intentos permitidos ante
        errores temporales de red o respuesta.

    Returns
    -------
    pd.DataFrame
        DataFrame construido a partir de la respuesta
        JSON devuelta por OpenF1.
        Si la API devuelve una lista vacía, se obtiene
        un DataFrame vacío.
    """
    # Se reutiliza una sesión HTTP persistente para reducir overhead en múltiples consultas
    url = f"{BASE_API}/{endpoint}"
    ultimo_error = None

    for k in range(reintentos):
        try:
            r = _sesion_http.get(url, params=params, timeout=60)
            r.raise_for_status()
            # Se aplica una pequeña pausa tras cada petición válida para reducir errores 429
            data = r.json()
            time.sleep(pausa)
            return pd.DataFrame(data)
        except Exception as e:
            ultimo_error = e
            # Backoff incremental ante errores temporales de red o limitación de tasa
            time.sleep(0.8 * (k + 1))

    raise RuntimeError(f"Error OpenF1 endpoint={endpoint} params={params} error={ultimo_error}")

