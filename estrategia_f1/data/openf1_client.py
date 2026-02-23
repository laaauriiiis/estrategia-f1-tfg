"""
openf1_client.py
TODO
"""

from __future__ import annotations

import time
import requests
import pandas as pd

from estrategia_f1.config import BASE_API

_sesion_http = requests.Session()


def openf1_descargar(endpoint: str, params: dict, pausa: float = 0.15, reintentos: int = 4) -> pd.DataFrame:
    """
    Llama a un endpoint de OpenF1 y devuelve la respuesta como DataFrame.
    """
    url = f"{BASE_API}/{endpoint}"
    ultimo_error = None

    for k in range(reintentos):
        try:
            r = _sesion_http.get(url, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            time.sleep(pausa)
            return pd.DataFrame(data)
        except Exception as e:
            ultimo_error = e
            time.sleep(0.8 * (k + 1))

    raise RuntimeError(f"Error OpenF1 endpoint={endpoint} params={params} error={ultimo_error}")

