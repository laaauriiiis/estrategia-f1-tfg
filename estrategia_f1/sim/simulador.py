"""
simulador.py
TODO
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from estrategia_f1.config import (
    COMPUESTOS,
    DEFAULT_PIT_LOSS,
    PENALIZACION_VIDA_UTIL,
    PENALIZACION_STINT,
    TEMP_MAP,
    EXTRA_PARADA_MULTIPLE,
)

from estrategia_f1.acciones import (
    obtener_parametros_compuesto,
    multiplicador_desgaste,
    normalizar_estrategia,
    compuestos_disponibles,
)

# Cálculo de tiempos----------------------------------------------------------------------------------------------------
def tiempo_stint(vueltas: int, ritmo: float, degradacion: float, vida: float, *, mult_deg: float = 1.0) -> float:
    """
    Tiempo aproximado (s) de un stint de vueltas_stint vueltas. Si vueltas_stint supera la vida útil,
    penaliza degradación a partir de 'vida' con el factor penalizacion_vida_util.
    """
    vueltas_stint = int(vueltas)
    if vueltas_stint <= 0:
        return 0.0

    vida_i = int(np.floor(float(vida)))

    factor_degradacion = float(degradacion) * float(mult_deg)
    base = (vueltas_stint * float(ritmo)) + (factor_degradacion * (vueltas_stint * (vueltas_stint + 1) / 2.0))

    if vueltas_stint <= vida_i:
        return float(base)

    d_extra = factor_degradacion * (float(PENALIZACION_VIDA_UTIL) - 1.0)
    extra = d_extra * ((vueltas_stint * (vueltas_stint + 1) - vida_i * (vida_i + 1)) / 2.0)
    return float(base + extra)


def obtener_longitudes_stints(n_vueltas: int, estrategia: list[str], fila: pd.Series, *, mult_deg: float = 1.0, ) -> list[int] | None:
    """
    Optimiza la partición de n_vueltas en k = len(estrategia) enteros positivos
    para minimizar el tiempo simulado, manteniendo fijo el orden de compuestos.

    Devuelve [vueltas_primer_stint, vueltas_segundo_stint, ...] o None si no se puede simular.
    """
    n_vueltas = int(n_vueltas)
    k = len(estrategia)

    if not (2 <= k <= 4):
        raise ValueError("La estrategia debe tener entre 2 y 4 stints")

    if n_vueltas < k:
        return None

    try:
        parametros = [obtener_parametros_compuesto(fila, comp) for comp in estrategia]
    except Exception:
        return None

    def coste_total(particion: list[int]) -> float:
        coste = 0.0
        for vueltas_stint, (ritmo, deg, vida) in zip(particion, parametros):
            coste += tiempo_stint(vueltas_stint, ritmo, deg, vida, mult_deg=mult_deg)
        return float(coste)

    mejor_coste = np.inf
    mejor_particion: list[int] | None = None

    if k == 2:
        for vueltas_primer_stint in range(1, n_vueltas):
            particion = [vueltas_primer_stint, n_vueltas - vueltas_primer_stint]
            c = coste_total(particion)
            if c < mejor_coste:
                mejor_coste, mejor_particion = c, particion

    elif k == 3:
        for vueltas_primer_stint in range(1, n_vueltas - 1):
            for vueltas_segundo_stint in range(1, n_vueltas - vueltas_primer_stint):
                vueltas_tercer_stint = n_vueltas - vueltas_primer_stint - vueltas_segundo_stint
                if vueltas_tercer_stint < 1:
                    continue
                particion = [vueltas_primer_stint, vueltas_segundo_stint, vueltas_tercer_stint]
                c = coste_total(particion)
                if c < mejor_coste:
                    mejor_coste, mejor_particion = c, particion

    else:  # k == 4
        for vueltas_primer_stint in range(1, n_vueltas - 2):
            for vueltas_segundo_stint in range(1, n_vueltas - vueltas_primer_stint - 1):
                for vueltas_tercer_stint in range(1, n_vueltas - vueltas_primer_stint - vueltas_segundo_stint):
                    vueltas_cuarto_stint = n_vueltas - vueltas_primer_stint - vueltas_segundo_stint - vueltas_tercer_stint
                    if vueltas_cuarto_stint < 1:
                        continue
                    particion = [vueltas_primer_stint, vueltas_segundo_stint, vueltas_tercer_stint, vueltas_cuarto_stint]
                    c = coste_total(particion)
                    if c < mejor_coste:
                        mejor_coste, mejor_particion = c, particion

    if mejor_particion is None:
        base = [n_vueltas // k] * k
        base[-1] += n_vueltas - sum(base)
        mejor_particion = base

    return mejor_particion


# Simulación de la carrera----------------------------------------------------------------------------------------------
def simular_tiempo_carrera(fila: pd.Series, estrategia_compuestos) -> float:
    """
    Devuelve el tiempo total simulado para una estrategia.
    Si no es simulable por falta de datos/compuestos no disponibles, devuelve np.nan.
    """
    estrategia = normalizar_estrategia(estrategia_compuestos)
    if estrategia is None:
        raise ValueError(f"Estrategia inválida: {estrategia_compuestos}")

    if any(c not in COMPUESTOS for c in estrategia):
        raise ValueError(f"Estrategia con compuesto inválido: {estrategia}")

    n_vueltas = pd.to_numeric(fila.get("n_laps", np.nan), errors="coerce")
    if not np.isfinite(n_vueltas) or int(n_vueltas) <= 0:
        return np.nan
    n_vueltas = int(n_vueltas)

    pit_loss = pd.to_numeric(fila.get("pit_loss_s", np.nan), errors="coerce")
    pit_loss = float(pit_loss) if np.isfinite(pit_loss) else float(DEFAULT_PIT_LOSS)

    # Degradación multiplicada por desgaste y temperatura (si existe)
    mult_deg = float(multiplicador_desgaste(fila.get("wear_index", None)))
    temp_cat = fila.get("track_temp_cat", None)
    mult_deg *= TEMP_MAP.get(temp_cat, 1.0)

    disponibles = compuestos_disponibles(fila)
    if not all(c in disponibles for c in estrategia):
        return np.nan

    longitudes = obtener_longitudes_stints(n_vueltas=n_vueltas, estrategia=estrategia, fila=fila, mult_deg=mult_deg)
    if longitudes is None:
        return np.nan

    total = 0.0
    for comp, L in zip(estrategia, longitudes):
        try:
            ritmo, deg, vida = obtener_parametros_compuesto(fila, comp)
        except Exception:
            return np.nan

        total += tiempo_stint(vueltas=int(L), ritmo=ritmo, degradacion=deg, vida=vida, mult_deg=mult_deg)

    n_paradas = len(estrategia) - 1
    total += n_paradas * pit_loss
    total += n_paradas * float(PENALIZACION_STINT)
    total += max(0, n_paradas - 1) * float(EXTRA_PARADA_MULTIPLE)
    return float(total)