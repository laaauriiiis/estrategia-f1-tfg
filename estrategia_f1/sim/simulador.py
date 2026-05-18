"""
simulador.py

Simulación del tiempo total de carrera para estrategias de neumáticos.

Este módulo contiene la lógica necesaria para:
- Estimar el tiempo de cada stint en función del ritmo,
  la degradación y la vida útil del compuesto.
- Optimizar la distribución de vueltas entre stints
  manteniendo fijo el orden de compuestos.
- Incorporar factores contextuales como el desgaste,
  la temperatura de pista y el coste de las paradas.
- Simular el tiempo total de carrera asociado a una
  estrategia completa de neumáticos.
"""

# IMPORTS
from __future__ import annotations
import numpy as np
import pandas as pd
import estrategia_f1.config as cfg
from estrategia_f1.acciones import (
    obtener_parametros_compuesto,
    multiplicador_desgaste,
    normalizar_estrategia,
    compuestos_disponibles,
)

# CÁLCULO DE STINTS ----------------------------------------------------------------------------------------------------
def tiempo_stint(vueltas: int, ritmo: float, degradacion: float, vida: float, *, mult_deg: float = 1.0) -> float:
    """
    Calcula el tiempo simulado de un stint completo.

    El tiempo se estima a partir del ritmo base del compuesto,
    su degradación acumulada por vuelta y su vida útil estimada.
    Si el stint supera dicha vida útil, se aplica una penalización
    adicional sobre la degradación.

    Parámetros
    ----------
    vueltas : int
        Número de vueltas del stint.
    ritmo : float
        Tiempo base por vuelta del compuesto, en segundos.
    degradacion : float
        Incremento esperado del tiempo por vuelta debido
        al desgaste del neumático.
    vida : float
        Vida útil estimada del compuesto, expresada en vueltas.
    mult_deg : float, optional
        Multiplicador externo aplicado sobre la degradación
        para modelar factores contextuales como temperatura
        o desgaste del circuito.

    Returns
    -------
    float
        Tiempo total simulado del stint, en segundos.
    """
    vueltas_stint = int(vueltas)
    if vueltas_stint <= 0:
        return 0.0

    vida_i = int(np.floor(float(vida)))

    factor_degradacion = float(degradacion) * float(mult_deg)
    base = (vueltas_stint * float(ritmo)) + (factor_degradacion * (vueltas_stint * (vueltas_stint + 1) / 2.0))

    if vueltas_stint <= vida_i:
        return float(base)

    d_extra = factor_degradacion * (float(cfg.PENALIZACION_VIDA_UTIL) - 1.0)
    extra = d_extra * ((vueltas_stint * (vueltas_stint + 1) - vida_i * (vida_i + 1)) / 2.0)
    return float(base + extra)

def obtener_longitudes_stints(n_vueltas: int, estrategia: list[str], fila: pd.Series, *, mult_deg: float = 1.0, ) -> list[int] | None:
    """
    Calcula la mejor distribución de vueltas entre los stints.

    La función mantiene fijo el orden de compuestos de la estrategia
    y prueba distintas particiones enteras del total de vueltas para
    seleccionar aquella que minimiza el tiempo simulado de carrera.

    Parámetros
    ----------
    n_vueltas : int
        Número total de vueltas de la carrera.
    estrategia : list[str]
        Secuencia ordenada de compuestos que forman la estrategia.
    fila : pd.Series
        Observación piloto-carrera con los parámetros necesarios
        para simular cada compuesto.
    mult_deg : float, optional
        Multiplicador aplicado sobre la degradación de los neumáticos.

    Returns
    -------
    list[int] | None
        Lista con el número de vueltas asignadas a cada stint.
        Devuelve None si la estrategia no puede simularse.
    """
    n_vueltas = int(n_vueltas)
    k = len(estrategia)

    # El simulador solo contempla estrategias de 2 a 4 stints
    if not (2 <= k <= 4):
        raise ValueError("La estrategia debe tener entre 2 y 4 stints.")

    # Cada stint debe recibir al menos una vuelta
    if n_vueltas < k:
        return None

    # Se recuperan una sola vez los parámetros de cada compuesto para evitar recalcularlos durante la búsqueda de particiones
    try:
        parametros = [obtener_parametros_compuesto(fila, comp) for comp in estrategia]
    except Exception:
        return None

    def coste_total(particion: list[int]) -> float:
        # El coste de una partición es la suma del tiempo simulado de cada stint con su compuesto correspondiente
        coste = 0.0
        for vueltas_stint, (ritmo, deg, vida) in zip(particion, parametros):
            coste += tiempo_stint(vueltas_stint, ritmo, deg, vida, mult_deg=mult_deg)
        return float(coste)

    mejor_coste = np.inf
    mejor_particion: list[int] | None = None

    # Búsqueda exhaustiva de particiones enteras manteniendo fijo el orden de compuestos de la estrategia
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

    # Si no se encuentra ninguna partición válida, se utiliza una distribución uniforme como fallback conservador
    if mejor_particion is None:
        base = [n_vueltas // k] * k
        base[-1] += n_vueltas - sum(base)
        mejor_particion = base

    return mejor_particion


# SIMULACIÓN DE LA CARRERA ---------------------------------------------------------------------------------------------
def simular_tiempo_carrera(fila: pd.Series, estrategia_compuestos) -> float:
    """
    Simula el tiempo total de carrera para una estrategia de neumáticos.

    La función valida la estrategia recibida, comprueba que los compuestos
    estén disponibles para la observación evaluada, calcula la mejor
    distribución de vueltas entre stints y suma el tiempo estimado de
    cada stint junto con las penalizaciones asociadas a las paradas.

    Parámetros
    ----------
    fila : pd.Series
        Observación piloto-carrera con las variables necesarias para
        la simulación.
    estrategia_compuestos : Any
        Estrategia de neumáticos a evaluar, representada como una
        secuencia de compuestos.

    Returns
    -------
    float
        Tiempo total simulado de carrera, en segundos.
        Devuelve np.nan si la estrategia no puede simularse.
    """

    # Se normaliza la estrategia para trabajar con un formato homogéneo
    estrategia = normalizar_estrategia(estrategia_compuestos)
    if estrategia is None:
        raise ValueError(f"Estrategia inválida: {estrategia_compuestos}")

    # Se valida que los compuestos estén dentro del espacio considerado por el simulador
    if any(c not in cfg.COMPUESTOS for c in estrategia):
        raise ValueError(f"Estrategia con compuesto inválido: {estrategia}")

    # Se recupera el número total de vueltas de la carrera
    n_vueltas = pd.to_numeric(fila.get("n_laps", np.nan), errors="coerce")
    if not np.isfinite(n_vueltas) or int(n_vueltas) <= 0:
        return np.nan
    n_vueltas = int(n_vueltas)

    # Si no existe pit loss histórico válido, se utiliza un valor por defecto
    pit_loss = pd.to_numeric(fila.get("pit_loss_s", np.nan), errors="coerce")
    pit_loss = float(pit_loss) if np.isfinite(pit_loss) else float(cfg.DEFAULT_PIT_LOSS)

    # La degradación se ajusta según desgaste del circuito y temperatura de pista
    mult_deg = float(multiplicador_desgaste(fila.get("wear_index", None)))
    temp_cat = fila.get("track_temp_cat", None)
    mult_deg *= cfg.TEMP_MAP.get(temp_cat, 1.0)

    disponibles = compuestos_disponibles(fila)
    if not all(c in disponibles for c in estrategia):
        return np.nan

    # Se obtienen las duraciones de cada stint
    longitudes = obtener_longitudes_stints(n_vueltas=n_vueltas, estrategia=estrategia, fila=fila, mult_deg=mult_deg)
    if longitudes is None:
        return np.nan

    # Suma del tiempo simulado de cada stint con su compuesto correspondiente
    total = 0.0
    for comp, L in zip(estrategia, longitudes):
        try:
            ritmo, deg, vida = obtener_parametros_compuesto(fila, comp)
        except Exception:
            return np.nan

        total += tiempo_stint(vueltas=int(L), ritmo=ritmo, degradacion=deg, vida=vida, mult_deg=mult_deg)

    # Penalización fija asociada al número de paradas realizadas
    n_paradas = len(estrategia) - 1
    total += n_paradas * pit_loss

    # Penalizaciones adicionales para evitar estrategias artificialmente extremas
    total += n_paradas * float(cfg.PENALIZACION_STINT)
    total += max(0, n_paradas - 1) * float(cfg.EXTRA_PARADA_MULTIPLE)

    return float(total)