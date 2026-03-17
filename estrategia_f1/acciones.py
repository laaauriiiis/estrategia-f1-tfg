"""
acciones.py
TODO + separar
"""

from __future__ import annotations

import ast
from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd

from estrategia_f1.config import (
    COMPUESTOS, MIN_STINTS, MAX_STINTS,
    WEAR_MAP,
    TEMP_REF, TEMP_SLOPE, TEMP_CLIP,
    DEG_MIN, DEG_MAX, TOPK, BASELINE_PRIORIDAD,
)

# Utils internos--------------------------------------------------------------------------------------------------------
def a_float_o_nan(x) -> float:
    """
    Convierte a float si puede, si no, devuelve np.nan.
    """
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def limpiar_compuestos(secuencia: Iterable) -> list[str]:
    """
    Normaliza una secuencia a lista de compuestos en mayúsculas.
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


# Validación y normalización de estrategias-----------------------------------------------------------------------------
def estrategia_valida(estrategia: list[str]) -> bool:
    """
    Comprueba si una estrategia cumple las restricciones del dominio.
    """
    estrategia = limpiar_compuestos(estrategia)

    if not (MIN_STINTS <= len(estrategia) <= MAX_STINTS):
        return False
    if any(c not in COMPUESTOS for c in estrategia):
        return False
    if len(set(estrategia)) < 2:
        return False

    return True


def normalizar_estrategia(x) -> list[str] | None:
    """
    Normaliza una estrategia que puede venir en distintos formatos:
    - list/tuple/np.ndarray
    - string tipo "['SOFT','HARD']"
    - NaN/None

    Devuelve list[str] si es válida; si no, None.
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


# Espacio de acciones (mapas)-------------------------------------------------------------------------------------------
def construir_mapa_acciones() -> dict[int, list[str]]:
    """
    Devuelve un mapa: accion_id -> estrategia (lista de compuestos).
    Genera todas las estrategias de MIN_STINTS - MAX_STINTS con repetición,
    obligando a usar al menos 2 compuestos distintos.
    """
    mapa: dict[int, list[str]] = {}
    accion_id = 0

    # Recorremos el número de stints permitidos
    for n_stints in range(MIN_STINTS, MAX_STINTS + 1):
        # Todas las secuencias posibles con repetición (para mantener el orden)
        for secuencia in product(COMPUESTOS, repeat=n_stints):
            if len(set(secuencia)) < 2:
                continue
            mapa[accion_id] = list(secuencia)
            accion_id += 1

    return mapa


MAPA_ACCIONES = construir_mapa_acciones()

def construir_mapa_acciones_inverso(mapa_acciones: dict[int, list[str]]) -> dict[tuple[str, ...], int]:
    """
    Devuelve un mapa inverso: estrategia (lista de compuestos) -> accion_id.
    """
    mapa_inverso: dict[tuple[str, ...], int] = {}
    for accion_id, lista_compuestos in mapa_acciones.items():
        mapa_inverso[tuple(limpiar_compuestos(lista_compuestos))] = accion_id
    return mapa_inverso


def estrategia_desde_accion_id(accion_id: int, mapa_acciones: dict[int, list[str]]) -> list[str]:
    """
    Convierte un accion_id en su estrategia (lista de compuestos).
    """
    if accion_id not in mapa_acciones:
        raise KeyError(f"La accion_id={accion_id} no existe en mapa_acciones")

    estrategia = mapa_acciones[accion_id]
    if isinstance(estrategia, str):
        estrategia = ast.literal_eval(estrategia)

    if not isinstance(estrategia, (list, tuple)):
        raise TypeError(f"Estrategia inválida para accion_id={accion_id}: {estrategia}")

    # Por si a caso
    return limpiar_compuestos(estrategia)


def accion_id_desde_estrategia(estrategia: list[str], mapa_inverso: dict[tuple[str, ...], int]) -> int:
    """
    Devuelve el accion_id de una estrategia; si no existe, devuelve -1.
    """
    estrategia = limpiar_compuestos(estrategia)
    if not estrategia_valida(estrategia):
        return -1
    return mapa_inverso.get(tuple(estrategia), -1)


# Acciones válidas según el estado (fila)-------------------------------------------------------------------------------
def compuestos_disponibles(fila: pd.Series) -> set[str]:
    """
    Devuelve el conjunto de compuestos disponibles para una carrera.
    Un compuesto estará disponible si existen datos válidos de pace/deg/life.
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
    Devuelve las accion_id cuyas estrategias solo usan compuestos disponibles.
    """
    disponibles = compuestos_disponibles(fila)

    validas: list[int] = []
    for accion_id, lista_compuestos in mapa_acciones.items():
        lista_compuestos_limpia = limpiar_compuestos(lista_compuestos)
        if all(c in disponibles for c in lista_compuestos_limpia):
            validas.append(accion_id)

    return validas


# Helpers de simulación-------------------------------------------------------------------------------------------------
def multiplicador_desgaste(wear_categoria: str | None) -> float:
    """
    Multiplicador según categoría de desgaste.
    """
    if wear_categoria is None:
        return 1.0
    key = str(wear_categoria).strip().lower()
    return float(WEAR_MAP.get(key, 1.0))

def obtener_parametros_compuesto(fila: pd.Series, compuesto: str) -> tuple[float, float, float]:
    """
    Lee (pace, deg, life) del compuesto desde una fila. Lanza ValueError si falta.
    """
    c = str(compuesto).strip().lower()
    if c not in ("soft", "medium", "hard"):
        raise ValueError(f"Compuesto inválido: {compuesto}")

    pace = pd.to_numeric(fila.get(f"pace_{c}"), errors="coerce")
    deg  = pd.to_numeric(fila.get(f"deg_{c}"), errors="coerce")
    life = pd.to_numeric(fila.get(f"life_{c}"), errors="coerce")

    if not (np.isfinite(pace) and np.isfinite(deg) and np.isfinite(life)):
        raise ValueError(f"Parámetros no disponibles para {compuesto}: pace/deg/life inválidos")

    deg = float(np.clip(float(deg), float(DEG_MIN), float(DEG_MAX)))
    return float(pace), float(deg), float(life)

def elegir_estrategia_baseline(fila: pd.Series) -> list[str] | None:
    """
    Baseline = estrategia real observada (action_id) si es válida.
    Si falta / no existe / no es compatible, fallback a baseline por prioridad.
    """
    # 1) intentar baseline real
    action_id = fila.get("action_id", None)

    if action_id is not None and not pd.isna(action_id):
        try:
            action_id_int = int(action_id)
        except (TypeError, ValueError):
            action_id_int = None

        if action_id_int is not None:
            estrategia_real = MAPA_ACCIONES.get(action_id_int)
            if estrategia_real is not None:
                # 2) validar compuestos disponibles (si aplica en tu simulación)
                disponibles = compuestos_disponibles(fila)
                if disponibles and all(c in disponibles for c in estrategia_real):
                    return estrategia_real
                # si disponibles vacío o no compatible, cae al fallback

    # 3) fallback: prioridad como antes (para no tirar la fila)
    disponibles = compuestos_disponibles(fila)

    for estrategia in BASELINE_PRIORIDAD:
        if all(c in disponibles for c in estrategia):
            return estrategia

    disp = sorted(list(disponibles))
    if len(disp) >= 2:
        return disp[:2]
    if len(disp) == 1:
        return [disp[0], disp[0]]

    return None

# Helpers de entrenamiento----------------------------------------------------------------------------------------------
def construir_grupos(df: pd.DataFrame) -> np.ndarray:
    """
    Agrupa por race_id.
    """
    if "race_id" not in df.columns:
        raise KeyError("El dataframe no tiene la columna 'race_id' para agrupar.")
    return df["race_id"].astype(str).to_numpy()

def construir_estado_df(df: pd.DataFrame, *, columnas: list[str], columnas_excluir: list[str] | None = None,
        imputar_numericas: bool = True,) -> pd.DataFrame:
    """
    TODO
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
        X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan)

        if imputar_numericas:
            medianas = X[num_cols].median(numeric_only=True)
            X[num_cols] = X[num_cols].fillna(medianas)

    return X

# revisar !!
def columnas_numericas(df: pd.DataFrame) -> list[str]:
    """
    Devuelve las columnas numéricas del dataframe.
    """
    return list(df.select_dtypes(include=[np.number]).columns)


# Output----------------------------------------------------------------------------------------------------------------
def imprimir_resumen_evaluacion(resultados: pd.DataFrame) -> None:
    """
    Imprime un resumen de la evaluación.
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
    for k in TOPK:
        col_regret = f"regret@{k}"
        col_hit = f"hit@{k}"

        if col_regret in resultados.columns:
            print(f"Regret@{k} medio : {resultados[col_regret].mean():.3f} s")

        if col_hit in resultados.columns:
            print(f"Hit@{k}          : {resultados[col_hit].mean() * 100:.1f}%")

    print("\n===============================================================\n")
