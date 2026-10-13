"""Kosinus-Ähnlichkeit und lineare Skalierung auf Prozent – bewusst ohne Softmax."""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def normalize(vector: np.ndarray) -> np.ndarray:
    """Skaliert auf Länge 1; der Nullvektor bleibt der Nullvektor."""
    return vector / max(float(np.linalg.norm(vector)), _EPS)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), _EPS))


def cosine_to_percent(cos: float | np.ndarray, floor: float, ceil: float) -> float | np.ndarray:
    """Bildet ``[floor, ceil]`` linear auf ``[0, 100]`` ab und kappt außerhalb.

    Jeder Wert wird einzeln umgerechnet. Es gibt keine Normierung über mehrere Studiengänge,
    die Summe der Prozentwerte ist daher beliebig.
    """
    if ceil <= floor:
        raise ValueError(f"cos_ceil ({ceil}) muss größer sein als cos_floor ({floor})")
    return np.clip((cos - floor) / (ceil - floor), 0.0, 1.0) * 100.0
