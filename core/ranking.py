"""Anzeige-Logik für die Rangliste: Prozentwerte unterhalb von COS_FLOOR nicht als hartes „0 %“ zeigen."""

from __future__ import annotations

from config import settings
from core.scoring import ProgramScore

LOW_MATCH_LABEL = "Geringe Übereinstimmung"
MIN_BAR_PERCENT = 2.0   # Mindestlänge des Balkens: zeigt, dass gerechnet wurde, auch wenn nichts passt


def percent_label(score: ProgramScore) -> str:
    """„73 %“ – oder „< 1 %“, wenn der Wert auf der Prozentskala praktisch bei 0 liegt."""
    return "< 1 %" if score.percent < 1.0 else f"{score.percent:.0f} %"


def bar_percent(score: ProgramScore) -> float:
    return max(score.percent, MIN_BAR_PERCENT)


def low_match_note(score: ProgramScore) -> str | None:
    """Erklärt, warum der Wert klein ist; nur für Studiengänge unter der Kosinus-Untergrenze."""
    if not score.below_floor:
        return None
    return (f"{LOW_MATCH_LABEL}: Kosinus {score.cosine:.2f} liegt unter der Untergrenze "
            f"COS_FLOOR = {settings.cos_floor:.2f}, die Skala beginnt hier bei 0 %.")
