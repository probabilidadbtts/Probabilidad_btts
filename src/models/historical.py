"""Modelo histórico simple (últimos N partidos)."""

from __future__ import annotations

from typing import Any, Dict, List

from ..utils import is_btts
from .base import BaseBTTSModel


class HistoricalBTTS(BaseBTTSModel):
    name = "historical"

    def __init__(self, n: int = 10) -> None:
        self.n = n

    def predict(
        self,
        home_matches: List[Dict[str, Any]],
        away_matches: List[Dict[str, Any]],
    ) -> float:
        home = home_matches[: self.n]
        away = away_matches[: self.n]
        if not home or not away:
            return 0.0

        home_rate = sum(1 for m in home if is_btts(m)) / len(home)
        away_rate = sum(1 for m in away if is_btts(m)) / len(away)
        return round(((home_rate + away_rate) / 2) * 100, 2)
