"""Clase base para todos los modelos de BTTS."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseBTTSModel(ABC):
    name: str = "base"

    @abstractmethod
    def predict(
        self,
        home_matches: List[Dict[str, Any]],
        away_matches: List[Dict[str, Any]],
    ) -> float:
        """
        Devuelve probabilidad de BTTS en escala 0-100.
        """
        ...
