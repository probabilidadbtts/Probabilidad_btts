"""Modelo Poisson Bivariada clásico para BTTS."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import poisson

from ..utils import extract_score
from .base import BaseBTTSModel


class BivariatePoisson(BaseBTTSModel):
    name = "bivariate_poisson"

    def __init__(self, n: int = 20) -> None:
        self.n = n

    def _team_goal_stats(
        self, matches: List[Dict[str, Any]], team_id: str
    ) -> Tuple[float, float]:
        """Promedio de goles anotados y recibidos por el equipo."""
        scored, conceded = [], []
        for m in matches[: self.n]:
            h, a = extract_score(m)
            if h is None or a is None:
                continue
            home_id = (m.get("home_team") or {}).get("id")
            if home_id == team_id:
                scored.append(h)
                conceded.append(a)
            else:
                scored.append(a)
                conceded.append(h)
        if not scored:
            return 1.2, 1.2  # prior suave
        return float(np.mean(scored)), float(np.mean(conceded))

    def predict(
        self,
        home_matches: List[Dict[str, Any]],
        away_matches: List[Dict[str, Any]],
    ) -> float:
        if not home_matches or not away_matches:
            return 0.0

        home_id = (home_matches[0].get("home_team") or home_matches[0].get("away_team") or {}).get("id")
        away_id = (away_matches[0].get("home_team") or away_matches[0].get("away_team") or {}).get("id")

        # λ_home = goles anotados local * goles recibidos visitante / 2 (ajuste simple)
        home_scored, home_conc = self._team_goal_stats(home_matches, home_id or "")
        away_scored, away_conc = self._team_goal_stats(away_matches, away_id or "")

        lambda_home = (home_scored + away_conc) / 2
        lambda_away = (away_scored + home_conc) / 2
        # dependencia débil (covarianza)
        lambda_cov = 0.15

        # P(X>0, Y>0) = 1 - P(X=0) - P(Y=0) + P(X=0,Y=0)
        # usando Poisson independiente + ajuste de covarianza simple
        p_x0 = poisson.pmf(0, lambda_home + lambda_cov)
        p_y0 = poisson.pmf(0, lambda_away + lambda_cov)
        p_00 = poisson.pmf(0, lambda_home + lambda_away + lambda_cov)

        p_btts = 1.0 - p_x0 - p_y0 + p_00
        return round(max(0.0, min(100.0, p_btts * 100)), 2)
