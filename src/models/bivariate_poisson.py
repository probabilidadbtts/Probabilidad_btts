# src/models/bivariate_poisson.py
"""Modelo clásico de Poisson bivariada para BTTS."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from ..utils import extract_score, safe_div
from .base import BaseBTTSModel


def _poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * lam ** k / math.factorial(k)


class BivariatePoisson(BaseBTTSModel):
    name = "bivariate_poisson"

    def __init__(self, n: int = 20, max_goals: int = 6, rho: float = 0.05) -> None:
        self.n = n
        self.max_goals = max_goals
        self.rho = rho  # correlación ligera entre goles

    @staticmethod
    def _team_lambda(matches: List[Dict[str, Any]], team_id=None) -> Tuple[float, float]:
        """(lambda_ataque, lambda_defensa) estimados del equipo."""
        scored, conceded, played = [], [], 0
        for m in matches:
            h, a = extract_score(m)
            if h is None or a is None:
                continue
            home_id = (m.get("home_team") or {}).get("id")
            is_home = team_id is None or home_id == team_id
            gf, ga = (h, a) if is_home else (a, h)
            scored.append(gf)
            conceded.append(ga)
            played += 1
        if played == 0:
            return 1.2, 1.2
        return sum(scored) / played, sum(conceded) / played

    def predict(self, home_matches, away_matches) -> float:
        h_att, h_def = self._team_lambda(home_matches)
        a_att, a_def = self._team_lambda(away_matches)

        # Lambda esperados del partido
        lam_home = max(0.2, (h_att + a_def) / 2)
        lam_away = max(0.2, (a_att + h_def) / 2)

        # P(BTTS) = 1 - P(home=0) - P(away=0) + P(0-0)
        p_h0 = _poisson_pmf(0, lam_home)
        p_a0 = _poisson_pmf(0, lam_away)
        p_00 = _poisson_pmf(0, lam_home + lam_away - self.rho * min(lam_home, lam_away))

        btts = 1.0 - p_h0 - p_a0 + p_00
        return round(max(0.0, min(100.0, btts * 100)), 2)
