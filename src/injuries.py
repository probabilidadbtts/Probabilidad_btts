"""Modelo de análisis de bajas y su impacto en las predicciones BTTS."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Categorías de jugadores por impacto en BTTS
CRITICAL_POSITIONS = {"STRIKER", "FORWARD", "CENTRE-FORWARD", "CF", "ST"}
HIGH_IMPACT_POSITIONS = {"ATTACKING MIDFIELD", "LEFT WINGER", "RIGHT WINGER", "AM", "LW", "RW"}
MID_IMPACT_POSITIONS = {"MIDFIELD", "CENTRE MIDFIELD", "CM", "CAM"}
DEFENCE_POSITIONS = {"DEFENDER", "LEFT-BACK", "RIGHT-BACK", "CENTRE-BACK", "CB", "LB", "RB"}
GOALKEEPER_POSITIONS = {"GOALKEEPER", "GK"}


class InjuryImpactAnalyzer:
    """Analiza el impacto de bajas en predicciones BTTS."""

    @staticmethod
    def _normalize_position(position: str) -> str:
        return (position or "").strip().upper()

    @classmethod
    def _categorize_player_impact(cls, position: str) -> float:
        """Devuelve un multiplicador de impacto según la posición (0.0 a 1.0)."""
        norm = cls._normalize_position(position)

        if any(token in norm for token in CRITICAL_POSITIONS):
            return 0.90
        if any(token in norm for token in HIGH_IMPACT_POSITIONS):
            return 0.75
        if any(token in norm for token in MID_IMPACT_POSITIONS):
            return 0.50
        if any(token in norm for token in DEFENCE_POSITIONS):
            return 0.40
        if any(token in norm for token in GOALKEEPER_POSITIONS):
            return 0.70
        return 0.30

    def analyze_injuries(
        self, home_id: str, away_id: str, client: Any
    ) -> Dict[str, Any]:
        """Obtiene y analiza bajas del equipo."""
        try:
            home_injuries = self._fetch_team_injuries(home_id, client)
            away_injuries = self._fetch_team_injuries(away_id, client)

            home_impact = self._calculate_injury_impact(home_injuries)
            away_impact = self._calculate_injury_impact(away_injuries)

            return {
                "home_injuries": home_injuries,
                "away_injuries": away_injuries,
                "home_injury_impact": home_impact,
                "away_injury_impact": away_impact,
                "combined_injury_severity": home_impact + away_impact,
            }
        except Exception as exc:
            logger.warning("Error al obtener bajas: %s", exc)
            return {
                "home_injuries": [],
                "away_injuries": [],
                "home_injury_impact": 0.0,
                "away_injury_impact": 0.0,
                "combined_injury_severity": 0.0,
            }

    @staticmethod
    def _fetch_team_injuries(team_id: str, client: Any) -> List[Dict[str, Any]]:
        """Obtiene la lista de bajas de un equipo."""
        try:
            data = client.get(f"/football/teams/{team_id}/injuries")
            return data.get("data", [])
        except Exception as exc:
            logger.debug("No se pudieron obtener bajas para equipo %s: %s", team_id, exc)
            return []

    def _calculate_injury_impact(self, injuries: List[Dict[str, Any]]) -> float:
        """Calcula el impacto total de bajas (0.0 a 1.0)."""
        if not injuries:
            return 0.0

        total_impact = 0.0
        num_relevant = 0

        for injury in injuries:
            player = injury.get("player") or {}
            position = player.get("position") or injury.get("position") or ""
            impact = self._categorize_player_impact(position)

            if impact > 0.3:
                total_impact += impact
                num_relevant += 1

        if num_relevant == 0:
            return 0.0

        avg_impact = total_impact / num_relevant
        return min(avg_impact, 0.80)

    @staticmethod
    def adjust_ensemble_by_injuries(
        ensemble_score: float,
        home_injury_impact: float,
        away_injury_impact: float,
    ) -> float:
        """Ajusta el score del ensemble según bajas."""
        if home_injury_impact == 0.0 and away_injury_impact == 0.0:
            return ensemble_score

        if home_injury_impact > 0.5:
            ensemble_score *= 1.0 - (home_injury_impact * 0.20)

        if away_injury_impact > 0.5:
            ensemble_score *= 1.0 - (away_injury_impact * 0.20)

        return round(max(0.0, min(100.0, ensemble_score)), 2)
