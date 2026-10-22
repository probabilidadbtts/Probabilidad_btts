# src/injuries.py
"""Modelo de análisis de bajas y su impacto en las predicciones BTTS."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

from .lineups import LineupAnalyzer

logger = logging.getLogger(__name__)


class InjuryImpactAnalyzer:
    """Analiza el impacto de bajas usando la alineación confirmada cuando existe."""

    def analyze_injuries(
        self,
        home_id: str,
        away_id: str,
        client: Any,
        match_id: Optional[int | str] = None,
        home_name: Optional[str] = None,
        away_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Obtiene y analiza bajas reales desde lineups confirmados.

        Hace UN solo fetch de lineups por partido y lo comparte entre
        local y visitante. Fallback seguro si no hay lineups confirmados.
        """
        empty = {
            "home_injuries": [],
            "away_injuries": [],
            "home_injury_impact": 0.0,
            "away_injury_impact": 0.0,
            "combined_injury_severity": 0.0,
        }
        try:
            if match_id is None:
                return empty

            raw = LineupAnalyzer.fetch_lineups(client, match_id)
            if not raw:
                return empty

            home_l = LineupAnalyzer.analyze_match_lineups(
                client, match_id, home_name, raw_payload=raw
            )
            away_l = LineupAnalyzer.analyze_match_lineups(
                client, match_id, away_name, raw_payload=raw
            )

            return {
                "home_injuries": home_l["absences"],
                "away_injuries": away_l["absences"],
                "home_injury_impact": home_l["impact"],
                "away_injury_impact": away_l["impact"],
                "combined_injury_severity": round(
                    min(home_l["impact"] + away_l["impact"], 0.90), 2
                ),
            }
        except Exception as exc:
            logger.debug("Error al obtener bajas: %s", exc)
            return empty

    @staticmethod
    def adjust_ensemble_by_injuries(
        ensemble_score: float,
        home_injury_impact: float,
        away_injury_impact: float,
    ) -> float:
        """Ajusta el score del ensemble según bajas confirmadas."""
        if home_injury_impact == 0.0 and away_injury_impact == 0.0:
            return ensemble_score

        if home_injury_impact > 0.5:
            ensemble_score *= 1.0 - (home_injury_impact * 0.20)
        if away_injury_impact > 0.5:
            ensemble_score *= 1.0 - (away_injury_impact * 0.20)

        return round(max(0.0, min(100.0, ensemble_score)), 2)
