"""Listado final de ligas de prioridad para el predictor.

Solo se mantienen competiciones de liga, sin copas ni torneos no permanentes.
Incluye ligas conocidas de Europa, América y Asia, con MLS y Liga MX.
"""

from __future__ import annotations

import logging
from typing import Dict, Set

logger = logging.getLogger(__name__)

TOP_20_LEAGUES: Dict[int, str] = {
    39: "Premier League",
    140: "La Liga",
    135: "Serie A",
    78: "Bundesliga",
    61: "Ligue 1",
    203: "Primeira Liga",
    94: "Eredivisie",
    262: "Liga MX",
    848: "MLS",
    179: "Super Lig",
    766: "Saudi Pro League",
    251: "Chinese Super League",
    307: "Süper Lig",
    235: "Ukrainian Premier League",
    44: "Liga Profesional Argentina",
    9: "Brasileirão",
    201: "A-League",
    24: "Belgian Pro League",
    31: "Scottish Premiership",
    72: "Czech First League",
}

LEAGUE_ALIASES: Dict[int, int] = {
    307: 179,
    94: 94,
}


class LeagueFilter:
    """Filtro para mantener solo ligas de primer nivel."""

    TOP_LEAGUE_IDS: Set[int] = set(TOP_20_LEAGUES.keys())

    @classmethod
    def is_top_league(cls, competition_id: int | None) -> bool:
        if competition_id is None:
            return False
        return competition_id in cls.TOP_LEAGUE_IDS

    @classmethod
    def get_league_name(cls, competition_id: int | None) -> str:
        if competition_id is None:
            return "Unknown"
        return TOP_20_LEAGUES.get(competition_id, "Unknown")

    @classmethod
    def filter_matches(cls, matches: list[dict]) -> list[dict]:
        filtered = [m for m in matches if cls.is_top_league(m.get("competition_id"))]
        logger.info(
            "Filtrados %d/%d partidos → ligas de prioridad",
            len(filtered),
            len(matches),
        )
        return filtered
