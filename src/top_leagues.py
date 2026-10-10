"""Listado de ligas de prioridad para el filtro del proyecto.

Este repositorio solo debe quedarse en las ligas de mayor peso y en el rango
fecha solicitado (hoy y siguientes). No se debe incluir competiciones no
ligas como Mundiales, Copas continentales o torneos no permanentes.
"""

from __future__ import annotations

import logging
from typing import Dict, Set

logger = logging.getLogger(__name__)

# Lista de ligas de prioridad a mantener en el rango actual.
# Se concentra en competiciones de liga, incluyendo MLS y Liga MX.
# Si la API cambia IDs o se añaden competiciones nuevas, se ajusta aquí.
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
    2: "Champions League",
    3: "Europa League",
    44: "Liga Profesional Argentina",
    9: "Brasileirão",
    93: "Eredivisie (alt)",
    201: "A-League",
}

# IDs alternativos / sinónimos encontrados
LEAGUE_ALIASES: Dict[int, int] = {
    307: 179,
    93: 94,
}


class LeagueFilter:
    """Filtro para mantener solo las ligas de prioridad del rango actual."""

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
        """Filtra partidos manteniendo solo las competiciones de liga priorizadas."""
        filtered = [m for m in matches if cls.is_top_league(m.get("competition_id"))]
        logger.info(
            "Filtrados %d/%d partidos → ligas de prioridad del rango actual",
            len(filtered),
            len(matches),
        )
        return filtered
