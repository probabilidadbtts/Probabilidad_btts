"""Configuración de las 20 ligas principales incluidas MLS y Liga MX."""

from __future__ import annotations

import logging
from typing import Dict, Set

logger = logging.getLogger(__name__)

# Las 20 ligas TOP a nivel mundial con sus IDs en TheStatsAPI
# Incluye: 5 ligas europeas top + MLS + Liga MX + competiciones europeas + otras principales
TOP_20_LEAGUES: Dict[int, str] = {
    # Ligas europeas (7)
    39: "Premier League",  # England
    140: "La Liga",  # Spain
    135: "Serie A",  # Italy
    78: "Bundesliga",  # Germany
    61: "Ligue 1",  # France
    203: "Primeira Liga",  # Portugal
    94: "Eredivisie",  # Netherlands
    # Competiciones europeas (2)
    2: "Champions League",
    3: "Europa League",
    # Liga MX (1)
    262: "Liga MX",  # Mexico
    # MLS (1)
    848: "MLS",  # USA/Canada
    # Ligas secundarias europeas (4)
    179: "Super Lig",  # Turkey
    307: "Süper Lig",  # Turkey (alternativa)
    766: "Saudi Pro League",  # Saudi Arabia
    235: "Ukrainian Premier League",  # Ukraine
    # Ligas americanas (2)
    71: "Copa América",  # South America
    # Ligas asiáticas (1)
    251: "Chinese Super League",  # China
    # Especiales (2)
    81: "World Cup",  # FIFA World Cup
    529: "Copa Libertadores",  # South America
}

# IDs alternativos / sinónimos encontrados
LEAGUE_ALIASES: Dict[int, int] = {
    307: 179,  # Süper Lig → Super Lig
}


class LeagueFilter:
    """Filtro para mantener solo las 20 ligas top."""

    TOP_LEAGUE_IDS: Set[int] = set(TOP_20_LEAGUES.keys())

    @classmethod
    def is_top_league(cls, competition_id: int | None) -> bool:
        """Verifica si una competición es una de las 20 ligas top.

        Args:
            competition_id: ID de la competición en TheStatsAPI

        Returns:
            True si la competición está en TOP_20_LEAGUES
        """
        if competition_id is None:
            return False
        return competition_id in cls.TOP_LEAGUE_IDS

    @classmethod
    def get_league_name(cls, competition_id: int | None) -> str:
        """Devuelve el nombre de la liga.

        Args:
            competition_id: ID de la competición

        Returns:
            Nombre de la liga o "Unknown" si no se encuentra
        """
        if competition_id is None:
            return "Unknown"
        return TOP_20_LEAGUES.get(competition_id, "Unknown")

    @classmethod
    def filter_matches(cls, matches: list[dict]) -> list[dict]:
        """Filtra partidos para mantener solo los de las 20 ligas top.

        Args:
            matches: Lista de partidos

        Returns:
            Lista filtrada con solo partidos de las 20 ligas top
        """
        filtered = [
            m for m in matches
            if cls.is_top_league(m.get("competition_id"))
        ]
        logger.info(
            "Filtrados %d/%d partidos → solo 20 ligas top",
            len(filtered),
            len(matches)
        )
        return filtered
