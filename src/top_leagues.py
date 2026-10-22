"""
Configuración de competiciones soportadas (ligas + torneos).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

LEAGUES: dict[str, dict] = {
    "premier_league": {"name": "Premier League", "country": "England",
                       "competition_id": "comp_3039", "type": "league"},
    "bundesliga": {"name": "Bundesliga", "country": "Germany",
                   "competition_id": "comp_4643", "type": "league"},
    "la_liga": {"name": "La Liga", "country": "Spain",
                "competition_id": "comp_8814", "type": "league"},
    "serie_a_ita": {"name": "Serie A", "country": "Italy",
                    "competition_id": "comp_5840", "type": "league"},
    "ligue_1": {"name": "Ligue 1", "country": "France",
                "competition_id": "comp_0256", "type": "league"},
    "brasileirao": {"name": "Brasileirão Série A", "country": "Brazil",
                    "competition_id": "comp_4795", "type": "league"},
    "championship": {"name": "Championship", "country": "England",
                     "competition_id": "comp_8321", "type": "league"},
    "liga_mx": {"name": "Liga MX", "country": "Mexico",
                "competition_id": "comp_298265", "type": "league"},
    "uefa_champions_league": {"name": "UEFA Champions League", "country": "Europe",
                              "competition_id": "comp_3498", "type": "tournament",
                              "cross_league": True},
    "uefa_europa_league": {"name": "UEFA Europa League", "country": "Europe",
                           "competition_id": "comp_7739", "type": "tournament",
                           "cross_league": True},
    "copa_libertadores": {"name": "CONMEBOL Libertadores", "country": "South America",
                          "competition_id": "comp_0499", "type": "tournament",
                          "cross_league": True},
}


class LeagueFilter:
    """Filtra partidos a las competiciones configuradas (por competition_id)."""

    _ids: Optional[set] = None

    @classmethod
    def _allowed_ids(cls) -> set:
        if cls._ids is None:
            cls._ids = {v["competition_id"] for v in LEAGUES.values()
                        if v.get("competition_id")}
        return cls._ids

    @classmethod
    def filter_matches(cls, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        allowed = cls._allowed_ids()
        return [m for m in matches if m.get("competition_id") in allowed]


def get_competition_id(key: str, client: Any = None) -> str:
    """Resuelve el competition_id. Requiere `client` para hacer lookup si no está mapeado."""
    league = LEAGUES.get(key)
    if league is None:
        raise KeyError(f"Competition '{key}' not found. Available: {list(LEAGUES.keys())}")
    if league["competition_id"]:
        return league["competition_id"]
    if client is None:
        raise RuntimeError(f"'{key}' requiere lookup: pasa un StatsAPIClient")
    data = client.get("/football/competitions", {"search": league["name"]})
    for comp in data.get("data", []):
        if comp.get("name", "").lower() == league["name"].lower():
            league["competition_id"] = comp["id"]
            LeagueFilter._ids = None  # invalidar caché de ids
            return comp["id"]
    raise RuntimeError(f"Could not resolve competition_id for '{league['name']}'")
