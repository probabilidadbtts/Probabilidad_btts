"""
Configuración de competiciones soportadas (ligas + torneos).
"""

LEAGUES: dict[str, dict] = {
    # ─── LIGAS TOP ───────────────────────────────────────────────────────────
    "premier_league": {
        "name": "Premier League",
        "country": "England",
        "competition_id": "comp_3039",
        "type": "league",
    },
    "bundesliga": {
        "name": "Bundesliga",
        "country": "Germany",
        "competition_id": "comp_4643",
        "type": "league",
    },
    "la_liga": {
        "name": "La Liga",
        "country": "Spain",
        "competition_id": "comp_8814",
        "type": "league",
    },
    "serie_a_ita": {
        "name": "Serie A",
        "country": "Italy",
        "competition_id": "comp_5840",
        "type": "league",
    },
    "ligue_1": {
        "name": "Ligue 1",
        "country": "France",
        "competition_id": "comp_0256",
        "type": "league",
    },
    "brasileirao": {
        "name": "Brasileirão Série A",
        "country": "Brazil",
        "competition_id": "comp_4795",
        "type": "league",
    },
    "liga_profesional_arg": {
        "name": "Liga Profesional",
        "country": "Argentina",
        "competition_id": None,  # lookup
        "type": "league",
    },
    "jupiler_pro": {
        "name": "Jupiler Pro League",
        "country": "Belgium",
        "competition_id": None,  # lookup
        "type": "league",
    },
    "primeira_liga": {
        "name": "Primeira Liga",
        "country": "Portugal",
        "competition_id": None,  # lookup
        "type": "league",
    },
    "championship": {
        "name": "Championship",
        "country": "England",
        "competition_id": "comp_8321",
        "type": "league",
    },
    "liga_mx": {
        "name": "Liga MX",
        "country": "Mexico",
        "competition_id": "comp_298265",  # Apertura; usar lookup para Clausura
        "type": "league",
    },
    "mls": {
        "name": "MLS",
        "country": "USA",
        "competition_id": None,  # lookup
        "type": "league",
    },

    # ─── TORNEOS UEFA ────────────────────────────────────────────────────────
    "uefa_europa_league": {
        "name": "UEFA Europa League",
        "country": "Europe",
        "competition_id": "comp_7739",
        "type": "tournament",
        "cross_league": True,
    },
    "uefa_conference_league": {
        "name": "UEFA Conference League",
        "country": "Europe",
        "competition_id": None,  # lookup — no confirmado en docs públicas
        "type": "tournament",
        "cross_league": True,
    },
    "uefa_champions_league": {
        "name": "UEFA Champions League",
        "country": "Europe",
        "competition_id": "comp_3498",
        "type": "tournament",
        "cross_league": True,
    },

    # ─── TORNEOS CONMEBOL ───────────────────────────────────────────────────
    "copa_libertadores": {
        "name": "CONMEBOL Libertadores",
        "country": "South America",
        "competition_id": "comp_0499",
        "type": "tournament",
        "cross_league": True,
    },
    "copa_sudamericana": {
        "name": "CONMEBOL Sudamericana",
        "country": "South America",
        "competition_id": None,  # lookup
        "type": "tournament",
        "cross_league": True,
    },
}


def get_competition_id(key: str) -> str:
    league = LEAGUES.get(key)
    if league is None:
        raise KeyError(f"Competition '{key}' not found. Available: {list(LEAGUES.keys())}")
    if league["competition_id"]:
        return league["competition_id"]
    from .api_client import api_get
    data = api_get("/football/competitions", {"search": league["name"]})
    for comp in data.get("data", []):
        if comp.get("name", "").lower() == league["name"].lower():
            league["competition_id"] = comp["id"]
            return comp["id"]
    raise RuntimeError(f"Could not resolve competition_id for '{league['name']}'")


def get_all_keys(competition_type: str = None) -> list[str]:
    """Devuelve todas las keys, filtradas por tipo si se especifica."""
    if competition_type is None:
        return list(LEAGUES.keys())
    return [k for k, v in LEAGUES.items() if v["type"] == competition_type]


def get_cross_league_keys() -> list[str]:
    """Competiciones con equipos de múltiples ligas (torneos internacionales)."""
    return [k for k, v in LEAGUES.items() if v.get("cross_league", False)]   
