# src/data_fetcher.py
"""Obtención de partidos futuros e históricos desde TheStatsAPI."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .api_client import StatsAPIClient
from .top_leagues import LeagueFilter

logger = logging.getLogger(__name__)


class DataFetcher:
    """Obtiene datos de partidos desde TheStatsAPI con paginación."""

    def __init__(self, client: StatsAPIClient) -> None:
        self.client = client

    def get_upcoming_matches(
        self,
        date_from: str = "",
        date_to: str = "",
        days_ahead: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Obtiene partidos programados en un rango de fechas."""
        today = datetime.now(timezone.utc).date().isoformat()

        if days_ahead is not None and not date_from and not date_to:
            date_from = today
            date_to = (
                datetime.fromisoformat(today) + timedelta(days=days_ahead)
            ).date().isoformat()
        else:
            date_from = date_from.strip() if date_from else today
            date_to = date_to.strip() if date_to else date_from

        logger.info("Obteniendo partidos scheduled %s → %s", date_from, date_to)
        all_matches: List[Dict[str, Any]] = []
        page = 1
        per_page = 100

        while True:
            data = self.client.get(
                "/football/matches",
                params={
                    "status": "scheduled",
                    "date_from": date_from,
                    "date_to": date_to,
                    "page": page,
                    "per_page": per_page,
                },
            )
            matches = data.get("data", [])
            if not matches:
                break

            filtered = LeagueFilter.filter_matches(matches)
            all_matches.extend(filtered)

            meta = data.get("meta", {})
            total_pages = meta.get("total_pages", 1)
            logger.info(
                "Página %d/%d → %d partidos (top leagues) (total: %d)",
                page, total_pages, len(filtered), len(all_matches),
            )
            if page >= total_pages:
                break
            page += 1

        logger.info(
            "Total partidos futuros (ligas top) obtenidos: %d", len(all_matches)
        )
        return all_matches

    def get_team_finished_matches(
        self, team_id: str, n: int = 30
    ) -> List[Dict[str, Any]]:
        """Obtiene los últimos N partidos finalizados de un equipo."""
        data = self.client.get(
            "/football/matches",
            params={
                "team_id": team_id,
                "status": "finished",
                "per_page": min(n + 15, 100),
                "page": 1,
            },
        )
        matches = data.get("data", [])
        matches.sort(key=lambda m: m.get("utc_date", ""), reverse=True)
        return matches[:n]

    def get_match(self, match_id, use_cache: bool = False) -> Dict[str, Any]:
        """Obtiene un partido concreto por id."""
        data = self.client.get(
            f"/football/matches/{match_id}", use_cache=use_cache
        )
        return data.get("data", data)
