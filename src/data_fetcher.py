"""Obtención de partidos futuros e históricos reales."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .api_client import StatsAPIClient

logger = logging.getLogger(__name__)


class DataFetcher:
    def __init__(self, client: StatsAPIClient) -> None:
        self.client = client

    def get_upcoming_matches(
        self, days_ahead: int = 14
    ) -> List[Dict[str, Any]]:
        today = datetime.now(timezone.utc).date()
        date_from = today.isoformat()
        date_to = (today + timedelta(days=days_ahead)).isoformat()

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
            all_matches.extend(matches)
            meta = data.get("meta", {})
            total_pages = meta.get("total_pages", 1)
            logger.info("Página %d/%d → %d partidos", page, total_pages, len(all_matches))
            if page >= total_pages:
                break
            page += 1

        logger.info("Total partidos futuros: %d", len(all_matches))
        return all_matches

    def get_team_finished_matches(
        self, team_id: str, n: int = 30
    ) -> List[Dict[str, Any]]:
        """Últimos N partidos finalizados (ordenados más reciente primero)."""
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
