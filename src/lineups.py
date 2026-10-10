"""Módulo para extraer información de lineups confirmados y calcular ausencia relevante."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

CRITICAL_POSITIONS = {"STRIKER", "FORWARD", "CF", "ST", "WINGER", "LW", "RW", "AM"}
HIGH_IMPACT_POSITIONS = {"ATTACKING MIDFIELD", "LEFT MIDFIELD", "RIGHT MIDFIELD", "CM", "CAM"}
MID_IMPACT_POSITIONS = {"MIDFIELD", "CENTRE MIDFIELD", "CENTRE-MIDFIELD"}
DEFENCE_POSITIONS = {"DEFENDER", "CENTRE-BACK", "CB", "LEFT-BACK", "LB", "RIGHT-BACK", "RB"}
GOALKEEPER_POSITIONS = {"GOALKEEPER", "GK"}


class LineupAnalyzer:
    """Analiza lineups confirmados para detectar ausencias relevantes."""

    @staticmethod
    def _normalize_position(position: Optional[str]) -> str:
        return (position or "").strip().upper()

    @classmethod
    def _position_impact(cls, position: Optional[str]) -> float:
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

    @staticmethod
    def fetch_lineups(client: Any, match_id: int | str) -> Dict[str, Any]:
        """Obtiene alineaciones desde /api/football/matches/{match_id}/lineups."""
        try:
            return client.get(f"/api/football/matches/{match_id}/lineups")
        except Exception as exc:  # 404 si la alineación aún no está confirmada
            logger.debug("No hay lineups confirmados para match_id=%s: %s", match_id, exc)
            return {}

    @classmethod
    def extract_absences_from_lineup(cls, lineup_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extrae jugadores ausentes relevantes a partir de un lineup confirmado.

        Si la respuesta incluye `confirmed`, `lineups` o `teams`, intenta detectar
        diferencias entre once inicial y suplentes o marcadores de ausencia.
        """
        if not lineup_payload:
            return []

        absences: List[Dict[str, Any]] = []

        teams = lineup_payload.get("data") or lineup_payload.get("teams") or []
        if isinstance(teams, dict):
            teams = [teams]

        for team in teams:
            if not isinstance(team, dict):
                continue

            team_name = team.get("name") or team.get("team") or team.get("team_name") or "Unknown"
            roster = (
                team.get("lineup")
                or team.get("starting_lineup")
                or team.get("startingXI")
                or team.get("players")
                or []
            )

            if not isinstance(roster, list):
                continue

            # Caso 1: lineup con `players` / `starting_lineup`
            for player in roster:
                player_name = player.get("name") or player.get("player") or player.get("full_name")
                if not player_name:
                    continue

                status = str(player.get("status") or player.get("availability") or "").lower()
                if status in {"out", "injured", "injury", "suspended", "doubtful", "not_available"}:
                    absences.append(
                        {
                            "team": team_name,
                            "player": player_name,
                            "position": player.get("position") or player.get("role") or "",
                            "status": status,
                            "reason": player.get("reason") or status,
                        }
                    )

            # Caso 2: si la API expone un field `confirmed` con por ejemplo `selected` o `bench`
            # se ignora si no hay señales explícitas de ausencia
            bench = team.get("bench") or team.get("substitutes") or []
            if isinstance(bench, list):
                for player in bench:
                    player_name = player.get("name") or player.get("player")
                    if not player_name:
                        continue
                    status = str(player.get("status") or "").lower()
                    if status in {"out", "injured", "suspended", "not_available"}:
                        absences.append(
                            {
                                "team": team_name,
                                "player": player_name,
                                "position": player.get("position") or "",
                                "status": status,
                                "reason": player.get("reason") or status,
                            }
                        )

        # Deduplicación
        unique: Dict[str, Dict[str, Any]] = {}
        for item in absences:
            key = f"{item.get('team')}::{item.get('player')}::{item.get('status')}"
            unique[key] = item
        return list(unique.values())

    @classmethod
    def evaluate_absence_impact(cls, absences: Iterable[Dict[str, Any]]) -> float:
        """Calcula el impacto de las ausencias usando la posición del jugador."""
        total = 0.0
        count = 0
        for absence in absences:
            impact = cls._position_impact(absence.get("position"))
            if impact > 0.0:
                total += impact
                count += 1
        if count == 0:
            return 0.0
        return round(min(total / count, 0.90), 2)

    @classmethod
    def analyze_match_lineups(cls, client: Any, match_id: int | str, team_name: str | None = None) -> Dict[str, Any]:
        """Obtiene y resuelve el impacto de absencias a partir de lineups confirmados."""
        raw = cls.fetch_lineups(client, match_id)
        absences = cls.extract_absences_from_lineup(raw)
        impact = cls.evaluate_absence_impact(absences)

        if team_name:
            absences = [item for item in absences if item.get("team") == team_name]
            impact = cls.evaluate_absence_impact(absences)

        return {
            "match_id": match_id,
            "confirmed": bool(raw),
            "absences": absences,
            "impact": impact,
        }
