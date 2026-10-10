"""Orquestador principal de análisis multi-modelo de BTTS con impacto de bajas."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from .data_fetcher import DataFetcher
from .injuries import InjuryImpactAnalyzer
from .models import (
    BivariatePoisson,
    CatBoostBTTS,
    HistoricalBTTS,
    LSTMMomentum,
    XGBoostBTTS,
)

logger = logging.getLogger(__name__)


class BTTSAnalyzer:
    """Orquesta 5 modelos para predecir BTTS con ensemble, considerando bajas."""

    def __init__(self, fetcher: DataFetcher, client: Any) -> None:
        self.fetcher = fetcher
        self.client = client
        self.injury_analyzer = InjuryImpactAnalyzer()
        self.models = [
            HistoricalBTTS(n=10),
            BivariatePoisson(n=20),
            LSTMMomentum(seq_len=8, epochs=35),
            XGBoostBTTS(n=15),
            CatBoostBTTS(n=15),
        ]
        logger.info(
            "Inicializado analyzer con %d modelos: %s",
            len(self.models),
            ", ".join(m.name for m in self.models),
        )

    def analyze(
        self,
        upcoming: List[Dict[str, Any]],
        min_pct: float = 80.0,
        top_n: int = 20,
    ) -> List[Dict[str, Any]]:
        """Analiza partidos con todos los modelos y devuelve un ensemble ajustado por bajas."""
        team_cache: Dict[str, List[Dict[str, Any]]] = {}
        results: List[Dict[str, Any]] = []
        total_matches = len(upcoming)

        for idx, match in enumerate(upcoming, 1):
            home = match.get("home_team") or {}
            away = match.get("away_team") or {}
            home_id = home.get("id")
            away_id = away.get("id")
            if not home_id or not away_id:
                logger.debug("[%d/%d] Saltando partido sin IDs", idx, total_matches)
                continue

            logger.info(
                "[%d/%d] %s vs %s",
                idx,
                total_matches,
                home.get("name"),
                away.get("name"),
            )

            if home_id not in team_cache:
                team_cache[home_id] = self.fetcher.get_team_finished_matches(
                    home_id, n=30
                )
            if away_id not in team_cache:
                team_cache[away_id] = self.fetcher.get_team_finished_matches(
                    away_id, n=30
                )

            home_m = team_cache[home_id]
            away_m = team_cache[away_id]

            scores = {}
            for model in self.models:
                try:
                    scores[model.name] = model.predict(home_m, away_m)
                except Exception as exc:
                    logger.warning(
                        "Modelo %s falló para %s vs %s: %s",
                        model.name,
                        home.get("name"),
                        away.get("name"),
                        exc,
                    )
                    scores[model.name] = 0.0

            valid = [v for v in scores.values() if v > 0]
            ensemble = round(sum(valid) / len(valid), 2) if valid else 0.0

            injury_data = self.injury_analyzer.analyze_injuries(
                str(home_id), str(away_id), self.client
            )

            adjusted_ensemble = self.injury_analyzer.adjust_ensemble_by_injuries(
                ensemble,
                injury_data["home_injury_impact"],
                injury_data["away_injury_impact"],
            )

            if adjusted_ensemble >= min_pct:
                results.append(
                    {
                        "match_id": match.get("id"),
                        "utc_date": match.get("utc_date"),
                        "competition_id": match.get("competition_id"),
                        "home_team": home.get("name"),
                        "away_team": away.get("name"),
                        "home_team_id": home_id,
                        "away_team_id": away_id,
                        **{f"prob_{k}": v for k, v in scores.items()},
                        "ensemble_pct": ensemble,
                        "adjusted_ensemble_pct": adjusted_ensemble,
                        "home_injury_impact": injury_data["home_injury_impact"],
                        "away_injury_impact": injury_data["away_injury_impact"],
                        "home_injuries_count": len(injury_data["home_injuries"]),
                        "away_injuries_count": len(injury_data["away_injuries"]),
                    }
                )

        results.sort(key=lambda r: r["adjusted_ensemble_pct"], reverse=True)
        final = results[:top_n]
        logger.info(
            "Análisis completado: %d/%d partidos cumplen umbral de %.1f%% (post-bajas)",
            len(final),
            total_matches,
            min_pct,
        )
        return final
