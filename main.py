#!/usr/bin/env python3
"""
BTTS Multi-Model Predictor – TheStatsAPI
Listo para producción y GitHub Actions.
"""

from __future__ import annotations

import csv
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

from dotenv import load_dotenv
from tabulate import tabulate

from src.api_client import StatsAPIClient
from src.analyzer import BTTSAnalyzer
from src.data_fetcher import DataFetcher

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")

MIN_PCT = 80.0
TOP_N = 20
CSV_NAME = f"btts_top20_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv"


def print_table(results: List[Dict[str, Any]]) -> None:
    if not results:
        print("\nNo hay partidos con ensemble BTTS > 80 %.\n")
        return

    table = []
    for i, r in enumerate(results, 1):
        table.append(
            [
                i,
                (r.get("utc_date") or "")[:16].replace("T", " "),
                r["home_team"],
                r["away_team"],
                f"{r.get('prob_historical', 0):.1f}",
                f"{r.get('prob_bivariate_poisson', 0):.1f}",
                f"{r.get('prob_lstm_momentum', 0):.1f}",
                f"{r.get('prob_xgboost', 0):.1f}",
                f"{r.get('prob_catboost', 0):.1f}",
                f"{r['ensemble_pct']:.1f}",
            ]
        )

    headers = [
        "#", "Fecha UTC", "Local", "Visitante",
        "Hist%", "Poisson%", "LSTM%", "XGB%", "Cat%", "Ensemble%"
    ]
    print("\n" + tabulate(table, headers=headers, tablefmt="github"))
    print(f"\nTop {len(results)} partidos con Ensemble BTTS > {MIN_PCT}%\n")


def export_csv(results: List[Dict[str, Any]], filename: str) -> None:
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info("CSV exportado → %s", filename)


def main() -> None:
    logger.info("=== BTTS Multi-Model Predictor (TheStatsAPI) ===")
    try:
        client = StatsAPIClient()
        fetcher = DataFetcher(client)
        analyzer = BTTSAnalyzer(fetcher)

        upcoming = fetcher.get_upcoming_matches(days_ahead=14)
        if not upcoming:
            logger.warning("No hay partidos programados.")
            sys.exit(0)

        top = analyzer.analyze(upcoming, min_pct=MIN_PCT, top_n=TOP_N)
        print_table(top)
        export_csv(top, CSV_NAME)
        logger.info("Proceso terminado correctamente.")
    except Exception as exc:
        logger.exception("Error fatal: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
