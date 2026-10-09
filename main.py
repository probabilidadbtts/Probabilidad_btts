#!/usr/bin/env python3
"""
BTTS Multi-Model Predictor – TheStatsAPI
Versión completa y correcta con selector de fechas (día a día / rango).
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from datetime import datetime, timezone
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BTTS Multi-Model Predictor – solo partidos del rango indicado"
    )
    parser.add_argument(
        "--date-from",
        type=str,
        default=os.environ.get("DATE_FROM", ""),
        help="Fecha desde (YYYY-MM-DD). Vacío = hoy (UTC)",
    )
    parser.add_argument(
        "--date-to",
        type=str,
        default=os.environ.get("DATE_TO", ""),
        help="Fecha hasta (YYYY-MM-DD). Vacío = mismo día que date-from",
    )
    parser.add_argument(
        "--min-pct",
        type=float,
        default=float(os.environ.get("MIN_PCT", "80")),
        help="Umbral mínimo de Ensemble BTTS (default 80)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=int(os.environ.get("TOP_N", "20")),
        help="Máximo de partidos a devolver (default 20)",
    )
    return parser.parse_args()


def print_table(results: List[Dict[str, Any]], min_pct: float) -> None:
    if not results:
        print(f"\nNo hay partidos con Ensemble BTTS > {min_pct}%.\n")
        return

    table = []
    for i, r in enumerate(results, 1):
        table.append(
            [
                i,
                (r.get("utc_date") or "")[:16].replace("T", " "),
                r.get("home_team", ""),
                r.get("away_team", ""),
                f"{r.get('prob_historical', 0):.1f}",
                f"{r.get('prob_bivariate_poisson', 0):.1f}",
                f"{r.get('prob_lstm_momentum', 0):.1f}",
                f"{r.get('prob_xgboost', 0):.1f}",
                f"{r.get('prob_catboost', 0):.1f}",
                f"{r.get('ensemble_pct', 0):.1f}",
            ]
        )

    headers = [
        "#",
        "Fecha UTC",
        "Local",
        "Visitante",
        "Hist%",
        "Poisson%",
        "LSTM%",
        "XGB%",
        "Cat%",
        "Ensemble%",
    ]
    print("\n" + tabulate(table, headers=headers, tablefmt="github"))
    print(f"\nTop {len(results)} partidos con Ensemble BTTS > {min_pct}%\n")


def export_csv(results: List[Dict[str, Any]], date_from: str, date_to: str) -> str:
    if not results:
        logger.warning("No hay resultados para exportar a CSV.")
        return ""

    filename = f"btts_{date_from}_to_{date_to}.csv"
    fieldnames = list(results[0].keys())

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info("CSV exportado → %s", filename)
    return filename


def main() -> None:
    args = parse_args()

    # Normalizamos fechas
    today = datetime.now(timezone.utc).date().isoformat()
    date_from = args.date_from.strip() if args.date_from else today
    date_to = args.date_to.strip() if args.date_to else date_from

    logger.info("=== BTTS Multi-Model Predictor (TheStatsAPI) ===")
    logger.info("Rango de fechas : %s → %s", date_from, date_to)
    logger.info("Umbral mínimo   : %.1f%%", args.min_pct)
    logger.info("Top N           : %d", args.top_n)

    try:
        client = StatsAPIClient()
        fetcher = DataFetcher(client)
        analyzer = BTTSAnalyzer(fetcher)

        upcoming = fetcher.get_upcoming_matches(
            date_from=date_from,
            date_to=date_to,
        )

        if not upcoming:
            logger.warning("No se encontraron partidos programados en el rango indicado.")
            sys.exit(0)

        top_results = analyzer.analyze(
            upcoming,
            min_pct=args.min_pct,
            top_n=args.top_n,
        )

        print_table(top_results, args.min_pct)
        export_csv(top_results, date_from, date_to)

        logger.info("Proceso terminado correctamente.")

    except Exception as exc:
        logger.exception("Error fatal: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
