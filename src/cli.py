from __future__ import annotations

import argparse
import logging
import os
from typing import Sequence

from .api_client import StatsAPIClient
from .analyzer import BTTSAnalyzer
from .config import Settings
from .data_fetcher import DataFetcher
from .exporter import export_csv, print_table

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parsea los argumentos de la CLI."""
    parser = argparse.ArgumentParser(
        description="BTTS Multi-Model Predictor – TheStatsAPI"
    )
    parser.add_argument("--date-from", default=os.environ.get("DATE_FROM", ""))
    parser.add_argument("--date-to", default=os.environ.get("DATE_TO", ""))
    parser.add_argument("--min-pct", type=float, default=float(os.environ.get("MIN_PCT", "80")))
    parser.add_argument("--top-n", type=int, default=int(os.environ.get("TOP_N", "20")))
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    return parser.parse_args(argv)


def cli(argv: Sequence[str] | None = None) -> int:
    """Ejecuta el flujo principal de predicción con análisis de bajas."""
    args = parse_args(argv)
    settings = Settings.from_dict(
        {
            "date_from": args.date_from,
            "date_to": args.date_to,
            "min_pct": args.min_pct,
            "top_n": args.top_n,
            "log_level": args.log_level,
        }
    )

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("=== BTTS Multi-Model Predictor (TheStatsAPI) ===")
    logger.info("Rango de fechas : %s → %s", settings.date_from, settings.date_to)
    logger.info("Umbral mínimo   : %.1f%%", settings.min_pct)
    logger.info("Top N           : %d", settings.top_n)

    try:
        client = StatsAPIClient()
        fetcher = DataFetcher(client)
        analyzer = BTTSAnalyzer(fetcher)

        upcoming = fetcher.get_upcoming_matches(
            date_from=settings.date_from,
            date_to=settings.date_to,
        )

        if not upcoming:
            logger.warning("No se encontraron partidos en el rango indicado.")
            return 0

        top_results = analyzer.analyze(
            upcoming,
            min_pct=settings.min_pct,
            top_n=settings.top_n,
        )

        print_table(top_results, settings.min_pct)
        export_csv(top_results, settings.date_from, settings.date_to)
        logger.info("Proceso terminado correctamente.")
        return 0
    except Exception as exc:  # pragma: no cover - código de ejecución
        logger.exception("Error fatal: %s", exc)
        return 1
