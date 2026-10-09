from __future__ import annotations

import csv
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def print_table(results: List[Dict[str, Any]], min_pct: float) -> None:
    """Imprime la tabla final de resultados."""
    if not results:
        print(f"\nNo hay partidos con Ensemble BTTS > {min_pct}%.\n")
        return

    table = []
    for idx, result in enumerate(results, 1):
        table.append(
            [
                idx,
                (result.get("utc_date") or "")[:16].replace("T", " "),
                result.get("home_team", ""),
                result.get("away_team", ""),
                f"{result.get('prob_historical', 0):.1f}",
                f"{result.get('prob_bivariate_poisson', 0):.1f}",
                f"{result.get('prob_lstm_momentum', 0):.1f}",
                f"{result.get('prob_xgboost', 0):.1f}",
                f"{result.get('prob_catboost', 0):.1f}",
                f"{result.get('ensemble_pct', 0):.1f}",
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
    try:
        from tabulate import tabulate

        print("\n" + tabulate(table, headers=headers, tablefmt="github"))
    except ImportError:  # pragma: no cover
        print(table)
    print(f"\nTop {len(results)} partidos con Ensemble BTTS > {min_pct}%\n")


def export_csv(results: List[Dict[str, Any]], date_from: str, date_to: str) -> str:
    """Exporta resultados al CSV correspondiente."""
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
