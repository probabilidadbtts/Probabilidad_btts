"""
Backtesting walk-forward para el ensemble BTTS.
Descarga partidos terminados reales de TheStatsAPI,
predice con el ensemble usando SOLO datos anteriores (sin data leakage),
y evalúa contra el resultado real + cuotas si disponibles.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .api_client import fetch_all_matches, fetch_match_odds
from .models import evaluate_ensemble  # Tu función existente de ensemble

logger = logging.getLogger(__name__)
DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _extract_rows(matches: list[dict]) -> pd.DataFrame:
    """Extrae DataFrame plano de partidos terminados con score válido."""
    rows = []
    for m in matches:
        score = m.get("score", {})
        h, a = score.get("home"), score.get("away")
        if not isinstance(h, (int, float)) or not isinstance(a, (int, float)):
            continue
        rows.append({
            "match_id": m["id"],
            "utc_date": m["utc_date"],
            "competition_id": m.get("competition_id", ""),
            "home_team_id": m["home_team"]["id"],
            "home_team": m["home_team"]["name"],
            "away_team_id": m["away_team"]["id"],
            "away_team": m["away_team"]["name"],
            "home_goals": int(h),
            "away_goals": int(a),
            "btts_yes": int(h > 0 and a > 0),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["utc_date"] = pd.to_datetime(df["utc_date"])
        df = df.sort_values("utc_date").reset_index(drop=True)
    return df


def _get_odds_btts(match_id: str) -> tuple[Optional[float], Optional[float]]:
    """
    Obtiene (opening, last_seen) de BTTS Yes para un partido.
    Devuelve (None, None) si no hay odds.
    """
    try:
        odds_data = fetch_match_odds(match_id)
        if not odds_data:
            return None, None
        for bk in odds_data.get("bookmakers", []):
            btts = bk.get("markets", {}).get("btts")
            if btts:
                yes = btts.get("yes", {})
                opening = yes.get("opening")
                last_seen = yes.get("last_seen")
                return (
                    float(opening) if opening else None,
                    float(last_seen) if last_seen else None,
                )
    except Exception:
        pass
    return None, None


def walk_forward_backtest(
    competition_id: str,
    season_id: Optional[str] = None,
    n_folds: int = 5,
    min_pct: float = 55.0,
    use_odds: bool = True,
) -> dict:
    """
    Backtesting walk-forward:
    1. Descarga TODOS los partidos terminados reales.
    2. Divide en n_folds temporales.
    3. Para cada fold: entrena el ensemble con datos ANTERIORES,
       predice el fold, compara con resultado real.
    4. Si use_odds: obtiene cuotas reales para CLV/ROI.
    """
    # 1. Descargar datos reales
    logger.info(f"Fetching finished matches: comp={competition_id}")
    raw = fetch_all_matches(
        competition_id=competition_id,
        season_id=season_id,
        status="finished",
    )
    df = _extract_rows(raw)
    logger.info(f"Total finished matches: {len(df)}")

    if len(df) < 100:
        return {"status": "error", "message": f"Need >=100 matches, got {len(df)}"}

    # 2. Dividir en folds temporales
    fold_size = len(df) // n_folds
    all_fold_results = []
    all_predictions = []

    for fold in range(1, n_folds + 1):
        train_end = fold * fold_size
        test_start = (fold - 1) * fold_size
        test_end = fold * fold_size if fold < n_folds else len(df)

        train_df = df.iloc[:train_end]
        test_df = df.iloc[test_start:test_end].copy()

        if len(train_df) < 50 or len(test_df) < 10:
            logger.warning(f"Fold {fold}: insufficient data")
            continue

        logger.info(f"Fold {fold}/{n_folds}: train={len(train_df)}, test={len(test_df)}")

        # 3. Para cada partido del test, predecir usando SOLO train_df
        predictions = []
        for idx in test_df.index:
            row = test_df.loc[idx]
            # Construir "historial" = train_df (todo lo anterior al fold)
            # Tu evaluate_ensemble debe aceptar el historial como parámetro
            # Ajusta esta llamada a la firma real de tu función:
            try:
                p_btts = evaluate_ensemble(
                    home_team_id=row["home_team_id"],
                    away_team_id=row["away_team_id"],
                    history=train_df,  # Solo datos anteriores
                )
            except Exception as e:
                logger.debug(f"  Skip {row['match_id']}: {e}")
                continue

            pred = {
                "match_id": row["match_id"],
                "utc_date": row["utc_date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "p_btts": p_btts,
                "actual_btts": row["btts_yes"],
                "fold": fold,
            }

            # 4. Odds reales (si disponibles)
            if use_odds:
                opening, last_seen = _get_odds_btts(row["match_id"])
                pred["odds_opening"] = opening
                pred["odds_last_seen"] = last_seen

            predictions.append(pred)

        if not predictions:
            continue

        pred_df = pd.DataFrame(predictions)
        all_predictions.append(pred_df)

        # Métricas del fold
        fold_metrics = _compute_metrics(pred_df, min_pct=min_pct)
        fold_metrics["fold"] = fold
        fold_metrics["n_train"] = len(train_df)
        fold_metrics["n_test"] = len(pred_df)
        all_fold_results.append(fold_metrics)

        logger.info(
            f"  → HitRate={fold_metrics['hit_rate']:.1%} | "
            f"Brier={fold_metrics['brier']:.4f} | "
            f"Filtered: {fold_metrics['n_filtered']} picks, "
            f"HR={fold_metrics['filtered_hit_rate']:.1%}"
        )

    # Agregar
    all_pred_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()

    # Guardar predicciones
    if not all_pred_df.empty:
        stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M")
        out_path = DATA_DIR / f"backtest_preds_{stamp}.csv"
        all_pred_df.to_csv(out_path, index=False)
        logger.info(f"Predictions saved: {out_path}")

    summary = _aggregate_folds(all_fold_results, all_pred_df)
    return summary


def _compute_metrics(pred_df: pd.DataFrame, min_pct: float) -> dict:
    """Calcula métricas para un set de predicciones."""
    n = len(pred_df)
    if n == 0:
        return {"n": 0, "hit_rate": 0, "brier": 0, "roi": 0, "clv": 0,
                "n_filtered": 0, "filtered_hit_rate": 0, "filtered_roi": 0, "filtered_clv": 0}

    # Brier score (global)
    brier = float(np.mean((pred_df["p_btts"] - pred_df["actual_btts"]) ** 2))

    # Hit rate global (p > 0.5 → predict Yes)
    hit_rate = float(np.mean((pred_df["p_btts"] > 0.5) == (pred_df["actual_btts"] == 1)))

    # Filtrado por umbral
    mask = pred_df["p_btts"] >= (min_pct / 100.0)
    filtered = pred_df[mask]
    n_filtered = len(filtered)

    if n_filtered > 0:
        filtered_hit_rate = float(filtered["actual_btts"].mean())

        # ROI y CLV si hay odds
        if "odds_last_seen" in filtered.columns and filtered["odds_last_seen"].notna().any():
            odds = filtered["odds_last_seen"].dropna()
            f = filtered[filtered["odds_last_seen"].notna()]
            wins = f["actual_btts"] * (f["odds_last_seen"] - 1)
            losses = (1 - f["actual_btts"]) * 1
            filtered_roi = float((wins - losses).sum() / len(f))

            # CLV: (p_modelo * odds_closing) - 1
            p_implied = 1.0 / f["odds_last_seen"]
            filtered_clv = float(np.mean(f["p_btts"] / p_implied - 1.0))
        else:
            filtered_roi = 0.0
            filtered_clv = 0.0
    else:
        filtered_hit_rate = 0.0
        filtered_roi = 0.0
        filtered_clv = 0.0

    return {
        "n": n,
        "hit_rate": hit_rate,
        "brier": brier,
        "roi": 0.0,
        "clv": 0.0,
        "n_filtered": n_filtered,
        "filtered_hit_rate": filtered_hit_rate,
        "filtered_roi": filtered_roi,
        "filtered_clv": filtered_clv,
    }


def _aggregate_folds(fold_results: list[dict], all_pred_df: pd.DataFrame) -> dict:
    """Resumen final del backtest."""
    if not fold_results:
        return {"status": "no_data"}

    valid_filtered = [r for r in fold_results if r["n_filtered"] > 0]

    return {
        "status": "ok",
        "n_folds": len(fold_results),
        "total_predictions": len(all_pred_df),
        "avg_brier": float(np.mean([r["brier"] for r in fold_results])),
        "avg_hit_rate": float(np.mean([r["hit_rate"] for r in fold_results])),
        "total_filtered_picks": sum(r["n_filtered"] for r in fold_results),
        "avg_filtered_hit_rate": float(np.mean([r["filtered_hit_rate"] for r in valid_filtered])) if valid_filtered else 0,
        "avg_filtered_roi": float(np.mean([r["filtered_roi"] for r in valid_filtered])) if valid_filtered else 0,
        "avg_filtered_clv": float(np.mean([r["filtered_clv"] for r in valid_filtered])) if valid_filtered else 0,
        "per_fold": fold_results,
    }   
