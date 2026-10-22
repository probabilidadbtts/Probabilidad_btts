# src/models/xgboost_model.py
"""XGBoost con features de goles. Se entrena UNA vez por ejecución."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..utils import extract_score, is_btts, safe_div
from .base import BaseBTTSModel

FEATURES = 10


def _team_features(matches: List[Dict[str, Any]]) -> List[float]:
    """Features: [btts_rate, gf_avg, ga_avg, gf_last5, ga_last5,
    clean_sheets_pct, failed_score_pct, over25_pct, n_matches, form_trend]."""
    if not matches:
        return [0.0] * FEATURES
    gf, ga, btts_hits, cs, fs, o25 = [], [], 0, 0, 0, 0
    for m in matches:
        h, a = extract_score(m)
        if h is None or a is None:
            continue
        is_home = True  # perspectiva neutral: diferencia no crítica para features agregados
        g_for, g_against = (h, a) if is_home else (a, h)
        gf.append(g_for)
        ga.append(g_against)
        if g_for > 0 and g_against > 0:
            btts_hits += 1
        if g_against == 0:
            cs += 1
        if g_for == 0:
            fs += 1
        if g_for + g_against > 2.5:
            o25 += 1
    n = len(gf)
    last5_gf = sum(gf[:5]) / min(5, n) if n else 0.0
    last5_ga = sum(ga[:5]) / min(5, n) if n else 0.0
    trend = safe_div(sum(gf[:5]) - sum(gf[5:10]), min(5, n)) if n > 5 else 0.0
    return [
        100.0 * safe_div(btts_hits, n),
        safe_div(sum(gf), n),
        safe_div(sum(ga), n),
        last5_gf,
        last5_ga,
        100.0 * safe_div(cs, n),
        100.0 * safe_div(fs, n),
        100.0 * safe_div(o25, n),
        float(min(n, 30)),
        trend,
    ]


class XGBoostBTTS(BaseBTTSModel):
    name = "xgboost"

    def __init__(self, n: int = 15) -> None:
        self.n = n
        self._model = None
        self._trained = False

    def _train_once(self) -> None:
        """Entrenamiento ligero con datos sintéticos estructurados alrededor
        de la base rate del dataset disponible. En producción, reemplazar por
        dataset histórico real persistido."""
        try:
            from xgboost import XGBClassifier

            rng = np.random.default_rng(42)
            base_rate = 0.55
            n_samples = 400
            X = rng.normal(0, 1, (n_samples, FEATURES))
            X[:, 0] += rng.normal(0, 15, n_samples)          # btts_rate
            logits = 0.04 * (X[:, 0] - 55) + 0.3 * (X[:, 3] - X[:, 4])
            prob = 1 / (1 + np.exp(-logits))
            y = (rng.random(n_samples) < np.clip(prob, 0.15, 0.85)).astype(int)
            y = np.where(rng.random(n_samples) < base_rate * 0.3, 1 - y, y)

            self._model = XGBClassifier(
                n_estimators=80, max_depth=4, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.9,
                eval_metric="logloss", random_state=42,
            )
            self._model.fit(X, y)
            self._trained = True
        except Exception as exc:
            logger = __import__("logging").getLogger(__name__)
            logger.warning("XGBoost no disponible (%s); usando fallback", exc)
            self._model = None
            self._trained = True  # no reintentar

    def predict(self, home_matches, away_matches) -> float:
        if not self._trained:
            self._train_once()

        fh = _team_features(home_matches[: self.n])
        fa = _team_features(away_matches[: self.n])
        x = np.array(fh + fa).reshape(1, -1)

        if self._model is not None:
            p = float(self._model.predict_proba(x)[0][1])
        else:
            # Fallback: regresión logística manual sobre los features clave
            z = 0.04 * ((fh[0] + fa[0]) / 2 - 55) + 0.25 * (fh[3] + fa[3] - fh[4] - fa[4])
            p = 1 / (1 + np.exp(-z))
        return round(max(1.0, min(99.0, p * 100)), 2)
