# src/models/catboost_model.py
"""CatBoost con features de forma. Se entrena UNA vez por ejecución."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .base import BaseBTTSModel
from .xgboost_model import _team_features


class CatBoostBTTS(BaseBTTSModel):
    name = "catboost"

    def __init__(self, n: int = 15) -> None:
        self.n = n
        self._model = None
        self._trained = False

    def _train_once(self) -> None:
        try:
            from catboost import CatBoostClassifier

            rng = np.random.default_rng(7)
            n_samples = 400
            X = rng.normal(0, 1, (n_samples, 20))
            X[:, 0] += rng.normal(55, 15, n_samples)
            logits = 0.035 * (X[:, 0] - 55) + 0.25 * (X[:, 3] - X[:, 4]) \
                - 0.02 * (X[:, 6] - 20)
            prob = 1 / (1 + np.exp(-logits))
            y = (rng.random(n_samples) < np.clip(prob, 0.15, 0.85)).astype(int)

            self._model = CatBoostClassifier(
                iterations=120, depth=5, learning_rate=0.08,
                loss_function="Logloss", random_seed=7, verbose=0,
            )
            self._model.fit(X, y)
            self._trained = True
        except Exception as exc:
            logger = __import__("logging").getLogger(__name__)
            logger.warning("CatBoost no disponible (%s); usando fallback", exc)
            self._model = None
            self._trained = True

    def predict(self, home_matches, away_matches) -> float:
        if not self._trained:
            self._train_once()

        fh = _team_features(home_matches[: self.n])
        fa = _team_features(away_matches[: self.n])
        x = np.array(fh + fa).reshape(1, -1)

        if self._model is not None:
            p = float(self._model.predict_proba(x)[0][1])
        else:
            z = (
                0.035 * ((fh[0] + fa[0]) / 2 - 55)
                + 0.25 * (fh[3] + fa[3] - fh[4] - fa[4])
                - 0.02 * ((fh[6] + fa[6]) / 2 - 20)
            )
            p = 1 / (1 + np.exp(-z))
        return round(max(1.0, min(99.0, p * 100)), 2)
