"""CatBoost con las mismas features que XGBoost."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from catboost import CatBoostClassifier

from ..utils import extract_score, is_btts
from .base import BaseBTTSModel


class CatBoostBTTS(BaseBTTSModel):
    name = "catboost"

    def __init__(self, n: int = 15) -> None:
        self.n = n

    def _features(self, matches: List[Dict[str, Any]], team_id: str) -> List[float]:
        scored, conceded, btts_flags = [], [], []
        for m in matches[: self.n]:
            h, a = extract_score(m)
            if h is None or a is None:
                continue
            home_id = (m.get("home_team") or {}).get("id")
            if home_id == team_id:
                scored.append(h)
                conceded.append(a)
            else:
                scored.append(a)
                conceded.append(h)
            btts_flags.append(1.0 if (scored[-1] > 0 and conceded[-1] > 0) else 0.0)

        if not scored:
            return [1.2, 1.2, 0.5, 0.0, 0.0]

        return [
            float(np.mean(scored)),
            float(np.mean(conceded)),
            float(np.mean(btts_flags)),
            float(np.std(scored)) if len(scored) > 1 else 0.0,
            float(np.mean(scored[-5:])) if len(scored) >= 5 else float(np.mean(scored)),
        ]

    def predict(
        self,
        home_matches: List[Dict[str, Any]],
        away_matches: List[Dict[str, Any]],
    ) -> float:
        if not home_matches or not away_matches:
            return 0.0

        home_id = (home_matches[0].get("home_team") or {}).get("id") or ""
        away_id = (away_matches[0].get("home_team") or {}).get("id") or ""

        hf = self._features(home_matches, home_id)
        af = self._features(away_matches, away_id)
        X = np.array([hf + af])

        X_train, y_train = [], []
        for matches, tid in [(home_matches, home_id), (away_matches, away_id)]:
            for i in range(5, min(len(matches), self.n)):
                feat = self._features(matches[i:], tid)
                label = 1 if is_btts(matches[i - 1]) else 0
                X_train.append(feat + feat)
                y_train.append(label)

        if len(X_train) < 6:
            rate = np.mean([is_btts(m) for m in (home_matches + away_matches)[:10]])
            return round(rate * 100, 2)

        model = CatBoostClassifier(
            iterations=80,
            depth=3,
            learning_rate=0.1,
            verbose=0,
            loss_function="Logloss",
        )
        model.fit(np.array(X_train), np.array(y_train))
        proba = model.predict_proba(X)[0][1]
        return round(max(0.0, min(100.0, proba * 100)), 2)
