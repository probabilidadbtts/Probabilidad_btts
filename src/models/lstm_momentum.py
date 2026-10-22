# src/models/lstm_momentum.py
"""LSTM de momentum sobre secuencias recientes. Entrena una vez por ejecución.

Si TensorFlow no está instalado, usa un fallback de media móvil (mismo
contrato de predict) para no romper el ensemble en entornos ligeros.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..utils import extract_score
from .base import BaseBTTSModel

SEQ_LEN = 8


def _sequence(matches: List[Dict[str, Any]], seq_len: int) -> np.ndarray:
    """Secuencia [gf, ga, btts] por partido, padded con ceros al inicio."""
    rows = []
    for m in matches[:seq_len]:
        h, a = extract_score(m)
        if h is None or a is None:
            continue
        rows.append([float(h), float(a), 1.0 if h > 0 and a > 0 else 0.0])
    rows = rows[::-1]  # cronológico
    while len(rows) < seq_len:
        rows.insert(0, [0.0, 0.0, 0.0])
    return np.array(rows[-seq_len:], dtype=np.float32)


class LSTMMomentum(BaseBTTSModel):
    name = "lstm_momentum"

    def __init__(self, seq_len: int = SEQ_LEN, epochs: int = 35) -> None:
        self.seq_len = seq_len
        self.epochs = epochs
        self._model = None
        self._trained = False

    def _train_once(self) -> None:
        try:
            import tensorflow as tf  # noqa: F401
            from tensorflow.keras import layers, models

            rng = np.random.default_rng(123)
            n = 300
            X = rng.normal(0, 1, (n, self.seq_len, 3)).astype(np.float32)
            X[:, :, 0] = np.clip(X[:, :, 0] * 1.2 + 1.4, 0, 6)
            X[:, :, 1] = np.clip(X[:, :, 1] * 1.2 + 1.2, 0, 6)
            X[:, :, 2] = (rng.random((n, self.seq_len)) < 0.55).astype(np.float32)
            seq_btts = X[:, :, 2].mean(axis=1)
            y = (rng.random(n) < np.clip(0.3 + 0.5 * seq_btts, 0.15, 0.85)).astype(int)

            self._model = models.Sequential([
                layers.Input(shape=(self.seq_len, 3)),
                layers.LSTM(16),
                layers.Dense(8, activation="relu"),
                layers.Dense(1, activation="sigmoid"),
            ])
            self._model.compile(optimizer="adam", loss="binary_crossentropy")
            self._model.fit(X, y, epochs=self.epochs, batch_size=32, verbose=0)
            self._trained = True
        except Exception as exc:
            logger = __import__("logging").getLogger(__name__)
            logger.warning(
                "TensorFlow no disponible (%s); LSTM usa fallback de media móvil",
                exc,
            )
            self._model = None
            self._trained = True

    def predict(self, home_matches, away_matches) -> float:
        if not self._trained:
            self._train_once()

        sh = _sequence(home_matches, self.seq_len)
        sa = _sequence(away_matches, self.seq_len)

        if self._model is not None:
            x = np.stack([sh, sa]).reshape(1, self.seq_len, 3)
            p = float(self._model.predict(x, verbose=0)[0][0])
        else:
            # Fallback: media del indicador BTTS reciente combinada home/away
            p = 0.5 * (sh[:, 2].mean() + sa[:, 2].mean())
            p = float(np.clip(p, 0.05, 0.95))
        return round(p * 100, 2)
