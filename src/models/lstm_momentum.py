"""LSTM de momentum sobre secuencias de goles recientes."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from ..utils import extract_score, is_btts
from .base import BaseBTTSModel


class _LSTMNet(nn.Module):
    def __init__(self, input_size: int = 4, hidden: int = 32) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)


class LSTMMomentum(BaseBTTSModel):
    name = "lstm_momentum"

    def __init__(self, seq_len: int = 8, epochs: int = 40) -> None:
        self.seq_len = seq_len
        self.epochs = epochs
        self.device = torch.device("cpu")

    def _build_sequence(
        self, matches: List[Dict[str, Any]], team_id: str
    ) -> np.ndarray:
        """Secuencia [goles_a favor, goles_en_contra, btts, resultado]."""
        seq = []
        for m in matches[: self.seq_len * 2]:
            h, a = extract_score(m)
            if h is None or a is None:
                continue
            home_id = (m.get("home_team") or {}).get("id")
            if home_id == team_id:
                gf, ga = h, a
            else:
                gf, ga = a, h
            btts = 1.0 if (gf > 0 and ga > 0) else 0.0
            result = 1.0 if gf > ga else (0.5 if gf == ga else 0.0)
            seq.append([gf, ga, btts, result])
        seq = seq[: self.seq_len]
        while len(seq) < self.seq_len:
            seq.insert(0, [1.0, 1.0, 0.5, 0.5])  # padding neutro
        return np.array(seq, dtype=np.float32)

    def predict(
        self,
        home_matches: List[Dict[str, Any]],
        away_matches: List[Dict[str, Any]],
    ) -> float:
        if not home_matches or not away_matches:
            return 0.0

        home_id = (home_matches[0].get("home_team") or {}).get("id") or ""
        away_id = (away_matches[0].get("home_team") or {}).get("id") or ""

        home_seq = self._build_sequence(home_matches, home_id)
        away_seq = self._build_sequence(away_matches, away_id)

        # Feature vector combinado (promedio de ambas secuencias)
        combined = (home_seq + away_seq) / 2.0
        X = torch.tensor(combined[np.newaxis, ...], dtype=torch.float32)

        model = _LSTMNet().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCELoss()

        # Target débil basado en tasa histórica (self-supervised rápido)
        target_rate = np.mean([is_btts(m) for m in (home_matches + away_matches)[:12]])
        y = torch.tensor([[target_rate]], dtype=torch.float32)

        model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            prob = model(X).item()
        return round(max(0.0, min(100.0, prob * 100)), 2)
