from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Settings:
    """Configuración central del proyecto."""

    date_from: str
    date_to: str
    min_pct: float = 80.0
    top_n: int = 20
    log_level: str = "INFO"

    @classmethod
    def from_dict(cls, values: Optional[Dict[str, Any]] = None) -> "Settings":
        payload = values or {}
        today = datetime.now(timezone.utc).date().isoformat()
        date_from = _normalize_date(payload.get("date_from"), today)
        date_to = _normalize_date(payload.get("date_to"), date_from)
        min_pct = float(payload.get("min_pct", os.environ.get("MIN_PCT", "80")))
        top_n = int(payload.get("top_n", os.environ.get("TOP_N", "20")))
        log_level = str(payload.get("log_level", os.environ.get("LOG_LEVEL", "INFO"))).upper()
        return cls(
            date_from=date_from,
            date_to=date_to,
            min_pct=min_pct,
            top_n=top_n,
            log_level=log_level,
        )


def _normalize_date(value: Optional[str], default: str) -> str:
    clean = (value or "").strip()
    return clean if clean else default
