"""Utilidades compartidas."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def extract_score(match: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    score = match.get("score") or {}
    final = score.get("final_score") or {}
    home = final.get("home") if final.get("home") is not None else score.get("home")
    away = final.get("away") if final.get("away") is not None else score.get("away")
    try:
        return (int(home) if home is not None else None,
                int(away) if away is not None else None)
    except (TypeError, ValueError):
        return None, None


def is_btts(match: Dict[str, Any]) -> bool:
    h, a = extract_score(match)
    return h is not None and a is not None and h > 0 and a > 0


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0
