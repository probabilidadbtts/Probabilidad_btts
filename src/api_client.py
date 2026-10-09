"""Cliente HTTP robusto para TheStatsAPI."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.thestatsapi.com/api"
REQUEST_TIMEOUT = 30
MAX_RETRIES = 4
RETRY_BACKOFF = 1.8
RATE_LIMIT_SLEEP = 0.30


class StatsAPIClient:
    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.environ.get("THESTATSAPI_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "THESTATSAPI_KEY no está definida. "
                "Exporta la variable o usa un archivo .env"
            )
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "User-Agent": "BTTS-Predictor/2.0 (production)",
            }
        )

    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{BASE_URL}{endpoint}"
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.session.get(
                    url, params=params or {}, timeout=REQUEST_TIMEOUT
                )
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", RETRY_BACKOFF * attempt))
                    logger.warning("Rate-limit 429 → esperando %ss", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                time.sleep(RATE_LIMIT_SLEEP)
                return resp.json()
            except requests.RequestException as exc:
                logger.warning(
                    "Intento %d/%d fallido (%s): %s",
                    attempt, MAX_RETRIES, endpoint, exc
                )
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(RETRY_BACKOFF * attempt)
        raise RuntimeError(f"Fallo definitivo en {endpoint}")
