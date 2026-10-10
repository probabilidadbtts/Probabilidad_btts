"""Cliente HTTP robusto para TheStatsAPI con reintentos y manejo de rate limits."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

BASE_URL = "https://api.thestatsapi.com/api"
REQUEST_TIMEOUT = 30
MAX_RETRIES = 4
RETRY_BACKOFF = 1.8
RATE_LIMIT_SLEEP = 0.30


class StatsAPIClient:
    """Cliente de TheStatsAPI con manejo de reintentos y rate limiting."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.environ.get("THESTATSAPI_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "THESTATSAPI_KEY no está definida. "
                "Exporta la variable o usa un archivo .env"
            )
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Crea una sesión con reintentos automáticos."""
        session = requests.Session()
        session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "User-Agent": "BTTS-Predictor/2.0 (production)",
            }
        )
        retry_strategy = Retry(
            total=MAX_RETRIES,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            backoff_factor=RETRY_BACKOFF,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    @staticmethod
    def _normalize_endpoint(endpoint: str) -> str:
        """Normaliza endpoints para evitar duplicar el prefijo /api."""
        if not endpoint:
            return "/"
        endpoint = endpoint.strip()
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        if endpoint.startswith("/api"):
            endpoint = endpoint[len("/api") :]
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        return endpoint

    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Realiza una solicitud GET con manejo de errores y rate limits."""
        normalized = self._normalize_endpoint(endpoint)
        url = f"{BASE_URL}{normalized}"
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.session.get(
                    url,
                    params=params or {},
                    timeout=REQUEST_TIMEOUT,
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
                    attempt,
                    MAX_RETRIES,
                    normalized,
                    exc,
                )
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(RETRY_BACKOFF * attempt)
        raise RuntimeError(f"Fallo definitivo en {normalized}")
