"""
Cliente HTTP robusto para TheStatsAPI.
- Rate limiting proactivo (token bucket)
- Reintento con backoff exponencial + jitter (sin doble retry)
- Respeta Retry-After del servidor
- Caché local para evitar re-descargas
- Tracking de rate-limit headers
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.thestatsapi.com/api"
REQUEST_TIMEOUT = 30
MAX_RETRIES = 5
BACKOFF_BASE = 2.0       # segundos
BACKOFF_MAX = 60.0       # tope de espera
JITTER = 0.2             # ±20%


# ─── Rate Limiter (token bucket simplificado) ────────────────────────────────

class RateLimiter:
    """
    Token bucket: garantiza no superar `max_rpm` requests por minuto.
    Starter plan TheStatsAPI = 120 req/min.
    """

    def __init__(self, max_rpm: int = 120):
        self.max_rpm = max_rpm
        self.min_interval = 60.0 / max_rpm  # intervalo mínimo entre requests
        self._last_request: float = 0.0

    def wait(self) -> None:
        """Bloquea hasta que se pueda hacer el siguiente request."""
        now = time.monotonic()
        elapsed = now - self._last_request
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)
        self._last_request = time.monotonic()


# ─── Cliente ─────────────────────────────────────────────────────────────────

class StatsAPIClient:
    """
    Cliente de TheStatsAPI con:
    - Rate limiting proactivo (evita 429 antes de que ocurran)
    - Backoff exponencial con jitter (solo en el bucle manual, NO urllib3)
    - Respeta Retry-After
    - Caché local (opcional)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_rpm: int = 120,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
    ) -> None:
        self.api_key = api_key or os.environ.get("THESTATSAPI_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "THESTATSAPI_KEY no está definida. "
                "Exporta la variable o usa un archivo .env"
            )

        self.rate_limiter = RateLimiter(max_rpm=max_rpm)
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir or os.environ.get("CACHE_DIR", "./cache"))
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.session = self._create_session()
        self._requests_made = 0

    def _create_session(self) -> requests.Session:
        """
        Sesión SIN Retry de urllib3.
        Todo el manejo de reintentos se hace en el bucle manual
        para tener control total sobre Retry-After y jitter.
        """
        session = requests.Session()
        session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "User-Agent": "BTTS-Predictor/2.0 (production)",
        })
        # NO montar HTTPAdapter con Retry → evitar doble reintento
        return session

    # ─── Caché ───────────────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(endpoint: str, params: Optional[Dict]) -> str:
        raw = endpoint + json.dumps(params or {}, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _cache_get(self, key: str) -> Optional[Dict]:
        if not self.use_cache:
            return None
        path = self._cache_path(key)
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return None

    def _cache_set(self, key: str, data: Dict) -> None:
        if not self.use_cache:
            return
        with open(self._cache_path(key), "w") as f:
            json.dump(data, f)

    # ─── Normalización ───────────────────────────────────────────────────────

    @staticmethod
    def _normalize_endpoint(endpoint: str) -> str:
        if not endpoint:
            return "/"
        endpoint = endpoint.strip()
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        if endpoint.startswith("/api"):
            endpoint = endpoint[len("/api"):]
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        return endpoint

    # ─── Core GET ────────────────────────────────────────────────────────────

    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        use_cache: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        GET con rate-limiting proactivo, backoff+jitter, Retry-After y caché.
        """
        normalized = self._normalize_endpoint(endpoint)
        url = f"{BASE_URL}{normalized}"
        cache_enabled = self.use_cache if use_cache is None else use_cache
        ck = self._cache_key(normalized, params)

        # 1. Caché
        if cache_enabled:
            cached = self._cache_get(ck)
            if cached is not None:
                logger.debug("CACHE HIT: %s", normalized)
                return cached

        # 2. Bucle de reintentos (único mecanismo)
        last_exc: Optional[Exception] = None

        for attempt in range(1, MAX_RETRIES + 1):
            # Rate limit proactivo
            self.rate_limiter.wait()

            try:
                resp = self.session.get(url, params=params or {}, timeout=REQUEST_TIMEOUT)
                self._requests_made += 1

                # ── 429: Rate limit ──
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        wait = float(retry_after)
                    else:
                        wait = min(BACKOFF_BASE ** attempt, BACKOFF_MAX)
                    # Jitter ±20%
                    wait *= (1 + random.uniform(-JITTER, JITTER))
                    logger.warning(
                        "429 Rate limit → esperando %.1fs (attempt %d/%d)",
                        wait, attempt, MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue

                # ── 5xx: Error transitorio del servidor ──
                if resp.status_code in (500, 502, 503, 504):
                    wait = min(BACKOFF_BASE ** attempt, BACKOFF_MAX)
                    wait *= (1 + random.uniform(-JITTER, JITTER))
                    logger.warning(
                        "HTTP %d → retry en %.1fs (attempt %d/%d)",
                        resp.status_code, wait, attempt, MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue

                # ── 401: Auth error → NO retry, fail fast ──
                if resp.status_code == 401:
                    logger.error(
                        "401 Unauthorized. Verifica THESTATSAPI_KEY. "
                        "URL: %s", url,
                    )
                    resp.raise_for_status()

                # ── Otros 4xx: No retry ──
                if 400 <= resp.status_code < 500:
                    logger.error("HTTP %d (no retryable): %s", resp.status_code, resp.text[:200])
                    resp.raise_for_status()

                # ── 200: Éxito ──
                data = resp.json()

                # Loggear rate-limit info si el servidor la provee
                remaining = resp.headers.get("X-RateLimit-Remaining")
                if remaining is not None:
                    logger.debug("Rate limit remaining: %s", remaining)

                if cache_enabled:
                    self._cache_set(ck, data)

                return data

            except requests.Timeout:
                last_exc = requests.Timeout(f"Timeout after {REQUEST_TIMEOUT}s: {url}")
                wait = min(BACKOFF_BASE ** attempt, BACKOFF_MAX)
                wait *= (1 + random.uniform(-JITTER, JITTER))
                logger.warning("Timeout → retry en %.1fs (attempt %d/%d)", wait, attempt, MAX_RETRIES)
                time.sleep(wait)

            except requests.ConnectionError as exc:
                last_exc = exc
                wait = min(BACKOFF_BASE ** attempt, BACKOFF_MAX)
                wait *= (1 + random.uniform(-JITTER, JITTER))
                logger.warning("ConnectionError → retry en %.1fs (attempt %d/%d)", wait, attempt, MAX_RETRIES)
                time.sleep(wait)

            except requests.HTTPError:
                raise  # Ya se logueó arriba

            except requests.RequestException as exc:
                last_exc = exc
                logger.warning("RequestException: %s (attempt %d/%d)", exc, attempt, MAX_RETRIES)
                if attempt == MAX_RETRIES:
                    break
                time.sleep(BACKOFF_BASE * attempt)

        raise RuntimeError(
            f"Fallo definitivo en {normalized} after {MAX_RETRIES} attempts. "
            f"Last error: {last_exc}"
        )

    # ─── Paginación ──────────────────────────────────────────────────────────

    def get_all(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        per_page: int = 100,
    ) -> list[Dict[str, Any]]:
        """
        GET paginado: descarga todas las páginas automáticamente.
        """
        params = dict(params or {})
        params["per_page"] = per_page
        all_items: list[Dict] = []
        page = 1

        while True:
            params["page"] = page
            data = self.get(endpoint, params)
            items = data.get("data", [])
            all_items.extend(items)

            meta = data.get("meta", {})
            total_pages = meta.get("total_pages", 1)
            logger.debug("Page %d/%d → %d items (total: %d)", page, total_pages, len(items), len(all_items))

            if page >= total_pages:
                break
            page += 1

        return all_items

    # ─── Context manager ─────────────────────────────────────────────────────

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "StatsAPIClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"StatsAPIClient(requests_made={self._requests_made})"   
