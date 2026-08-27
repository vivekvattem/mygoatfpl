"""Reusable client for the official Fantasy Premier League API."""

from typing import Any

import requests

from .config import BASE_API_URL, REQUEST_TIMEOUT


class FPLAPIError(RuntimeError):
    """Raised when the official FPL API cannot be queried successfully."""


class FPLAPIClient:
    """Small, mockable HTTP client for the public FPL endpoints."""

    def __init__(
        self,
        base_url: str = BASE_API_URL,
        timeout: float = REQUEST_TIMEOUT,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def _get_json(self, endpoint: str) -> Any:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            raise FPLAPIError(f"Failed to fetch valid FPL data from {url}: {exc}") from exc

    def get_bootstrap_static(self) -> dict[str, Any]:
        """Return current players, teams, positions, and Gameweek metadata."""
        payload = self._get_json("bootstrap-static/")
        if not isinstance(payload, dict):
            raise FPLAPIError("FPL bootstrap response was not a JSON object")
        return payload

    def get_fixtures(self) -> list[dict[str, Any]]:
        """Return the current season fixture list."""
        payload = self._get_json("fixtures/")
        if not isinstance(payload, list):
            raise FPLAPIError("FPL fixtures response was not a JSON list")
        return payload
