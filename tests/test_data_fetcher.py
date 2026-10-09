from __future__ import annotations

from unittest.mock import MagicMock

from src.data_fetcher import DataFetcher


def test_data_fetcher_accepts_date_range_contract() -> None:
    client = MagicMock()
    client.get.return_value = {"data": [], "meta": {"total_pages": 1}}

    fetcher = DataFetcher(client)
    result = fetcher.get_upcoming_matches(date_from="2026-10-09", date_to="2026-10-10")

    assert result == []
    client.get.assert_called_once()
    assert client.get.call_args.kwargs["params"]["date_from"] == "2026-10-09"
    assert client.get.call_args.kwargs["params"]["date_to"] == "2026-10-10"
