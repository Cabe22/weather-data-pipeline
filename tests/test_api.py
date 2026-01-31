"""
Tests for weather API integration.
Unit tests use mocks; integration tests hit the live API.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from requests.exceptions import HTTPError, Timeout

from src.data_collection.weather_collector import WeatherCollector, WeatherConfig


class TestAPIUnit:
    def test_fetch_weather_data_success(self, mock_collector):
        collector, mock_get = mock_collector
        result = collector.fetch_weather_data("New York")

        assert result is not None
        assert result["name"] == "New York"
        assert result["main"]["temp"] == 22.5
        mock_get.assert_called_once()

    def test_fetch_weather_data_http_error(self, weather_config):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = HTTPError("404 Not Found")

        with patch(
            "src.data_collection.weather_collector.requests.get",
            return_value=mock_response,
        ):
            collector = WeatherCollector(weather_config)
            result = collector.fetch_weather_data("InvalidCity")

        assert result is None

    def test_fetch_weather_data_timeout(self, weather_config):
        with patch(
            "src.data_collection.weather_collector.requests.get",
            side_effect=Timeout("Connection timed out"),
        ):
            collector = WeatherCollector(weather_config)
            result = collector.fetch_weather_data("New York")

        assert result is None


@pytest.mark.integration
class TestAPIIntegration:
    def test_live_api_connection(self):
        api_key = os.getenv("OPENWEATHER_API_KEY")
        if not api_key:
            pytest.skip("OPENWEATHER_API_KEY not set")

        config = WeatherConfig(api_key=api_key, cities=["New York"])
        collector = WeatherCollector(config)
        data = collector.fetch_weather_data("New York")

        assert data is not None
        assert data["name"] == "New York"
        assert "main" in data
        assert "temp" in data["main"]
