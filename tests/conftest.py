"""
Shared pytest fixtures for weather data pipeline tests.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from configs.config import WeatherConfig
from src.data_collection.weather_collector import WeatherCollector
from src.data_collection.weather_collector import WeatherConfig as CollectorWeatherConfig


@pytest.fixture
def sample_raw_api_response():
    """Dict mimicking a raw OpenWeatherMap API JSON response."""
    return {
        "coord": {"lon": -74.006, "lat": 40.7143},
        "weather": [
            {
                "id": 800,
                "main": "Clear",
                "description": "clear sky",
                "icon": "01d",
            }
        ],
        "base": "stations",
        "main": {
            "temp": 22.5,
            "feels_like": 21.8,
            "temp_min": 20.0,
            "temp_max": 25.0,
            "pressure": 1013,
            "humidity": 65,
        },
        "visibility": 10000,
        "wind": {"speed": 3.5, "deg": 180},
        "clouds": {"all": 10},
        "dt": 1700000000,
        "sys": {
            "type": 2,
            "id": 2039034,
            "country": "US",
            "sunrise": 1699960000,
            "sunset": 1699996000,
        },
        "timezone": -18000,
        "id": 5128581,
        "name": "New York",
        "cod": 200,
    }


@pytest.fixture
def sample_parsed_weather_data():
    """Dict matching the output of WeatherCollector.parse_weather_data()."""
    return {
        "city": "New York",
        "country": "US",
        "timestamp": datetime.fromtimestamp(1700000000),
        "temperature": 22.5,
        "feels_like": 21.8,
        "temp_min": 20.0,
        "temp_max": 25.0,
        "pressure": 1013,
        "humidity": 65,
        "wind_speed": 3.5,
        "wind_deg": 180,
        "cloudiness": 10,
        "weather_main": "Clear",
        "weather_description": "clear sky",
        "visibility": 10000,
        "rain_1h": 0,
        "snow_1h": 0,
        "lat": 40.7143,
        "lon": -74.006,
        "timezone": -18000,
    }


@pytest.fixture
def sample_weather_dataframe():
    """50-row DataFrame with 2 cities x 25 hourly observations."""
    np.random.seed(42)
    cities = ["New York", "Los Angeles"]
    rows_per_city = 25
    rows = []
    base_time = datetime(2024, 1, 1)

    for city in cities:
        base_temp = 22.0 if city == "New York" else 28.0
        for i in range(rows_per_city):
            rows.append(
                {
                    "city": city,
                    "timestamp": base_time + timedelta(hours=i),
                    "temperature": base_temp + np.random.normal(0, 3),
                    "humidity": int(np.clip(65 + np.random.normal(0, 10), 0, 100)),
                    "pressure": int(1013 + np.random.normal(0, 5)),
                    "wind_speed": max(0, 3.5 + np.random.normal(0, 1.5)),
                    "cloudiness": int(np.clip(np.random.uniform(0, 100), 0, 100)),
                    "weather_main": np.random.choice(
                        ["Clear", "Clouds", "Rain", "Snow"]
                    ),
                }
            )

    return pd.DataFrame(rows)


@pytest.fixture
def weather_config(tmp_path):
    """WeatherConfig instance pointing to a temporary database."""
    return CollectorWeatherConfig(
        api_key="fake_api_key_for_testing",
        cities=["New York", "Los Angeles"],
        db_path=str(tmp_path / "test_weather.db"),
    )


@pytest.fixture
def mock_collector(weather_config, sample_raw_api_response):
    """WeatherCollector with requests.get patched to return sample data."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = sample_raw_api_response
    mock_response.raise_for_status.return_value = None

    with patch(
        "src.data_collection.weather_collector.requests.get",
        return_value=mock_response,
    ) as mock_get:
        collector = WeatherCollector(weather_config)
        yield collector, mock_get
