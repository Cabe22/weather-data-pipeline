"""
Tests for weather data collection, storage, and retrieval.
Unit tests use mocks and tmp_path; integration tests use the live API.
"""

import os
import sqlite3
import pytest

from src.data_collection.weather_collector import WeatherCollector, WeatherConfig


class TestDatabaseSetup:
    def test_database_file_created(self, weather_config):
        WeatherCollector(weather_config)
        assert os.path.exists(weather_config.db_path)

    def test_weather_data_table_exists(self, weather_config):
        WeatherCollector(weather_config)
        conn = sqlite3.connect(weather_config.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        assert "weather_data" in tables

    def test_required_columns_present(self, weather_config):
        WeatherCollector(weather_config)
        conn = sqlite3.connect(weather_config.db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(weather_data);")
        columns = [col[1] for col in cursor.fetchall()]
        conn.close()

        required = ["city", "temperature", "humidity", "pressure", "wind_speed"]
        for col in required:
            assert col in columns, f"Missing column: {col}"


class TestParseWeatherData:
    def test_all_expected_keys_present(self, weather_config, sample_raw_api_response):
        collector = WeatherCollector(weather_config)
        parsed = collector.parse_weather_data(sample_raw_api_response)

        expected_keys = [
            "city", "country", "timestamp", "temperature", "feels_like",
            "temp_min", "temp_max", "pressure", "humidity", "wind_speed",
            "wind_deg", "cloudiness", "weather_main", "weather_description",
            "visibility", "rain_1h", "snow_1h", "lat", "lon", "timezone",
        ]
        for key in expected_keys:
            assert key in parsed, f"Missing key: {key}"

    def test_correct_values_extracted(self, weather_config, sample_raw_api_response):
        collector = WeatherCollector(weather_config)
        parsed = collector.parse_weather_data(sample_raw_api_response)

        assert parsed["city"] == "New York"
        assert parsed["country"] == "US"
        assert parsed["temperature"] == 22.5
        assert parsed["humidity"] == 65
        assert parsed["pressure"] == 1013
        assert parsed["wind_speed"] == 3.5
        assert parsed["weather_main"] == "Clear"


class TestStoreAndRetrieve:
    def test_store_inserts_row(self, mock_collector, sample_raw_api_response):
        collector, _ = mock_collector
        parsed = collector.parse_weather_data(sample_raw_api_response)
        collector.store_weather_data(parsed)

        conn = sqlite3.connect(collector.config.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM weather_data")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 1

    def test_collect_all_cities_stores_one_per_city(self, mock_collector):
        collector, _ = mock_collector
        collector.collect_all_cities()

        conn = sqlite3.connect(collector.config.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM weather_data")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == len(collector.config.cities)


@pytest.mark.integration
class TestDataCollectionIntegration:
    def test_live_api_connection(self):
        api_key = os.getenv("OPENWEATHER_API_KEY")
        if not api_key:
            pytest.skip("OPENWEATHER_API_KEY not set")

        config = WeatherConfig(api_key=api_key, cities=["New York"])
        collector = WeatherCollector(config)
        data = collector.fetch_weather_data("New York")

        assert data is not None
        assert data["main"]["temp"] is not None

    def test_live_database_setup(self, tmp_path):
        config = WeatherConfig(
            api_key="dummy_key",
            cities=["New York"],
            db_path=str(tmp_path / "live_test.db"),
        )
        collector = WeatherCollector(config)
        assert os.path.exists(config.db_path)

        conn = sqlite3.connect(config.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        assert "weather_data" in tables

    def test_live_data_collection(self, tmp_path):
        api_key = os.getenv("OPENWEATHER_API_KEY")
        if not api_key:
            pytest.skip("OPENWEATHER_API_KEY not set")

        config = WeatherConfig(
            api_key=api_key,
            cities=["New York", "Los Angeles"],
            db_path=str(tmp_path / "collect_test.db"),
        )
        collector = WeatherCollector(config)
        collector.collect_all_cities()

        conn = sqlite3.connect(config.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM weather_data")
        count = cursor.fetchone()[0]
        conn.close()

        assert count >= 2
