"""
Tests for configs/config.py dataclasses.
"""

from configs.config import WeatherConfig, DatabaseConfig, ModelConfig


class TestWeatherConfig:
    def test_default_cities_populated(self):
        config = WeatherConfig()
        assert len(config.cities) == 10

    def test_default_cities_includes_new_york(self):
        config = WeatherConfig()
        assert "New York" in config.cities

    def test_custom_cities_preserved(self):
        config = WeatherConfig(cities=["London", "Paris"])
        assert config.cities == ["London", "Paris"]

    def test_default_db_path(self):
        config = WeatherConfig()
        assert config.db_path == "data/weather.db"


class TestDatabaseConfig:
    def test_default_db_path(self):
        config = DatabaseConfig()
        assert config.db_path == "data/weather.db"

    def test_default_backup_path(self):
        config = DatabaseConfig()
        assert config.backup_path == "data/backups/"


class TestModelConfig:
    def test_default_model_dir(self):
        config = ModelConfig()
        assert config.model_dir == "models/"

    def test_default_random_state(self):
        config = ModelConfig()
        assert config.random_state == 42

    def test_default_test_size(self):
        config = ModelConfig()
        assert config.test_size == 0.2
