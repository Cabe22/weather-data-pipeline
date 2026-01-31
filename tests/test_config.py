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

    def test_custom_db_path_preserved(self):
        config = WeatherConfig(db_path="/tmp/custom.db")
        assert config.db_path == "/tmp/custom.db"

    def test_default_update_interval(self):
        config = WeatherConfig()
        assert config.update_interval == 1800


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

    def test_custom_model_dir(self):
        config = ModelConfig(model_dir="/tmp/my_models/")
        assert config.model_dir == "/tmp/my_models/"

    def test_custom_test_size(self):
        config = ModelConfig(test_size=0.3)
        assert config.test_size == 0.3
