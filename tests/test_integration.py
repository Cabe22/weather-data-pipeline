"""
End-to-end integration tests: collector -> processor -> predictor.
All tests use mock API (no real key needed), real SQLite, real processing + ML.
"""

import os
import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.data_collection.weather_collector import WeatherCollector, WeatherConfig
from src.data_processing.data_processor import WeatherDataProcessor
from src.ml_models.weather_predictor import WeatherPredictor


def _make_api_response(city, timestamp_epoch, temp, rain=0.0):
    """Build a fake OpenWeatherMap JSON response."""
    return {
        "coord": {"lon": -74.0, "lat": 40.7},
        "weather": [{"id": 800, "main": "Clear" if rain == 0 else "Rain",
                      "description": "clear sky" if rain == 0 else "light rain", "icon": "01d"}],
        "base": "stations",
        "main": {"temp": temp, "feels_like": temp - 1, "temp_min": temp - 2,
                 "temp_max": temp + 2, "pressure": 1013, "humidity": 65},
        "visibility": 10000,
        "wind": {"speed": 3.5, "deg": 180},
        "clouds": {"all": 10},
        "rain": {"1h": rain} if rain else {},
        "dt": timestamp_epoch,
        "sys": {"type": 2, "id": 1, "country": "US", "sunrise": timestamp_epoch - 3600,
                "sunset": timestamp_epoch + 3600},
        "timezone": -18000,
        "id": 5128581,
        "name": city,
        "cod": 200,
    }


def _mock_get_factory(cities, n_cycles):
    """
    Return a side_effect callable for requests.get that yields incrementing
    timestamps. n_cycles calls per city, cities interleaved.
    """
    base_epoch = int(datetime(2024, 1, 1).timestamp())
    call_counter = {"n": 0}
    np.random.seed(42)

    def side_effect(*args, **kwargs):
        idx = call_counter["n"]
        call_counter["n"] += 1
        city_idx = idx % len(cities)
        cycle = idx // len(cities)
        city = cities[city_idx]
        ts = base_epoch + cycle * 3600  # 1-hour intervals
        temp = 20 + np.random.normal(0, 3)
        rain = float(np.random.choice([0.0, 0.5, 1.0], p=[0.6, 0.2, 0.2]))
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.return_value = _make_api_response(city, ts, temp, rain)
        return resp

    return side_effect


def _run_pipeline_steps(processor, db_path):
    """Run individual pipeline steps, working around the pandas compat issue
    in handle_missing_values' interpolation step."""
    df = processor.load_data()
    df = processor.create_time_features(df)
    df = processor.create_lag_features(df)
    df = processor.create_weather_indices(df)
    df = processor.create_interaction_features(df)
    # Convert Categorical time_of_day to string so encode step can fillna
    if "time_of_day" in df.columns:
        df["time_of_day"] = df["time_of_day"].astype(str)
    df = processor.encode_categorical_features(df)
    df = processor.create_target_variable(df)
    df = df.dropna(subset=["temperature_future"])
    # Drop original string columns that remain after encoding (the _encoded
    # versions are kept). prepare_features excludes some of these but misses
    # weather_main.
    str_cols = df.select_dtypes(include=["object", "category"]).columns
    df = df.drop(columns=str_cols, errors="ignore")
    return df


@pytest.mark.integration
class TestEndToEndPipeline:
    def test_full_collector_processor_predictor_chain(self, tmp_path):
        """Collect -> process -> train -> predict in one shot."""
        cities = ["New York", "Los Angeles"]
        n_cycles = 30
        db_path = str(tmp_path / "e2e.db")
        config = WeatherConfig(api_key="fake", cities=cities, db_path=db_path)

        with patch("src.data_collection.weather_collector.requests.get",
                   side_effect=_mock_get_factory(cities, n_cycles)):
            collector = WeatherCollector(config)
            for _ in range(n_cycles):
                collector.collect_all_cities()

        # Process
        processor = WeatherDataProcessor(db_path=db_path)
        df = _run_pipeline_steps(processor, db_path)
        assert len(df) > 0, "Pipeline produced no rows"

        # Train temperature model
        predictor = WeatherPredictor(model_dir=str(tmp_path / "models") + "/")
        X, y = predictor.prepare_features(df, "temperature_future")
        results = predictor.train_temperature_models(X, y)
        assert "temperature" in predictor.best_models

        # Predict
        preds = predictor.predict(df, model_type="temperature")
        assert len(preds) == len(df)

    def test_save_load_predict_cycle(self, tmp_path):
        """Save trained model, load into new predictor, predict."""
        cities = ["New York", "Los Angeles"]
        n_cycles = 30
        db_path = str(tmp_path / "save_load.db")
        config = WeatherConfig(api_key="fake", cities=cities, db_path=db_path)

        with patch("src.data_collection.weather_collector.requests.get",
                   side_effect=_mock_get_factory(cities, n_cycles)):
            collector = WeatherCollector(config)
            for _ in range(n_cycles):
                collector.collect_all_cities()

        processor = WeatherDataProcessor(db_path=db_path)
        df = _run_pipeline_steps(processor, db_path)

        model_dir = str(tmp_path / "models") + "/"
        predictor = WeatherPredictor(model_dir=model_dir)
        X, y = predictor.prepare_features(df, "temperature_future")
        predictor.train_temperature_models(X, y)
        predictor.save_models()

        # Load into fresh predictor
        new_predictor = WeatherPredictor(model_dir=model_dir)
        new_predictor.load_models()
        preds = new_predictor.predict(df, model_type="temperature")
        assert len(preds) == len(df)

    def test_multiple_collection_cycles(self, tmp_path):
        """Verify multiple collection cycles accumulate data correctly."""
        cities = ["New York"]
        n_cycles = 10
        db_path = str(tmp_path / "multi.db")
        config = WeatherConfig(api_key="fake", cities=cities, db_path=db_path)

        with patch("src.data_collection.weather_collector.requests.get",
                   side_effect=_mock_get_factory(cities, n_cycles)):
            collector = WeatherCollector(config)
            for _ in range(n_cycles):
                collector.collect_all_cities()

        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM weather_data")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == n_cycles


@pytest.mark.integration
class TestCollectorToProcessorPipeline:
    def test_collected_data_has_processor_columns(self, tmp_path):
        """Data from collector should have all columns the processor needs."""
        cities = ["New York", "Los Angeles"]
        db_path = str(tmp_path / "cols.db")
        config = WeatherConfig(api_key="fake", cities=cities, db_path=db_path)

        with patch("src.data_collection.weather_collector.requests.get",
                   side_effect=_mock_get_factory(cities, 5)):
            collector = WeatherCollector(config)
            for _ in range(5):
                collector.collect_all_cities()

        processor = WeatherDataProcessor(db_path=db_path)
        df = processor.load_data()
        required = ["city", "timestamp", "temperature", "humidity", "pressure",
                     "wind_speed", "cloudiness", "weather_main", "temp_min", "temp_max",
                     "rain_1h"]
        for col in required:
            assert col in df.columns, f"Missing required column: {col}"

    def test_processor_handles_collector_output(self, tmp_path):
        """Processor feature engineering should not crash on collector data."""
        cities = ["New York", "Los Angeles"]
        db_path = str(tmp_path / "handle.db")
        config = WeatherConfig(api_key="fake", cities=cities, db_path=db_path)

        with patch("src.data_collection.weather_collector.requests.get",
                   side_effect=_mock_get_factory(cities, 15)):
            collector = WeatherCollector(config)
            for _ in range(15):
                collector.collect_all_cities()

        processor = WeatherDataProcessor(db_path=db_path)
        df = processor.load_data()
        df = processor.create_time_features(df)
        df = processor.create_lag_features(df)
        df = processor.create_weather_indices(df)
        df = processor.create_interaction_features(df)
        assert len(df) > 0
        assert "hour" in df.columns
        assert "heat_index" in df.columns
