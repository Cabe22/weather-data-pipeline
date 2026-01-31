"""
Shared pytest fixtures for weather data pipeline tests.
"""

import pytest
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from configs.config import WeatherConfig
from src.data_collection.weather_collector import WeatherCollector
from src.data_collection.weather_collector import WeatherConfig as CollectorWeatherConfig
from src.data_processing.data_processor import WeatherDataProcessor
from src.ml_models.weather_predictor import WeatherPredictor


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


@pytest.fixture
def processor_weather_dataframe():
    """100-row DataFrame (2 cities x 50 hourly obs) with all columns the processor needs."""
    np.random.seed(42)
    cities = ["New York", "Los Angeles"]
    rows_per_city = 50
    rows = []
    base_time = datetime(2024, 1, 1)

    weather_options = ["Clear", "Clouds", "Rain", "Snow"]

    for city in cities:
        base_temp = 22.0 if city == "New York" else 28.0
        for i in range(rows_per_city):
            temp = base_temp + np.random.normal(0, 3)
            rain = np.random.choice([0.0, 0.5, 1.0, 2.0], p=[0.7, 0.1, 0.1, 0.1])
            rows.append({
                "city": city,
                "country": "US",
                "timestamp": base_time + timedelta(hours=i),
                "temperature": temp,
                "feels_like": temp - np.random.uniform(0, 2),
                "temp_min": temp - np.random.uniform(1, 3),
                "temp_max": temp + np.random.uniform(1, 3),
                "humidity": int(np.clip(65 + np.random.normal(0, 10), 0, 100)),
                "pressure": int(1013 + np.random.normal(0, 5)),
                "wind_speed": max(0, 3.5 + np.random.normal(0, 1.5)),
                "wind_deg": int(np.random.uniform(0, 360)),
                "cloudiness": int(np.clip(np.random.uniform(0, 100), 0, 100)),
                "weather_main": np.random.choice(weather_options),
                "weather_description": np.random.choice(
                    ["clear sky", "few clouds", "light rain", "heavy snow"]
                ),
                "visibility": int(np.random.uniform(5000, 10000)),
                "rain_1h": rain,
                "snow_1h": np.random.choice([0.0, 0.0, 0.5], p=[0.8, 0.1, 0.1]),
            })

    return pd.DataFrame(rows)


@pytest.fixture
def processor_instance(tmp_path):
    """WeatherDataProcessor pointing to a temporary database."""
    return WeatherDataProcessor(db_path=str(tmp_path / "test_processor.db"))


@pytest.fixture
def predictor_training_dataframe():
    """60-row synthetic DataFrame with engineered numeric features + targets."""
    np.random.seed(99)
    n = 60
    df = pd.DataFrame({
        "temperature": np.random.normal(20, 5, n),
        "humidity": np.random.uniform(30, 90, n),
        "pressure": np.random.normal(1013, 5, n),
        "wind_speed": np.random.uniform(0, 10, n),
        "cloudiness": np.random.uniform(0, 100, n),
        "hour_sin": np.random.uniform(-1, 1, n),
        "hour_cos": np.random.uniform(-1, 1, n),
        "temperature_lag_1h": np.random.normal(20, 5, n),
        "humidity_lag_1h": np.random.uniform(30, 90, n),
        "temperature_rolling_mean_24h": np.random.normal(20, 3, n),
        "heat_index": np.random.normal(25, 5, n),
        "wind_chill": np.random.normal(18, 4, n),
        "temp_humidity_interaction": np.random.normal(1200, 300, n),
        "city_encoded": np.random.choice([0, 1], n),
        "weather_main_encoded": np.random.choice([0, 1, 2, 3], n),
    })
    # Continuous target correlated with temperature
    df["temperature_future"] = df["temperature"] + np.random.normal(0, 2, n)
    # Binary target with ~50/50 split
    df["will_rain"] = (np.random.rand(n) > 0.5).astype(int)

    return df


@pytest.fixture
def populated_weather_db(tmp_path):
    """SQLite DB seeded with 100 rows matching the collector schema. Returns db_path string."""
    np.random.seed(42)
    db_path = str(tmp_path / "populated.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE weather_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT NOT NULL,
            country TEXT,
            timestamp DATETIME NOT NULL,
            temperature REAL,
            feels_like REAL,
            temp_min REAL,
            temp_max REAL,
            pressure INTEGER,
            humidity INTEGER,
            wind_speed REAL,
            wind_deg INTEGER,
            cloudiness INTEGER,
            weather_main TEXT,
            weather_description TEXT,
            visibility INTEGER,
            rain_1h REAL,
            snow_1h REAL,
            lat REAL,
            lon REAL,
            timezone INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE INDEX idx_city_timestamp ON weather_data(city, timestamp)
    """)

    cities = ["New York", "Los Angeles"]
    base_time = datetime(2024, 1, 1)
    weather_options = ["Clear", "Clouds", "Rain", "Snow"]

    for i in range(100):
        city = cities[i % 2]
        temp = 20.0 + np.random.normal(0, 5)
        cursor.execute(
            """INSERT INTO weather_data
               (city, country, timestamp, temperature, feels_like, temp_min, temp_max,
                pressure, humidity, wind_speed, wind_deg, cloudiness,
                weather_main, weather_description, visibility,
                rain_1h, snow_1h, lat, lon, timezone)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                city, "US",
                (base_time + timedelta(hours=i)).isoformat(),
                temp, temp - 1, temp - 2, temp + 2,
                int(1013 + np.random.normal(0, 5)),
                int(np.clip(65 + np.random.normal(0, 10), 0, 100)),
                max(0, 3.5 + np.random.normal(0, 1.5)),
                int(np.random.uniform(0, 360)),
                int(np.random.uniform(0, 100)),
                np.random.choice(weather_options),
                "clear sky",
                int(np.random.uniform(5000, 10000)),
                np.random.choice([0.0, 0.5, 1.0]),
                0.0,
                40.7 if city == "New York" else 34.0,
                -74.0 if city == "New York" else -118.2,
                -18000 if city == "New York" else -28800,
            ),
        )

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def predictor_instance(tmp_path):
    """WeatherPredictor with model_dir pointing to a temporary directory."""
    model_dir = str(tmp_path / "models") + "/"
    return WeatherPredictor(model_dir=model_dir)
