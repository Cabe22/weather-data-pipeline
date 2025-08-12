"""
Configuration settings for the weather data pipeline
"""

import os
from dataclasses import dataclass
from typing import List

@dataclass
class WeatherConfig:
    """Configuration for weather data collection"""
    api_key: str = os.getenv("OPENWEATHER_API_KEY")
    cities: List[str] = None
    update_interval: int = 1800  # 30 minutes
    db_path: str = "data/weather.db"
    
    def __post_init__(self):
        if self.cities is None:
            self.cities = [
                "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
                "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"
            ]

@dataclass 
class DatabaseConfig:
    """Database configuration"""
    db_path: str = "data/weather.db"
    backup_path: str = "data/backups/"

@dataclass
class ModelConfig:
    """ML model configuration"""
    model_dir: str = "models/"
    random_state: int = 42
    test_size: float = 0.2