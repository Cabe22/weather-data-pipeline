"""
Weather Data Collector Module
Collects real-time weather data from OpenWeatherMap API
"""

import requests
import pandas as pd
from datetime import datetime
import json
import sqlite3
import logging
from typing import Dict, List, Optional
import os
from dataclasses import dataclass
import schedule
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class WeatherConfig:
    """Configuration for weather data collection"""
    api_key: str
    cities: List[str]
    update_interval: int = 3600  # seconds
    db_path: str = "data/weather.db"
    
class WeatherCollector:
    """Collects weather data from OpenWeatherMap API"""
    
    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
    
    def __init__(self, config: WeatherConfig):
        self.config = config
        self.setup_database()
        
    def setup_database(self):
        """Initialize SQLite database with proper schema"""
        os.makedirs(os.path.dirname(self.config.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()
        
        # Create main weather data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS weather_data (
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
        
        # Create index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_city_timestamp 
            ON weather_data(city, timestamp)
        """)
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
        
    def fetch_weather_data(self, city: str) -> Optional[Dict]:
        """Fetch weather data for a single city"""
        try:
            params = {
                'q': city,
                'appid': self.config.api_key,
                'units': 'metric'  # Use metric units
            }
            
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Successfully fetched data for {city}")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data for {city}: {e}")
            return None
            
    def parse_weather_data(self, raw_data: Dict) -> Dict:
        """Parse raw API response into structured format"""
        parsed = {
            'city': raw_data.get('name'),
            'country': raw_data.get('sys', {}).get('country'),
            'timestamp': datetime.fromtimestamp(raw_data.get('dt', 0)),
            'temperature': raw_data.get('main', {}).get('temp'),
            'feels_like': raw_data.get('main', {}).get('feels_like'),
            'temp_min': raw_data.get('main', {}).get('temp_min'),
            'temp_max': raw_data.get('main', {}).get('temp_max'),
            'pressure': raw_data.get('main', {}).get('pressure'),
            'humidity': raw_data.get('main', {}).get('humidity'),
            'wind_speed': raw_data.get('wind', {}).get('speed'),
            'wind_deg': raw_data.get('wind', {}).get('deg'),
            'cloudiness': raw_data.get('clouds', {}).get('all'),
            'weather_main': raw_data.get('weather', [{}])[0].get('main'),
            'weather_description': raw_data.get('weather', [{}])[0].get('description'),
            'visibility': raw_data.get('visibility'),
            'rain_1h': raw_data.get('rain', {}).get('1h', 0),
            'snow_1h': raw_data.get('snow', {}).get('1h', 0),
            'lat': raw_data.get('coord', {}).get('lat'),
            'lon': raw_data.get('coord', {}).get('lon'),
            'timezone': raw_data.get('timezone')
        }
        return parsed
        
    def store_weather_data(self, weather_data: Dict):
        """Store parsed weather data in SQLite database"""
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()
        
        columns = list(weather_data.keys())
        placeholders = ['?' for _ in columns]
        
        query = f"""
            INSERT INTO weather_data ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
        """
        
        cursor.execute(query, list(weather_data.values()))
        conn.commit()
        conn.close()
        
        logger.info(f"Stored weather data for {weather_data['city']}")
        
    def collect_all_cities(self):
        """Collect weather data for all configured cities"""
        logger.info("Starting weather data collection cycle")
        
        for city in self.config.cities:
            raw_data = self.fetch_weather_data(city)
            
            if raw_data:
                parsed_data = self.parse_weather_data(raw_data)
                self.store_weather_data(parsed_data)
                
        logger.info("Completed weather data collection cycle")
        
    def get_recent_data(self, hours: int = 24) -> pd.DataFrame:
        """Retrieve recent weather data from database"""
        conn = sqlite3.connect(self.config.db_path)
        
        query = """
            SELECT * FROM weather_data
            WHERE datetime(timestamp) > datetime('now', '-{} hours')
            ORDER BY timestamp DESC
        """.format(hours)
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
        
    def run_scheduler(self):
        """Run continuous data collection on schedule"""
        # Initial collection
        self.collect_all_cities()
        
        # Schedule periodic updates
        schedule.every(self.config.update_interval).seconds.do(self.collect_all_cities)
        
        logger.info(f"Scheduler started. Collecting data every {self.config.update_interval} seconds")
        
        while True:
            schedule.run_pending()
            time.sleep(1)
            
# Example usage and configuration
if __name__ == "__main__":
    # Configuration
    config = WeatherConfig(
        api_key=os.getenv("OPENWEATHER_API_KEY", "your_api_key_here"),
        cities=[
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
            "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"
        ],
        update_interval=1800,  # 30 minutes
        db_path="data/weather.db"
    )
    
    # Initialize collector
    collector = WeatherCollector(config)
    
    # Run one collection cycle
    collector.collect_all_cities()
    
    # Get recent data
    recent_data = collector.get_recent_data(hours=24)
    print(f"Collected {len(recent_data)} records in the last 24 hours")
    print(recent_data.head())
    
    # Uncomment to run continuous collection
    # collector.run_scheduler()