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
import collections
import threading
from typing import Dict, List, Optional
import os
from dataclasses import dataclass, field
import schedule
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class APIMetrics:
    """Track API call metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    retried_requests: int = 0
    total_response_time: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests * 100

    @property
    def average_response_time(self) -> float:
        if self.successful_requests == 0:
            return 0.0
        return self.total_response_time / self.successful_requests

    def log_summary(self):
        logger.info(
            f"API Metrics - Total: {self.total_requests}, "
            f"Success: {self.successful_requests}, "
            f"Failed: {self.failed_requests}, "
            f"Retries: {self.retried_requests}, "
            f"Success Rate: {self.success_rate:.1f}%, "
            f"Avg Response Time: {self.average_response_time:.3f}s"
        )

    def reset(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.retried_requests = 0
        self.total_response_time = 0.0


class RateLimiter:
    """Sliding window rate limiter for API calls."""

    def __init__(self, max_calls: int = 60, period: float = 60.0):
        self.max_calls = max_calls
        self.period = period
        self._calls: collections.deque = collections.deque()
        self._lock = threading.Lock()

    def wait_if_needed(self):
        """Block until a call is allowed under the rate limit."""
        with self._lock:
            now = time.monotonic()
            # Remove expired entries
            while self._calls and self._calls[0] <= now - self.period:
                self._calls.popleft()

            if len(self._calls) >= self.max_calls:
                sleep_time = self._calls[0] - (now - self.period)
                if sleep_time > 0:
                    logger.warning(f"Rate limit reached. Waiting {sleep_time:.1f}s")
                    time.sleep(sleep_time)
                    # Clean up after sleeping
                    now = time.monotonic()
                    while self._calls and self._calls[0] <= now - self.period:
                        self._calls.popleft()

            self._calls.append(time.monotonic())


@dataclass
class WeatherConfig:
    """Configuration for weather data collection"""
    api_key: str
    cities: List[str]
    update_interval: int = 3600  # seconds
    db_path: str = "data/weather.db"
    max_retries: int = 3
    request_timeout: int = 10  # seconds
    rate_limit_calls: int = 60
    rate_limit_period: float = 60.0  # seconds
    
class WeatherCollector:
    """Collects weather data from OpenWeatherMap API"""
    
    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
    
    def __init__(self, config: WeatherConfig):
        self.config = config
        self.metrics = APIMetrics()
        self.rate_limiter = RateLimiter(
            max_calls=config.rate_limit_calls,
            period=config.rate_limit_period,
        )
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
        
    def _make_request_with_retry(self, city: str, params: Dict) -> requests.Response:
        """Make HTTP request with retry logic and exponential backoff.

        Retries on connection errors, timeouts, and server errors (5xx).
        Does not retry on client errors (4xx).
        """
        last_exception: Optional[Exception] = None
        max_attempts = self.config.max_retries + 1

        for attempt in range(1, max_attempts + 1):
            try:
                self.rate_limiter.wait_if_needed()
                start_time = time.monotonic()
                response = requests.get(
                    self.BASE_URL,
                    params=params,
                    timeout=self.config.request_timeout,
                )
                elapsed = time.monotonic() - start_time
                self.metrics.total_response_time += elapsed

                # Raise on server errors to trigger retry
                if response.status_code >= 500:
                    raise requests.exceptions.HTTPError(
                        f"{response.status_code} Server Error for {city}",
                        response=response,
                    )

                return response

            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                last_exception = e
                if attempt < max_attempts:
                    wait_time = 2 ** (attempt - 1)  # 1s, 2s, 4s
                    self.metrics.retried_requests += 1
                    logger.warning(
                        f"Request for {city} failed (attempt {attempt}/{max_attempts}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)

            except requests.exceptions.HTTPError as e:
                last_exception = e
                if (e.response is not None
                        and e.response.status_code >= 500
                        and attempt < max_attempts):
                    wait_time = 2 ** (attempt - 1)
                    self.metrics.retried_requests += 1
                    logger.warning(
                        f"Server error for {city} (attempt {attempt}/{max_attempts}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    # Client error (4xx) or final attempt — don't retry
                    raise

        raise last_exception

    def fetch_weather_data(self, city: str) -> Optional[Dict]:
        """Fetch weather data for a single city with retry and rate limiting."""
        self.metrics.total_requests += 1

        try:
            params = {
                'q': city,
                'appid': self.config.api_key,
                'units': 'metric'  # Use metric units
            }

            response = self._make_request_with_retry(city, params)
            response.raise_for_status()

            self.metrics.successful_requests += 1
            data = response.json()
            logger.info(f"Successfully fetched data for {city}")
            return data

        except requests.exceptions.RequestException as e:
            self.metrics.failed_requests += 1
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

        self.metrics.log_summary()
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