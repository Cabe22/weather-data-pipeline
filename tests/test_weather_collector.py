"""
Tests for additional WeatherCollector methods: get_recent_data, run_scheduler, db ops.
"""

import sqlite3
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from src.data_collection.weather_collector import WeatherCollector, WeatherConfig


class TestGetRecentData:
    def test_returns_dataframe(self, mock_collector):
        collector, _ = mock_collector
        df = collector.get_recent_data(hours=24)
        assert isinstance(df, pd.DataFrame)

    def test_empty_on_fresh_db(self, weather_config):
        collector = WeatherCollector(weather_config)
        df = collector.get_recent_data(hours=24)
        assert len(df) == 0

    def test_only_recent_rows_returned(self, mock_collector, sample_raw_api_response):
        collector, _ = mock_collector
        # Store two records: one "now" and one old
        from datetime import datetime, timedelta

        recent = collector.parse_weather_data(sample_raw_api_response)
        recent["timestamp"] = datetime.utcnow()
        collector.store_weather_data(recent)

        old = collector.parse_weather_data(sample_raw_api_response)
        old["timestamp"] = datetime.utcnow() - timedelta(hours=48)
        collector.store_weather_data(old)

        df = collector.get_recent_data(hours=24)
        # Only the recent row should come back
        assert len(df) == 1

    def test_timestamp_ordering(self, mock_collector, sample_raw_api_response):
        collector, _ = mock_collector
        from datetime import datetime, timedelta

        for delta_h in [2, 0, 1]:
            row = collector.parse_weather_data(sample_raw_api_response)
            row["timestamp"] = datetime.utcnow() - timedelta(hours=delta_h)
            collector.store_weather_data(row)

        df = collector.get_recent_data(hours=24)
        assert len(df) > 0
        # Should be ordered DESC by timestamp
        ts = pd.to_datetime(df["timestamp"])
        assert ts.is_monotonic_decreasing


class TestRunScheduler:
    def test_calls_collect_all_cities(self, mock_collector):
        collector, _ = mock_collector
        with patch.object(collector, "collect_all_cities") as mock_collect, \
             patch("src.data_collection.weather_collector.schedule") as mock_sched, \
             patch("src.data_collection.weather_collector.time.sleep", side_effect=KeyboardInterrupt):
            try:
                collector.run_scheduler()
            except KeyboardInterrupt:
                pass
            # Initial collection call
            mock_collect.assert_called()

    def test_registers_schedule_job(self, mock_collector):
        collector, _ = mock_collector
        with patch("src.data_collection.weather_collector.schedule") as mock_sched, \
             patch("src.data_collection.weather_collector.time.sleep", side_effect=KeyboardInterrupt), \
             patch("src.data_collection.weather_collector.requests.get", return_value=MagicMock(status_code=200, json=MagicMock(return_value={}))):
            try:
                collector.run_scheduler()
            except KeyboardInterrupt:
                pass
            mock_sched.every.assert_called()


class TestDatabaseOperationsAdditional:
    def test_store_multiple_records(self, mock_collector, sample_raw_api_response):
        collector, _ = mock_collector
        for _ in range(5):
            parsed = collector.parse_weather_data(sample_raw_api_response)
            collector.store_weather_data(parsed)

        conn = sqlite3.connect(collector.config.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM weather_data")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 5

    def test_index_exists(self, weather_config):
        collector = WeatherCollector(weather_config)
        conn = sqlite3.connect(collector.config.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_city_timestamp'")
        idx = cursor.fetchone()
        conn.close()
        assert idx is not None

    def test_null_optional_fields(self, mock_collector, sample_raw_api_response):
        collector, _ = mock_collector
        parsed = collector.parse_weather_data(sample_raw_api_response)
        parsed["rain_1h"] = None
        parsed["snow_1h"] = None
        collector.store_weather_data(parsed)

        conn = sqlite3.connect(collector.config.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT rain_1h, snow_1h FROM weather_data WHERE id=1")
        row = cursor.fetchone()
        conn.close()
        assert row[0] is None
        assert row[1] is None
