"""
Tests for API resilience features: retry logic, rate limiting, timeout handling, metrics.
"""

import time
import pytest
import requests
from unittest.mock import patch, MagicMock, PropertyMock

from src.data_collection.weather_collector import (
    WeatherCollector,
    WeatherConfig,
    APIMetrics,
    RateLimiter,
)


@pytest.fixture
def resilience_config(tmp_path):
    """WeatherConfig with fast retry settings for testing."""
    return WeatherConfig(
        api_key="fake_key",
        cities=["New York"],
        db_path=str(tmp_path / "test.db"),
        max_retries=2,
        request_timeout=5,
        rate_limit_calls=60,
        rate_limit_period=60.0,
    )


@pytest.fixture
def collector(resilience_config):
    """WeatherCollector instance for resilience tests."""
    return WeatherCollector(resilience_config)


@pytest.fixture
def success_response(sample_raw_api_response):
    """Mock response that succeeds."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = sample_raw_api_response
    resp.raise_for_status.return_value = None
    return resp


@pytest.fixture
def server_error_response():
    """Mock response with 500 status."""
    resp = MagicMock()
    resp.status_code = 500
    resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "500 Server Error", response=resp
    )
    return resp


@pytest.fixture
def not_found_response():
    """Mock response with 404 status."""
    resp = MagicMock()
    resp.status_code = 404
    resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "404 Not Found", response=resp
    )
    return resp


# --- APIMetrics tests ---


class TestAPIMetrics:
    def test_initial_values(self):
        m = APIMetrics()
        assert m.total_requests == 0
        assert m.successful_requests == 0
        assert m.failed_requests == 0
        assert m.retried_requests == 0
        assert m.total_response_time == 0.0

    def test_success_rate_no_requests(self):
        m = APIMetrics()
        assert m.success_rate == 0.0

    def test_success_rate(self):
        m = APIMetrics(total_requests=10, successful_requests=8)
        assert m.success_rate == 80.0

    def test_average_response_time_no_requests(self):
        m = APIMetrics()
        assert m.average_response_time == 0.0

    def test_average_response_time(self):
        m = APIMetrics(successful_requests=4, total_response_time=2.0)
        assert m.average_response_time == 0.5

    def test_reset(self):
        m = APIMetrics(
            total_requests=5,
            successful_requests=3,
            failed_requests=2,
            retried_requests=1,
            total_response_time=1.5,
        )
        m.reset()
        assert m.total_requests == 0
        assert m.successful_requests == 0
        assert m.failed_requests == 0
        assert m.retried_requests == 0
        assert m.total_response_time == 0.0

    def test_log_summary_does_not_raise(self):
        m = APIMetrics(total_requests=3, successful_requests=2, failed_requests=1)
        m.log_summary()  # should not raise


# --- RateLimiter tests ---


class TestRateLimiter:
    def test_allows_calls_under_limit(self):
        limiter = RateLimiter(max_calls=5, period=60.0)
        for _ in range(5):
            limiter.wait_if_needed()  # should not block

    @patch("src.data_collection.weather_collector.time.sleep")
    def test_blocks_when_limit_reached(self, mock_sleep):
        limiter = RateLimiter(max_calls=2, period=60.0)
        limiter.wait_if_needed()
        limiter.wait_if_needed()
        # Third call should trigger wait
        limiter.wait_if_needed()
        mock_sleep.assert_called()

    def test_expired_calls_are_removed(self):
        limiter = RateLimiter(max_calls=2, period=0.01)  # very short period
        limiter.wait_if_needed()
        limiter.wait_if_needed()
        time.sleep(0.02)  # let entries expire
        # Should not block since old entries expired
        limiter.wait_if_needed()


# --- Retry logic tests ---


class TestRetryOnConnectionError:
    @patch("src.data_collection.weather_collector.time.sleep")
    @patch("src.data_collection.weather_collector.requests.get")
    def test_retries_on_connection_error_then_succeeds(
        self, mock_get, mock_sleep, collector, success_response
    ):
        mock_get.side_effect = [
            requests.exceptions.ConnectionError("Connection refused"),
            success_response,
        ]
        result = collector.fetch_weather_data("New York")
        assert result is not None
        assert mock_get.call_count == 2
        assert collector.metrics.retried_requests == 1
        assert collector.metrics.successful_requests == 1

    @patch("src.data_collection.weather_collector.time.sleep")
    @patch("src.data_collection.weather_collector.requests.get")
    def test_fails_after_max_retries_on_connection_error(
        self, mock_get, mock_sleep, collector
    ):
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")
        result = collector.fetch_weather_data("New York")
        assert result is None
        # max_retries=2 means 3 total attempts
        assert mock_get.call_count == 3
        assert collector.metrics.retried_requests == 2
        assert collector.metrics.failed_requests == 1


class TestRetryOnTimeout:
    @patch("src.data_collection.weather_collector.time.sleep")
    @patch("src.data_collection.weather_collector.requests.get")
    def test_retries_on_timeout_then_succeeds(
        self, mock_get, mock_sleep, collector, success_response
    ):
        mock_get.side_effect = [
            requests.exceptions.Timeout("Request timed out"),
            success_response,
        ]
        result = collector.fetch_weather_data("New York")
        assert result is not None
        assert mock_get.call_count == 2
        assert collector.metrics.retried_requests == 1

    @patch("src.data_collection.weather_collector.time.sleep")
    @patch("src.data_collection.weather_collector.requests.get")
    def test_fails_after_max_retries_on_timeout(
        self, mock_get, mock_sleep, collector
    ):
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")
        result = collector.fetch_weather_data("New York")
        assert result is None
        assert mock_get.call_count == 3
        assert collector.metrics.failed_requests == 1


class TestRetryOnServerError:
    @patch("src.data_collection.weather_collector.time.sleep")
    @patch("src.data_collection.weather_collector.requests.get")
    def test_retries_on_500_then_succeeds(
        self, mock_get, mock_sleep, collector, server_error_response, success_response
    ):
        mock_get.side_effect = [server_error_response, success_response]
        result = collector.fetch_weather_data("New York")
        assert result is not None
        assert mock_get.call_count == 2
        assert collector.metrics.retried_requests == 1

    @patch("src.data_collection.weather_collector.time.sleep")
    @patch("src.data_collection.weather_collector.requests.get")
    def test_fails_after_max_retries_on_500(
        self, mock_get, mock_sleep, collector, server_error_response
    ):
        mock_get.return_value = server_error_response
        result = collector.fetch_weather_data("New York")
        assert result is None
        assert mock_get.call_count == 3
        assert collector.metrics.failed_requests == 1


class TestNoRetryOnClientError:
    @patch("src.data_collection.weather_collector.time.sleep")
    @patch("src.data_collection.weather_collector.requests.get")
    def test_no_retry_on_404(
        self, mock_get, mock_sleep, collector, not_found_response
    ):
        mock_get.return_value = not_found_response
        result = collector.fetch_weather_data("InvalidCity")
        assert result is None
        # Should only attempt once — no retry for 4xx
        assert mock_get.call_count == 1
        assert collector.metrics.retried_requests == 0
        assert collector.metrics.failed_requests == 1


# --- Exponential backoff timing tests ---


class TestExponentialBackoff:
    @patch("src.data_collection.weather_collector.time.sleep")
    @patch("src.data_collection.weather_collector.requests.get")
    def test_backoff_delays(self, mock_get, mock_sleep, collector):
        mock_get.side_effect = requests.exceptions.ConnectionError("fail")
        collector.fetch_weather_data("New York")
        # With max_retries=2: attempt 1 fails -> sleep(1), attempt 2 fails -> sleep(2)
        calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert calls == [1, 2]


# --- Metrics tracking tests ---


class TestMetricsTracking:
    @patch("src.data_collection.weather_collector.requests.get")
    def test_success_updates_metrics(self, mock_get, collector, success_response):
        mock_get.return_value = success_response
        collector.fetch_weather_data("New York")
        assert collector.metrics.total_requests == 1
        assert collector.metrics.successful_requests == 1
        assert collector.metrics.failed_requests == 0
        assert collector.metrics.total_response_time >= 0

    @patch("src.data_collection.weather_collector.time.sleep")
    @patch("src.data_collection.weather_collector.requests.get")
    def test_failure_updates_metrics(self, mock_get, mock_sleep, collector):
        mock_get.side_effect = requests.exceptions.ConnectionError("fail")
        collector.fetch_weather_data("New York")
        assert collector.metrics.total_requests == 1
        assert collector.metrics.successful_requests == 0
        assert collector.metrics.failed_requests == 1

    @patch("src.data_collection.weather_collector.requests.get")
    def test_collect_all_cities_logs_metrics(self, mock_get, collector, success_response):
        mock_get.return_value = success_response
        collector.collect_all_cities()
        assert collector.metrics.total_requests == len(collector.config.cities)
        assert collector.metrics.successful_requests == len(collector.config.cities)


# --- Config defaults tests ---


class TestWeatherConfigDefaults:
    def test_default_retry_settings(self):
        cfg = WeatherConfig(api_key="key", cities=["NYC"])
        assert cfg.max_retries == 3
        assert cfg.request_timeout == 10
        assert cfg.rate_limit_calls == 60
        assert cfg.rate_limit_period == 60.0

    def test_custom_retry_settings(self):
        cfg = WeatherConfig(
            api_key="key",
            cities=["NYC"],
            max_retries=5,
            request_timeout=30,
            rate_limit_calls=30,
            rate_limit_period=120.0,
        )
        assert cfg.max_retries == 5
        assert cfg.request_timeout == 30
        assert cfg.rate_limit_calls == 30
        assert cfg.rate_limit_period == 120.0


# --- Collector initialization tests ---


class TestCollectorInitialization:
    def test_has_metrics(self, collector):
        assert isinstance(collector.metrics, APIMetrics)

    def test_has_rate_limiter(self, collector):
        assert isinstance(collector.rate_limiter, RateLimiter)

    def test_rate_limiter_uses_config(self, resilience_config):
        cfg = WeatherConfig(
            api_key="key",
            cities=["NYC"],
            db_path=str(resilience_config.db_path),
            rate_limit_calls=30,
            rate_limit_period=120.0,
        )
        collector = WeatherCollector(cfg)
        assert collector.rate_limiter.max_calls == 30
        assert collector.rate_limiter.period == 120.0
