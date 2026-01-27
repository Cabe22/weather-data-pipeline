# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Weather Data Pipeline is a data engineering and ML project for real-time weather data collection, processing, and predictive analytics. It collects data from OpenWeatherMap API, stores in SQLite, engineers 40+ features, trains ML models (93%+ accuracy), and visualizes via Streamlit dashboard.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run data collection
python run_data_collection.py
# Or directly:
python src/data_collection/weather_collector.py

# Launch dashboard
streamlit run dashboard.py

# Run test suite
python test_data_collection.py

# Test API connectivity
python test_api.py

# Inspect database
python inspect_database.py
python quick_db_check.py

# Data quality report
python data_quality_report.py
```

## Architecture

```
OpenWeatherMap API → WeatherCollector → SQLite DB → WeatherDataProcessor → WeatherPredictor → Streamlit Dashboard
```

**Pipeline Flow**: API data collection → SQLite storage → Feature engineering (40+ features) → ML model training → Interactive visualization

## Key Classes

- **WeatherCollector** (`src/data_collection/weather_collector.py`): Fetches weather data from API, manages SQLite database, supports scheduled collection
- **WeatherDataProcessor** (`src/data_processing/data_processor.py`): Feature engineering pipeline with time features, lag features, rolling stats, weather indices, interaction features
- **WeatherPredictor** (`src/ml_models/weather_predictor.py`): Multiple regression models (Linear, Ridge, Random Forest, Gradient Boosting, XGBoost), rain classifier, ensemble with meta-learner

## Configuration

Environment variables in `.env` (see `.env.example`):
- `OPENWEATHER_API_KEY`: Required for data collection
- `DATABASE_PATH`: SQLite database path (default: `data/weather.db`)
- `LOG_LEVEL`: Logging level (default: INFO)
- `UPDATE_INTERVAL`: Collection interval in seconds (default: 1800)

Config classes in `configs/config.py`: `WeatherConfig`, `DatabaseConfig`, `ModelConfig`

## Database

SQLite database at `data/weather.db` with `weather_data` table. Key columns: city, timestamp, temperature, humidity, pressure, wind_speed, weather_main. Indexed on (city, timestamp).

## Models

Trained models saved as PKL files in `models/` directory using joblib.

## Important Patterns

- Dashboard uses Streamlit caching decorators (`@st.cache_data`, `@st.cache_resource`)
- Feature engineering uses pandas groupby and rolling windows for time series
- All modules use logging for execution tracking
- Type hints throughout codebase
