# API Reference

Complete reference for all classes, methods, and configuration in the Weather Data Pipeline.

---

## Table of Contents

- [Configuration](#configuration)
  - [WeatherConfig](#weatherconfig)
  - [DatabaseConfig](#databaseconfig)
  - [ModelConfig](#modelconfig)
- [Data Collection](#data-collection)
  - [WeatherCollector](#weathercollector)
  - [RateLimiter](#ratelimiter)
  - [APIMetrics](#apimetrics)
- [Data Processing](#data-processing)
  - [WeatherDataProcessor](#weatherdataprocessor)
- [Machine Learning](#machine-learning)
  - [WeatherPredictor](#weatherpredictor)
  - [ModelRegistry](#modelregistry)
- [Dashboard](#dashboard)

---

## Configuration

Defined in `configs/config.py`. All configuration classes use Python dataclasses.

### WeatherConfig

Controls API collection behavior and database location.

```python
from configs.config import WeatherConfig

config = WeatherConfig(
    api_key="your_api_key",
    cities=["New York", "Chicago"],
    update_interval=1800,
    db_path="data/weather.db"
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | `os.getenv("OPENWEATHER_API_KEY")` | OpenWeatherMap API key (required) |
| `cities` | `List[str]` | 10 major US cities | Cities to monitor |
| `update_interval` | `int` | `1800` | Collection interval in seconds |
| `db_path` | `str` | `"data/weather.db"` | SQLite database file path |

Default cities: New York, Los Angeles, Chicago, Houston, Phoenix, Philadelphia, San Antonio, San Diego, Dallas, San Jose.

### DatabaseConfig

Database and backup paths.

```python
from configs.config import DatabaseConfig

db_config = DatabaseConfig(
    db_path="data/weather.db",
    backup_path="data/backups/"
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db_path` | `str` | `"data/weather.db"` | Primary database path |
| `backup_path` | `str` | `"data/backups/"` | Backup directory path |

### ModelConfig

ML model training parameters.

```python
from configs.config import ModelConfig

model_config = ModelConfig(
    model_dir="models/",
    random_state=42,
    test_size=0.2
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_dir` | `str` | `"models/"` | Directory for saved models |
| `random_state` | `int` | `42` | Random seed for reproducibility |
| `test_size` | `float` | `0.2` | Proportion of data used for testing |

---

## Data Collection

Defined in `src/data_collection/weather_collector.py`.

### WeatherCollector

Fetches weather data from the OpenWeatherMap API and stores it in SQLite.

**Base URL:** `https://api.openweathermap.org/data/2.5/weather`

```python
from src.data_collection.weather_collector import WeatherCollector, WeatherConfig

config = WeatherConfig(api_key="your_key")
collector = WeatherCollector(config)
```

#### Constructor

```python
WeatherCollector(config: WeatherConfig)
```

Sets up the collector with the given configuration. Automatically calls `setup_database()` and initializes the `RateLimiter` and `APIMetrics`.

#### Methods

##### `setup_database() -> None`

Creates the `weather_data` table and indexes if they don't exist. Called automatically during initialization.

##### `fetch_weather_data(city: str) -> Optional[Dict]`

Fetches current weather data for a single city from the OpenWeatherMap API.

- **Parameters:** `city` - City name (e.g., `"New York"`)
- **Returns:** Parsed weather data dictionary, or `None` on failure
- **Side effects:** Updates `APIMetrics` counters

```python
data = collector.fetch_weather_data("New York")
# Returns: {'city': 'New York', 'temperature': 22.5, 'humidity': 65, ...}
```

##### `validate_api_response(data: Dict) -> bool`

Validates that an API response contains all required fields and values are within plausible physical ranges.

- **Parameters:** `data` - Raw API response dictionary
- **Returns:** `True` if valid, `False` otherwise

##### `parse_weather_data(raw_data: Dict) -> Dict`

Converts raw API JSON into a flat dictionary matching the database schema.

- **Parameters:** `raw_data` - Raw API response
- **Returns:** Dictionary with keys matching `weather_data` table columns

##### `store_weather_data(weather_data: Dict) -> None`

Inserts or updates a weather observation in the database. Uses `INSERT OR REPLACE` (UPSERT) on the `(city, timestamp)` unique constraint to prevent duplicates.

- **Parameters:** `weather_data` - Parsed weather dictionary

##### `collect_all_cities() -> None`

Iterates through all configured cities, fetches and stores data for each. Logs success/failure per city and prints an API metrics summary.

```python
collector.collect_all_cities()
# Collects data for all cities in config.cities
```

##### `get_recent_data(hours: int = 24) -> pd.DataFrame`

Queries the database for recent weather observations.

- **Parameters:** `hours` - Number of hours of history to retrieve (default: 24)
- **Returns:** DataFrame with weather data, sorted by timestamp descending

```python
df = collector.get_recent_data(hours=48)
```

##### `run_scheduler() -> None`

Starts continuous data collection on a schedule. Runs `collect_all_cities()` at the interval specified by `config.update_interval`. Blocks indefinitely (intended for production use).

```python
collector.run_scheduler()  # Runs forever, collecting every 30 minutes
```

### RateLimiter

Prevents exceeding the OpenWeatherMap API rate limit.

```python
RateLimiter(max_calls: int = 60, period: float = 60.0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_calls` | `int` | `60` | Maximum calls allowed per period |
| `period` | `float` | `60.0` | Time window in seconds |

##### `wait_if_needed() -> None`

Blocks the calling thread if the rate limit has been reached, resuming once the window resets.

### APIMetrics

Dataclass tracking API call statistics.

| Field | Type | Description |
|-------|------|-------------|
| `total_requests` | `int` | Total API calls made |
| `successful_requests` | `int` | Calls that returned HTTP 200 |
| `failed_requests` | `int` | Calls that failed |
| `retried_requests` | `int` | Calls that required retries |
| `total_response_time` | `float` | Cumulative response time (seconds) |

**Properties:**

- `success_rate -> float` - Ratio of successful to total requests
- `average_response_time -> float` - Mean response time in seconds

##### `log_summary() -> None`

Logs a summary of all metrics at INFO level.

##### `reset() -> None`

Resets all counters to zero.

---

## Data Processing

Defined in `src/data_processing/data_processor.py`.

### WeatherDataProcessor

Feature engineering pipeline that creates 40+ features from raw weather data.

```python
from src.data_processing.data_processor import WeatherDataProcessor

processor = WeatherDataProcessor(db_path="data/weather.db")
df = processor.process_pipeline()
```

#### Constructor

```python
WeatherDataProcessor(db_path: str = "data/weather.db")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db_path` | `str` | `"data/weather.db"` | Path to SQLite database |

Initializes `scaler` (StandardScaler) and `label_encoders` (Dict).

#### Data Loading

##### `load_data(start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame`

Loads raw weather data from the database with optional date filtering.

- **Parameters:**
  - `start_date` - ISO format start date filter (optional)
  - `end_date` - ISO format end date filter (optional)
- **Returns:** DataFrame with raw weather observations

```python
df = processor.load_data(start_date="2026-01-01", end_date="2026-01-31")
```

#### Quality Checks

##### `run_quality_checks(df: pd.DataFrame) -> Dict`

Validates data for duplicates, missing values, and out-of-range values. Uses `VALID_RANGES` class constant for physical plausibility checks.

- **Returns:** Dictionary with quality report (duplicate count, missing values, range violations)

#### Feature Engineering Methods

Each method takes a DataFrame and returns an augmented DataFrame with new columns.

##### `create_time_features(df: pd.DataFrame) -> pd.DataFrame`

Creates 10+ time-based features:

| Feature | Description |
|---------|-------------|
| `hour`, `day_of_week`, `month`, `quarter`, `day_of_year` | Basic temporal |
| `hour_sin`, `hour_cos`, `month_sin`, `month_cos`, `day_sin`, `day_cos` | Cyclical encoding |
| `time_of_day` | Categorical: Night/Morning/Afternoon/Evening |
| `is_weekend` | Binary weekend indicator |

##### `create_lag_features(df: pd.DataFrame, lag_hours: List[int] = [1, 3, 6, 12, 24]) -> pd.DataFrame`

Creates lagged values and rolling statistics per city.

- **Lag features:** `{metric}_lag_{hours}` for temperature, humidity, pressure, wind_speed (20 features)
- **Rolling features:** `{metric}_rolling_mean_24h`, `{metric}_rolling_std_24h` (8 features)

##### `create_weather_indices(df: pd.DataFrame) -> pd.DataFrame`

Creates domain-specific weather indices:

| Feature | Description |
|---------|-------------|
| `heat_index` | Perceived temperature from temp + humidity |
| `wind_chill` | Wind chill factor (applied when temp < 10 C) |
| `discomfort_index` | Temperature-humidity discomfort metric |
| `pressure_change` | Rate of pressure change |
| `temp_range` | Difference between max and min temperature |

##### `create_interaction_features(df: pd.DataFrame) -> pd.DataFrame`

Creates cross-feature products:

| Feature | Formula |
|---------|---------|
| `temp_humidity_interaction` | temperature * humidity |
| `wind_temp_interaction` | wind_speed * temperature |
| `pressure_temp_interaction` | pressure * temperature |
| `cloud_humidity_interaction` | cloudiness * humidity |

##### `handle_missing_values(df: pd.DataFrame) -> pd.DataFrame`

Handles missing data using:
1. Forward fill (limit=3) then linear interpolation for numeric columns
2. Mode imputation for categorical columns
3. Drops rows with more than 30% null values

##### `encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame`

Applies LabelEncoding to: `city`, `country`, `weather_main`, `time_of_day`.

##### `normalize_features(df: pd.DataFrame, exclude_cols: List[str] = None) -> pd.DataFrame`

Applies StandardScaler normalization to all numeric columns except those in `exclude_cols`.

#### Target Variable

##### `create_target_variable(df: pd.DataFrame, target_hours: int = 24) -> pd.DataFrame`

Creates prediction targets:

| Target | Description |
|--------|-------------|
| `temperature_future` | Temperature `target_hours` ahead |
| `temp_change` | Delta from current to future temperature |
| `temp_change_category` | Categorical: Decrease / Stable / Increase |
| `will_rain` | Binary: 1 if future weather is Rain/Drizzle/Thunderstorm |

#### Full Pipeline

##### `process_pipeline(start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame`

Runs the complete feature engineering pipeline in order:

1. Load data
2. Quality checks
3. Time features
4. Lag features
5. Weather indices
6. Interaction features
7. Handle missing values
8. Encode categoricals
9. Create target variables
10. Drop rows without targets

```python
df = processor.process_pipeline()
# Returns DataFrame with 40+ engineered features and target columns
```

##### `get_feature_importance_data(df: pd.DataFrame) -> pd.DataFrame`

Returns summary statistics for each feature: mean, std, min, max, null count, and correlation with temperature.

---

## Machine Learning

### WeatherPredictor

Defined in `src/ml_models/weather_predictor.py`. Trains and manages multiple ML models for weather prediction.

```python
from src.ml_models.weather_predictor import WeatherPredictor

predictor = WeatherPredictor(model_dir="models/")
```

#### Constructor

```python
WeatherPredictor(model_dir: str = "models/")
```

Initializes empty dictionaries for `models`, `best_models`, `feature_importance`, and `model_metadata`. Sets `scaler` and `feature_columns` to `None`.

#### Feature Preparation

##### `prepare_features(df: pd.DataFrame, target_col: str = 'temperature_future') -> Tuple[pd.DataFrame, pd.Series]`

Separates features from target, removes non-predictive columns (id, timestamp, city, etc.), drops features with >30% nulls, and fits a StandardScaler.

- **Returns:** Tuple of (feature DataFrame, target Series)
- **Side effects:** Sets `self.scaler` and `self.feature_columns`

#### Training Methods

##### `train_temperature_models(X: pd.DataFrame, y: pd.Series) -> Dict`

Trains six regression models and evaluates each on train/test split:

| Algorithm | Key Parameters |
|-----------|---------------|
| Linear Regression | Default |
| Ridge Regression | alpha=1.0 |
| Random Forest | n_estimators=100, max_depth=20 |
| Gradient Boosting | n_estimators=100, learning_rate=0.1 |
| XGBoost | n_estimators=100, learning_rate=0.1, max_depth=6 |
| MLP Neural Network | hidden_layers=(100, 50) |

Returns dictionary keyed by algorithm name, each containing: `model`, `train_mse`, `test_mse`, `train_r2`, `test_r2`, `test_mae`, `cv_scores`, `predictions`, `feature_importance`.

Automatically selects the best model (highest test R2) and stores it in `self.best_models['temperature']`.

##### `train_rain_classifier(X: pd.DataFrame, y: pd.Series) -> Dict`

Trains a Random Forest classifier for rain prediction.

- **Parameters:** `y` - Binary rain labels
- **Returns:** Dictionary with `model`, `classification_report`, `roc_auc`, `predictions`, `probabilities`

##### `hyperparameter_tuning(X: pd.DataFrame, y: pd.Series, model_type: str = 'xgboost') -> Dict`

Runs GridSearchCV with 5-fold cross-validation.

- **Supported types:** `'xgboost'`, `'random_forest'`
- **Returns:** `best_params`, `best_score`, test metrics, and fitted model

##### `create_ensemble_model(X: pd.DataFrame, y: pd.Series) -> Dict`

Builds a stacking ensemble with:
- **Base models:** Random Forest, Gradient Boosting, XGBoost
- **Meta-learner:** Linear Regression

Returns `test_mse`, `test_r2`, and `ensemble_predictions`.

#### Inference

##### `predict(X: pd.DataFrame, model_type: str = 'temperature') -> np.ndarray`

Generates predictions using the best model for the given type.

- **Parameters:**
  - `X` - Feature DataFrame (must contain columns matching `self.feature_columns`)
  - `model_type` - `'temperature'` or `'rain'`
- **Returns:** NumPy array of predictions
- **Note:** Automatically applies the saved scaler if present

```python
predictions = predictor.predict(features_df, model_type='temperature')
```

#### Model Persistence

##### `save_models() -> None`

Saves all trained models to the `model_dir` as PKL files using joblib. Each file bundles:
- The trained model
- The fitted StandardScaler
- Feature column names
- Training metadata (timestamp, algorithm, sample counts, metrics)

Also registers each model version in the `ModelRegistry`.

##### `load_models() -> None`

Loads previously saved models from disk. Restores the scaler, feature columns, and metadata.

#### Visualization

##### `plot_model_comparison(results: Dict) -> matplotlib.figure.Figure`

Creates a 2x2 subplot figure comparing all trained models:
1. MSE comparison (train vs test)
2. R2 comparison (train vs test)
3. Best model: predictions vs actual scatter
4. Residuals distribution histogram

##### `plot_feature_importance(top_n: int = 20) -> matplotlib.figure.Figure`

Bar chart showing the top N most important features from the best model.

---

### ModelRegistry

Defined in `src/ml_models/model_registry.py`. Tracks model versions, artifacts, and metadata in a JSON file.

```python
from src.ml_models.model_registry import ModelRegistry

registry = ModelRegistry("models/registry.json")
```

#### Constructor

```python
ModelRegistry(registry_path: str = "models/registry.json")
```

Creates the registry JSON file if it doesn't exist.

#### Methods

##### `register(model_type: str, artifact_path: str, metadata: Dict) -> str`

Registers a new model version.

- **Parameters:**
  - `model_type` - e.g., `"temperature"`, `"rain"`
  - `artifact_path` - Path to the saved model file
  - `metadata` - Training metadata (metrics, algorithm, etc.)
- **Returns:** Version string (e.g., `"v1"`, `"v2"`)

```python
version = registry.register(
    model_type="temperature",
    artifact_path="models/temperature_model.pkl",
    metadata={"best_algorithm": "XGBoost", "metrics": {"test_r2": 0.93}}
)
```

##### `list_versions(model_type: Optional[str] = None) -> List[Dict]`

Lists all registered versions, optionally filtered by model type.

- **Returns:** List of version entries with `version`, `model_type`, `registered_at`, `artifact_path`, `metadata`

##### `get_version(version: str) -> Optional[Dict]`

Retrieves a specific version entry by version string.

##### `get_latest(model_type: str) -> Optional[Dict]`

Returns the most recently registered version for a given model type.

##### `compare(version_a: str, version_b: str) -> Optional[Dict]`

Compares metrics between two versions.

- **Returns:** Dictionary with `metrics` containing per-metric values and delta

```python
diff = registry.compare("v1", "v2")
# {'metrics': {'test_r2': {'v1': 0.91, 'v2': 0.93, 'delta': 0.02}}}
```

---

## Dashboard

Defined in `dashboard.py`. Streamlit web application for interactive data exploration.

**Launch:** `streamlit run dashboard.py`

### Cached Functions

| Function | Cache | TTL | Description |
|----------|-------|-----|-------------|
| `load_data(db_path, hours)` | `@st.cache_data` | 60s | Loads weather data from SQLite |
| `load_predictor(model_dir)` | `@st.cache_resource` | None | Loads trained ML models |
| `engineer_features_for_prediction(db_path)` | `@st.cache_data` | 300s | Runs feature engineering pipeline |

### Chart Functions

| Function | Returns | Description |
|----------|---------|-------------|
| `create_temperature_chart(df, city)` | `go.Figure` | Multi-city temperature line chart |
| `create_weather_metrics(df)` | `Dict` | Summary statistics dictionary |
| `create_correlation_heatmap(df)` | `go.Figure` | Feature correlation matrix |
| `create_weather_distribution(df)` | `go.Figure` | Weather condition bar chart |
| `create_city_comparison(df)` | `go.Figure` | 2x2 city comparison subplots |
| `predict_temperature(predictor, df, city)` | `Dict` | Temperature prediction with metadata |

### Dashboard Sections

1. **Sidebar Controls** - Time range, city filter, auto-refresh, manual refresh
2. **Key Metrics** - Temperature, humidity, city count, record count
3. **Temperature Trends** - Interactive Plotly time series
4. **Weather Conditions** - Distribution of weather types
5. **Correlations** - Feature correlation heatmap
6. **City Comparison** - Multi-metric city comparison
7. **ML Predictions** - On-demand temperature predictions
8. **Model Information** - Training metadata and algorithm comparison
9. **Model Version History** - Registry-based version tracking and comparison
10. **Raw Data** - Expandable data table with CSV export
