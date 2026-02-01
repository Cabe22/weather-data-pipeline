# Troubleshooting Guide

Common issues and solutions for the Weather Data Pipeline.

---

## Table of Contents

- [API & Data Collection](#api--data-collection)
- [Database](#database)
- [ML Models](#ml-models)
- [Dashboard](#dashboard)
- [Docker](#docker)
- [Data Quality](#data-quality)

---

## API & Data Collection

### "Invalid API key" or 401 Unauthorized

**Symptom:** Data collection fails with authentication errors.

**Solutions:**
1. Verify your API key is set in `.env`:
   ```
   OPENWEATHER_API_KEY=your_actual_key_here
   ```
2. Check there are no extra spaces or quotes around the key
3. New API keys can take up to 2 hours to activate after creation
4. Verify the key works directly:
   ```bash
   curl "https://api.openweathermap.org/data/2.5/weather?q=London&appid=YOUR_KEY"
   ```
5. Ensure `python-dotenv` is installed so `.env` is loaded:
   ```bash
   pip install python-dotenv
   ```

### "429 Too Many Requests" / Rate Limit Exceeded

**Symptom:** API calls fail intermittently with rate limit errors.

**Solutions:**
1. The free tier allows 60 calls/minute. The default config (10 cities) uses 10 calls per cycle, well within limits.
2. If running multiple instances, reduce the city list or increase `UPDATE_INTERVAL`
3. The built-in `RateLimiter` should handle this automatically. Check that it is not being bypassed.
4. Wait a few minutes and retry — rate limits reset quickly.

### "Connection timeout" or Network Errors

**Symptom:** `requests.exceptions.ConnectionError` or `Timeout`.

**Solutions:**
1. Check internet connectivity: `ping api.openweathermap.org`
2. If behind a corporate proxy, configure it:
   ```bash
   export HTTPS_PROXY=http://proxy:port
   ```
3. The collector retries up to 3 times with exponential backoff (1s, 2s, 4s). Transient failures are handled automatically.
4. Increase `request_timeout` in `WeatherConfig` if on a slow connection.

### No Data Being Collected

**Symptom:** `run_data_collection.py` runs without errors but the database is empty.

**Solutions:**
1. Check the console output for "Successfully stored data for..." messages
2. Verify the database path: `ls data/weather.db`
3. Run `python quick_db_check.py` to inspect the database
4. Check logs for validation failures — the collector skips invalid API responses silently
5. Ensure the `data/` directory exists: `mkdir -p data`

---

## Database

### "no such table: weather_data"

**Symptom:** SQL error when querying the database.

**Solutions:**
1. The table is created automatically on first run. Run the collector once:
   ```bash
   python run_data_collection.py
   ```
2. If the database file exists but the table is missing, the file may be corrupted. Delete and recreate:
   ```bash
   rm data/weather.db
   python run_data_collection.py
   ```

### "database is locked"

**Symptom:** `sqlite3.OperationalError: database is locked`.

**Solutions:**
1. Only one process can write to SQLite at a time. Ensure you're not running multiple collectors simultaneously.
2. If the dashboard and collector run concurrently, this is usually fine (reads don't block). If it persists:
   ```python
   # Enable WAL mode for better concurrency
   import sqlite3
   conn = sqlite3.connect("data/weather.db")
   conn.execute("PRAGMA journal_mode=WAL")
   conn.close()
   ```
3. Close any open SQLite CLI sessions or DB browser tools.

### Database File Growing Large

**Solutions:**
1. Check the current size: `ls -lh data/weather.db`
2. Prune old data:
   ```sql
   sqlite3 data/weather.db "DELETE FROM weather_data WHERE datetime(timestamp) < datetime('now', '-90 days')"
   sqlite3 data/weather.db "VACUUM"
   ```
3. The typical row size is ~500 bytes. At 10 cities every 30 minutes, expect ~5 MB/year.

### Corrupted Database

**Symptom:** `sqlite3.DatabaseError: file is not a database` or similar.

**Solutions:**
1. Restore from a backup if available:
   ```bash
   cp data/backups/weather_latest.db data/weather.db
   ```
2. Attempt an integrity check:
   ```bash
   sqlite3 data/weather.db "PRAGMA integrity_check"
   ```
3. If unrecoverable, delete and recreate:
   ```bash
   rm data/weather.db
   python run_data_collection.py
   ```

---

## ML Models

### "No temperature model available"

**Symptom:** Dashboard prediction section shows this error.

**Solutions:**
1. Train models first:
   ```bash
   python src/ml_models/weather_predictor.py
   ```
2. Ensure you have enough data. At least 100 records are recommended; the model needs data to learn from.
3. Check that `models/` directory contains `.pkl` files after training.

### "Model metadata missing. Please retrain with updated pipeline."

**Symptom:** Predictions fail because the model was saved without metadata.

**Solutions:**
1. Retrain models with the current codebase (which saves metadata):
   ```bash
   python src/ml_models/weather_predictor.py
   ```
2. This happens when loading models saved by an older version of the code.

### "Feature mismatch: N expected features missing"

**Symptom:** The model expects features that the current data doesn't have.

**Solutions:**
1. This typically occurs when the model was trained on data with different columns than what's currently available.
2. Retrain the model on current data:
   ```bash
   python src/ml_models/weather_predictor.py
   ```
3. Check that enough data exists for lag features — lag features require historical data (e.g., `temperature_lag_24` needs 24 hours of history).

### Model Loading Errors (Pickle/Joblib)

**Symptom:** `ModuleNotFoundError`, `UnpicklingError`, or version warnings when loading models.

**Solutions:**
1. Ensure the same library versions are installed as when the model was trained. Check `requirements.txt`.
2. Particularly, `scikit-learn` and `xgboost` versions must match:
   ```bash
   pip install scikit-learn==1.3.0 xgboost==1.7.6
   ```
3. If versions can't be matched, retrain the model.

### Poor Model Accuracy

**Solutions:**
1. Collect more data — model accuracy improves significantly with more training data.
2. Check data quality: `python data_quality_report.py`
3. Look for data issues: too many missing values, very few cities, or short time span.
4. Review model comparison metrics in the dashboard's Model Information section.

---

## Dashboard

### "No weather data found for the selected time range"

**Symptom:** Dashboard displays an error instead of charts.

**Solutions:**
1. Run data collection: `python run_data_collection.py`
2. Try a wider time range from the sidebar dropdown
3. Verify the database has data: `python quick_db_check.py`
4. Check the database path — the dashboard defaults to `data/weather.db`

### Dashboard Won't Start

**Symptom:** `streamlit run dashboard.py` fails.

**Solutions:**
1. Install dependencies:
   ```bash
   pip install streamlit plotly pandas
   ```
2. Check for port conflicts:
   ```bash
   # Linux/macOS
   lsof -i :8501
   # Windows
   netstat -ano | findstr :8501
   ```
3. Try a different port:
   ```bash
   streamlit run dashboard.py --server.port 8502
   ```
4. Check for import errors by running:
   ```bash
   python -c "import dashboard"
   ```

### Charts Not Displaying

**Symptom:** Dashboard loads but charts are blank or show errors.

**Solutions:**
1. Ensure `plotly` is installed: `pip install plotly`
2. Check browser console for JavaScript errors
3. Try a hard refresh (Ctrl+Shift+R)
4. Verify the data has the expected columns:
   ```python
   import sqlite3, pandas as pd
   conn = sqlite3.connect("data/weather.db")
   df = pd.read_sql("SELECT * FROM weather_data LIMIT 5", conn)
   print(df.columns.tolist())
   conn.close()
   ```

### Auto-Refresh Not Working

**Symptom:** Dashboard doesn't update even with auto-refresh enabled.

**Solutions:**
1. Auto-refresh works by sleeping at the end of the script, then re-executing. The "Running..." indicator in the top right confirms it is active.
2. If the dashboard appears frozen, the sleep interval may be too long. Adjust the slider in the sidebar.
3. Click "Refresh Now" for an immediate manual refresh.
4. Cached data expires based on TTL (60 seconds for weather data). Even without auto-refresh, data refreshes when the cache expires and you interact with the page.

---

## Docker

### Container Fails to Start

**Symptom:** `docker compose up` shows errors or containers exit immediately.

**Solutions:**
1. Check logs: `docker compose logs dashboard`
2. Verify `.env` file exists in the project root
3. Ensure the `data/` and `models/` directories exist on the host:
   ```bash
   mkdir -p data models logs
   ```
4. Check Docker has enough resources (memory, disk)

### "Permission denied" on Volume Mounts

**Symptom:** Container can't read/write to mounted volumes.

**Solutions:**
1. On Linux, ensure the host directories are writable:
   ```bash
   chmod -R 777 data/ models/ logs/
   ```
2. Or run with the correct user:
   ```bash
   docker compose run --user $(id -u):$(id -g) dashboard
   ```

### Dashboard Not Accessible from Network

**Symptom:** `http://server-ip:8501` doesn't load from another machine.

**Solutions:**
1. Verify port mapping: `docker compose ps` should show `0.0.0.0:8501->8501/tcp`
2. Check firewall rules allow port 8501
3. On cloud instances, check security group / network ACL settings
4. Verify Streamlit is binding to `0.0.0.0` (the Dockerfile sets `--server.address=0.0.0.0`)

### Container Health Check Failing

**Symptom:** `docker inspect` shows unhealthy status.

**Solutions:**
1. The health check queries `http://localhost:8501/_stcore/health` inside the container
2. Allow 15 seconds for startup (configured in the Dockerfile)
3. Check if Streamlit is actually running: `docker compose exec dashboard ps aux`
4. Review container logs for errors: `docker compose logs dashboard`

### Rebuilding After Code Changes

```bash
# Rebuild images and restart
docker compose up -d --build

# Force a clean rebuild (no cache)
docker compose build --no-cache
docker compose up -d
```

---

## Data Quality

### Many Missing Values in Features

**Symptom:** Data quality report shows high null percentages.

**Solutions:**
1. Missing values in lag features (e.g., `temperature_lag_24`) are normal for the first 24 hours of data collection. Collect more data.
2. Missing values in `rain_1h` and `snow_1h` are expected — these fields are only present in API responses when rain/snow is occurring.
3. The data processor forward-fills gaps up to 3 consecutive missing values and interpolates the rest.

### Duplicate Records

**Symptom:** Same city+timestamp appears multiple times.

**Solutions:**
1. The UPSERT pattern (`INSERT OR REPLACE`) should prevent duplicates. If duplicates exist, they may be from before this feature was added.
2. Remove existing duplicates:
   ```sql
   DELETE FROM weather_data
   WHERE id NOT IN (
       SELECT MIN(id)
       FROM weather_data
       GROUP BY city, timestamp
   );
   ```

### Out-of-Range Values

**Symptom:** Temperature shows 300+ degrees or humidity exceeds 100%.

**Solutions:**
1. Check the API response format — OpenWeatherMap returns Kelvin by default. The collector converts to Celsius using `units=metric` parameter. Verify this parameter is being sent.
2. Run a quality check:
   ```bash
   python data_quality_report.py
   ```
3. Remove outliers manually:
   ```sql
   DELETE FROM weather_data WHERE temperature > 60 OR temperature < -60;
   ```

### Gaps in Time Series

**Symptom:** Charts show periods with no data points.

**Solutions:**
1. The collector may have been stopped during those periods. Check collection logs.
2. The `RateLimiter` or network issues may have caused skipped cycles.
3. Gaps don't break the pipeline — lag features will have nulls which are handled by `handle_missing_values()`.
4. For historical backfill, OpenWeatherMap's historical API is a paid feature.
