# Database Guide

Documentation for the SQLite database schema, data management, and migration patterns.

---

## Table of Contents

- [Overview](#overview)
- [Schema](#schema)
- [Indexes](#indexes)
- [Data Integrity](#data-integrity)
- [Querying Data](#querying-data)
- [Data Management](#data-management)
- [Migration Patterns](#migration-patterns)
- [Performance](#performance)

---

## Overview

The Weather Data Pipeline uses SQLite as its primary data store. The database file is located at `data/weather.db` by default (configurable via the `DATABASE_PATH` environment variable).

**Why SQLite:**
- Zero configuration, no server process
- Single-file database, easy to back up and move
- Sufficient for single-writer workloads (data collection runs serially)
- Built into Python's standard library

---

## Schema

### `weather_data` Table

```sql
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
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(city, timestamp)
);
```

### Column Reference

| Column | Type | Nullable | Description | Units |
|--------|------|----------|-------------|-------|
| `id` | INTEGER | No | Auto-incrementing primary key | - |
| `city` | TEXT | No | City name | - |
| `country` | TEXT | Yes | ISO country code (e.g., `US`) | - |
| `timestamp` | DATETIME | No | Observation time (UTC) | - |
| `temperature` | REAL | Yes | Current temperature | Celsius |
| `feels_like` | REAL | Yes | Perceived temperature | Celsius |
| `temp_min` | REAL | Yes | Minimum temperature in the area | Celsius |
| `temp_max` | REAL | Yes | Maximum temperature in the area | Celsius |
| `pressure` | INTEGER | Yes | Atmospheric pressure | hPa |
| `humidity` | INTEGER | Yes | Relative humidity | % (0-100) |
| `wind_speed` | REAL | Yes | Wind speed | m/s |
| `wind_deg` | INTEGER | Yes | Wind direction | degrees (0-360) |
| `cloudiness` | INTEGER | Yes | Cloud coverage | % (0-100) |
| `weather_main` | TEXT | Yes | Main condition (Clear, Rain, Snow, etc.) | - |
| `weather_description` | TEXT | Yes | Detailed description | - |
| `visibility` | INTEGER | Yes | Visibility distance | meters |
| `rain_1h` | REAL | Yes | Rain volume in last hour | mm |
| `snow_1h` | REAL | Yes | Snow volume in last hour | mm |
| `lat` | REAL | Yes | City latitude | degrees |
| `lon` | REAL | Yes | City longitude | degrees |
| `timezone` | INTEGER | Yes | UTC offset | seconds |
| `created_at` | DATETIME | Yes | Record insertion time | UTC (auto) |

### Valid Data Ranges

The `WeatherDataProcessor` enforces these physical plausibility ranges during quality checks:

| Measurement | Min | Max |
|-------------|-----|-----|
| Temperature | -60 C | 60 C |
| Feels Like | -70 C | 70 C |
| Pressure | 870 hPa | 1084 hPa |
| Humidity | 0% | 100% |
| Wind Speed | 0 m/s | 120 m/s |
| Cloudiness | 0% | 100% |
| Visibility | 0 m | 100,000 m |

---

## Indexes

```sql
CREATE INDEX IF NOT EXISTS idx_city_timestamp
ON weather_data(city, timestamp);
```

The composite index on `(city, timestamp)` optimizes:
- Filtering by city with time range queries (the dashboard's primary access pattern)
- Enforcing the UNIQUE constraint for deduplication
- Sorting by timestamp within a city

---

## Data Integrity

### Deduplication (UPSERT)

The collector uses `INSERT OR REPLACE` to prevent duplicate entries for the same city and timestamp:

```sql
INSERT OR REPLACE INTO weather_data
    (city, country, timestamp, temperature, ...)
VALUES (?, ?, ?, ?, ...)
```

If a record with the same `(city, timestamp)` already exists, it is replaced with the new data. This means re-running the collector for a time period that was already collected will update existing records rather than creating duplicates.

### API Response Validation

Before storing data, `WeatherCollector.validate_api_response()` checks:

1. Required fields are present (`main`, `wind`, `weather`, `name`)
2. Temperature values are within -100 C to 100 C
3. Humidity is between 0% and 100%
4. Wind speed is non-negative

Invalid responses are logged and skipped.

---

## Querying Data

### Using Python

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("data/weather.db")

# Recent data for all cities
df = pd.read_sql_query("""
    SELECT * FROM weather_data
    WHERE datetime(timestamp) > datetime('now', '-24 hours')
    ORDER BY timestamp DESC
""", conn)

# Average temperature by city
df = pd.read_sql_query("""
    SELECT city, AVG(temperature) as avg_temp, COUNT(*) as records
    FROM weather_data
    GROUP BY city
    ORDER BY avg_temp DESC
""", conn)

conn.close()
```

### Using the SQLite CLI

```bash
sqlite3 data/weather.db

-- Table info
.schema weather_data

-- Record count
SELECT COUNT(*) FROM weather_data;

-- Latest record per city
SELECT city, MAX(timestamp) as latest, temperature
FROM weather_data
GROUP BY city;

-- Data range
SELECT MIN(timestamp), MAX(timestamp) FROM weather_data;
```

### Using Project Utilities

```bash
# Detailed database inspection
python inspect_database.py

# Quick status check
python quick_db_check.py

# Data quality report
python data_quality_report.py
```

---

## Data Management

### Backup

SQLite databases are single files, making backup straightforward:

```bash
# Simple file copy (stop writes first for consistency)
cp data/weather.db data/backups/weather_$(date +%Y%m%d_%H%M%S).db

# Using SQLite's backup command (safe during writes)
sqlite3 data/weather.db ".backup data/backups/weather_backup.db"
```

For automated backups, use a cron job:

```bash
# Daily backup at midnight
0 0 * * * sqlite3 /path/to/data/weather.db ".backup /path/to/data/backups/weather_$(date +\%Y\%m\%d).db"
```

### Restore

```bash
# Stop any running collectors/dashboard first
cp data/backups/weather_backup.db data/weather.db
```

### Pruning Old Data

To remove data older than a certain date:

```sql
DELETE FROM weather_data
WHERE datetime(timestamp) < datetime('now', '-90 days');

-- Reclaim disk space after deletion
VACUUM;
```

### Export to CSV

```bash
sqlite3 -header -csv data/weather.db \
  "SELECT * FROM weather_data" > weather_export.csv
```

Or use the dashboard's built-in CSV download button.

---

## Migration Patterns

SQLite does not have built-in migration tooling. The following patterns can be used for schema changes.

### Adding a Column

```sql
ALTER TABLE weather_data ADD COLUMN uv_index REAL;
```

Existing rows will have `NULL` for the new column. The collector code must be updated to populate the new field.

### Renaming a Column (SQLite 3.25+)

```sql
ALTER TABLE weather_data RENAME COLUMN weather_main TO weather_condition;
```

### Dropping a Column (SQLite 3.35+)

```sql
ALTER TABLE weather_data DROP COLUMN snow_1h;
```

For older SQLite versions, use the table-rebuild approach:

```sql
-- 1. Create new table without the column
CREATE TABLE weather_data_new AS
SELECT id, city, country, timestamp, temperature, ...
FROM weather_data;

-- 2. Drop old table
DROP TABLE weather_data;

-- 3. Rename new table
ALTER TABLE weather_data_new RENAME TO weather_data;

-- 4. Recreate indexes
CREATE INDEX idx_city_timestamp ON weather_data(city, timestamp);
```

### Adding an Index

```sql
CREATE INDEX idx_weather_main ON weather_data(weather_main);
```

### Migration Script Pattern

For repeatable migrations, create numbered SQL scripts:

```
migrations/
  001_initial_schema.sql
  002_add_uv_index.sql
  003_add_air_quality.sql
```

Apply with:

```bash
sqlite3 data/weather.db < migrations/002_add_uv_index.sql
```

Track applied migrations in a metadata table:

```sql
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

INSERT INTO schema_migrations (version, description)
VALUES (2, 'Add UV index column');
```

---

## Performance

### Current Characteristics

- **Write pattern:** Sequential inserts, one city at a time (no concurrent writes)
- **Read pattern:** Time-range queries with city filtering (dashboard loads)
- **Typical size:** ~500 bytes per row, ~1 MB per 2,000 records

### Optimization Tips

1. **Use the index**: Always filter on `city` and/or `timestamp` in WHERE clauses
2. **Limit result sets**: Use `LIMIT` or time-range filters rather than `SELECT *`
3. **VACUUM periodically**: After large deletes, run `VACUUM` to reclaim space
4. **WAL mode**: For better concurrent read/write performance:
   ```sql
   PRAGMA journal_mode=WAL;
   ```
5. **Batch inserts**: When bulk-loading data, wrap in a transaction:
   ```python
   conn.execute("BEGIN TRANSACTION")
   for row in data:
       conn.execute("INSERT ...", row)
   conn.execute("COMMIT")
   ```

### Scaling Beyond SQLite

If the project outgrows SQLite (high write concurrency, multi-instance deployment, or very large datasets), consider migrating to PostgreSQL:

1. Export data to CSV
2. Create the same schema in PostgreSQL
3. Import data using `COPY` or `\copy`
4. Update connection strings in `WeatherConfig` and `WeatherDataProcessor`
5. Replace `INSERT OR REPLACE` with `INSERT ... ON CONFLICT ... DO UPDATE`
