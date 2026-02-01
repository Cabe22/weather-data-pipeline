# Deployment Guide

Instructions for deploying the Weather Data Pipeline locally, with Docker, and in cloud environments.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Environment Variables](#environment-variables)
- [Post-Deployment Verification](#post-deployment-verification)

---

## Prerequisites

- **Python 3.10+**
- **OpenWeatherMap API key** (free tier: [openweathermap.org/api](https://openweathermap.org/api))
- **Docker & Docker Compose** (for containerized deployment)
- **Git**

---

## Local Development

### 1. Clone and Install

```bash
git clone https://github.com/Cabe22/weather-data-pipeline.git
cd weather-data-pipeline
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example environment file and set your API key:

```bash
cp .env.example .env
```

Edit `.env`:

```
OPENWEATHER_API_KEY=your_actual_api_key
DATABASE_PATH=data/weather.db
LOG_LEVEL=INFO
UPDATE_INTERVAL=1800
```

### 3. Initialize Data Collection

Run an initial data collection to populate the database:

```bash
python run_data_collection.py
```

This creates `data/weather.db` and collects weather data for all configured cities. The default configuration monitors 10 major US cities.

### 4. Verify Data

```bash
python quick_db_check.py
```

This displays record counts, city coverage, and recent observations.

### 5. Train ML Models

Once you have sufficient data (100+ records recommended):

```bash
python src/ml_models/weather_predictor.py
```

Models are saved as PKL files in the `models/` directory.

### 6. Launch Dashboard

```bash
streamlit run dashboard.py
```

The dashboard opens at `http://localhost:8501`.

### 7. Continuous Collection (Optional)

For ongoing data collection, run the collector in a separate terminal or as a background process:

```bash
# Foreground (blocks terminal)
python run_data_collection.py

# Background (Linux/macOS)
nohup python run_data_collection.py > logs/collector.log 2>&1 &

# Background (Windows PowerShell)
Start-Process python -ArgumentList "run_data_collection.py" -WindowStyle Hidden
```

---

## Docker Deployment

The project includes a multi-stage Dockerfile and Docker Compose configuration for containerized deployment.

### Quick Start

```bash
docker compose up -d
```

This starts two services:
- **dashboard** - Streamlit web UI on port 8501
- **collector** - Background data collection

### Build and Run Manually

```bash
# Build the image
docker build -t weather-pipeline .

# Run dashboard only
docker run -d \
  --name weather-dashboard \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  --env-file .env \
  weather-pipeline

# Run collector only
docker run -d \
  --name weather-collector \
  -v $(pwd)/data:/app/data \
  --env-file .env \
  --entrypoint python \
  weather-pipeline run_data_collection.py
```

### Docker Compose Services

The `docker-compose.yml` defines two services:

```yaml
services:
  dashboard:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
    env_file:
      - .env
    restart: unless-stopped

  collector:
    build: .
    entrypoint: ["python", "run_data_collection.py"]
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    env_file:
      - .env
    restart: unless-stopped
    depends_on:
      - dashboard
```

**Volume mounts:**

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./data` | `/app/data` | SQLite database persistence |
| `./models` | `/app/models` | Trained model files |
| `./logs` | `/app/logs` | Application logs |

### Health Check

The dashboard container includes a health check that verifies the Streamlit server is responding:

```
Interval: 30s
Timeout: 10s
Start period: 15s
Retries: 3
```

Check container health:

```bash
docker inspect --format='{{.State.Health.Status}}' weather-pipeline-dashboard-1
```

### Managing Containers

```bash
# View logs
docker compose logs -f dashboard
docker compose logs -f collector

# Restart a service
docker compose restart dashboard

# Stop all services
docker compose down

# Rebuild after code changes
docker compose up -d --build
```

### Dockerfile Details

The Dockerfile uses a multi-stage build to minimize image size:

1. **Builder stage** (`python:3.10-slim`): Installs gcc and compiles Python dependencies
2. **Runtime stage** (`python:3.10-slim`): Copies only compiled packages and application code

The resulting image excludes build tools, test files, documentation, and version control artifacts (see `.dockerignore`).

---

## Cloud Deployment

### AWS EC2

1. Launch an EC2 instance (t3.small or larger recommended)
2. Install Docker and Docker Compose
3. Clone the repository and configure `.env`
4. Run `docker compose up -d`
5. Open port 8501 in the security group

```bash
# On EC2 instance
sudo yum install docker git -y
sudo systemctl start docker
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

git clone https://github.com/Cabe22/weather-data-pipeline.git
cd weather-data-pipeline
cp .env.example .env
# Edit .env with your API key
docker compose up -d
```

### AWS ECS / Fargate

For production, consider deploying the dashboard and collector as separate ECS tasks:

- Use an EFS volume for shared SQLite database access
- Set environment variables via ECS task definitions or Secrets Manager
- Configure an Application Load Balancer for the dashboard on port 8501

### Azure Container Instances

```bash
az container create \
  --resource-group myResourceGroup \
  --name weather-dashboard \
  --image weather-pipeline \
  --ports 8501 \
  --environment-variables OPENWEATHER_API_KEY=your_key
```

### General Cloud Considerations

- **Persistent storage**: The SQLite database must be on a persistent volume. Mount a host volume, EFS, or equivalent.
- **Secrets management**: Use your cloud provider's secrets manager for the API key rather than `.env` files.
- **HTTPS**: Place a reverse proxy (Nginx, ALB, Cloudflare) in front of the Streamlit dashboard for TLS termination.
- **Monitoring**: Integrate container health checks with your cloud monitoring (CloudWatch, Azure Monitor, etc.).
- **Scaling note**: SQLite supports a single writer at a time. For multi-instance deployments, consider migrating to PostgreSQL.

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENWEATHER_API_KEY` | Yes | None | API key from OpenWeatherMap |
| `DATABASE_PATH` | No | `data/weather.db` | Path to SQLite database file |
| `LOG_LEVEL` | No | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `UPDATE_INTERVAL` | No | `1800` | Data collection interval in seconds |

### Obtaining an API Key

1. Sign up at [openweathermap.org](https://openweathermap.org)
2. Navigate to **API Keys** in your account
3. Copy the key and set it in your `.env` file

The free tier allows 60 calls/minute and 1,000,000 calls/month. The default configuration (10 cities, 30-minute intervals) uses approximately 480 calls/day.

---

## Post-Deployment Verification

### Check Data Collection

```bash
python quick_db_check.py
```

Expected output: record counts per city and recent timestamps.

### Check Dashboard

Open `http://localhost:8501` (or your server's address) and verify:

1. Key metrics display (temperature, humidity, city count)
2. Temperature trend chart renders with data points
3. City comparison charts populate
4. Raw data table shows recent records

### Check Model Predictions

If models are trained, verify from the dashboard:

1. The "Machine Learning Predictions" section appears
2. Selecting a city and clicking "Generate Prediction" returns results
3. Model Information section shows training metadata

### Run Data Quality Report

```bash
python data_quality_report.py
```

Displays data completeness, coverage statistics, and weather pattern distributions.
