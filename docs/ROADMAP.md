# Weather Data Pipeline - Development Roadmap

**Created:** January 28, 2026
**Target Completion:** Q1 2026

---

## Overview

This roadmap organizes all tasks from PROJECT_STATUS.md into logical development phases with git branches, complexity estimates, and dependencies.

### Complexity Legend
- **Simple** - < 1 hour, straightforward changes
- **Medium** - 1-4 hours, requires some design decisions
- **Complex** - 4+ hours, significant implementation or refactoring

### Branch Naming Convention
- `fix/` - Bug fixes and corrections
- `feature/` - New functionality
- `test/` - Testing infrastructure and test cases
- `docs/` - Documentation updates

---

## Phase 1: Critical Bug Fixes

### Branch: `fix/critical-bugs`

**Priority:** CRITICAL
**Complexity:** Simple
**Dependencies:** None
**Estimated Tasks:** 4

Bundle all quick critical fixes into a single branch for immediate deployment.

| Task | File(s) | Change |
|------|---------|--------|
| Fix package init typo | `src/__int__.py` | Rename to `src/__init__.py` |
| Fix deprecated pandas method | `src/data_processing/data_processor.py:181` | Change `fillna(method='ffill')` to `ffill()` |
| Switch to HTTPS for API | `src/data_collection/weather_collector.py:36` | Change `http://` to `https://` |
| Remove global warning suppression | `src/ml_models/weather_predictor.py:20` | Remove or scope `filterwarnings` |

**Checklist:**
- [ ] Create branch from main
- [ ] Fix all 4 issues
- [ ] Run existing tests to verify no regressions
- [ ] Create PR with fixes
- [ ] Merge to main

---

## Phase 2: Feature Scaling Fix

### Branch: `fix/feature-scaling`

**Priority:** HIGH
**Complexity:** Medium
**Dependencies:** Phase 1 (critical bugs)
**Estimated Tasks:** 3

Fix the mismatch between training and inference scaling.

| Task | File(s) | Change |
|------|---------|--------|
| Save scaler with model | `src/ml_models/weather_predictor.py` | Save fitted StandardScaler alongside model |
| Load scaler at inference | `src/ml_models/weather_predictor.py` | Load and apply scaler before prediction |
| Update model save/load methods | `src/ml_models/weather_predictor.py` | Bundle scaler + model + feature list |

**Implementation Notes:**
- Create a model artifact dictionary: `{'model': model, 'scaler': scaler, 'features': feature_list}`
- Update `save_models()` and `load_models()` methods
- Ensure backward compatibility with existing saved models

---

## Phase 3: Dashboard Prediction Fix

### Branch: `fix/dashboard-prediction`

**Priority:** CRITICAL
**Complexity:** Complex
**Dependencies:** Phase 2 (feature scaling)
**Estimated Tasks:** 4

Fix the broken prediction feature in the dashboard.

| Task | File(s) | Change |
|------|---------|--------|
| Integrate WeatherDataProcessor | `dashboard.py:240-261` | Use processor for feature engineering |
| Apply correct feature set | `dashboard.py` | Match training features (40+) |
| Apply feature scaling | `dashboard.py` | Load and apply saved scaler |
| Improve prediction error handling | `dashboard.py` | Add specific error messages |

**Implementation Notes:**
- Import and instantiate WeatherDataProcessor
- Create prediction pipeline that mirrors training pipeline
- Handle edge cases (insufficient data for lag features)
- Display confidence intervals if available

---

## Phase 4: Testing Infrastructure

### Branch: `test/pytest-setup`

**Priority:** HIGH
**Complexity:** Medium
**Dependencies:** Phase 1 (critical bugs)
**Estimated Tasks:** 5

Set up proper testing framework with pytest.

| Task | File(s) | Change |
|------|---------|--------|
| Add pytest to requirements | `requirements.txt` | Add pytest, pytest-cov |
| Create pytest configuration | `pytest.ini` or `pyproject.toml` | Configure test discovery, coverage |
| Create test fixtures | `tests/conftest.py` | Shared fixtures (mock data, temp DB) |
| Convert test_api.py | `tests/test_api.py` | Rewrite with pytest assertions |
| Convert test_data_collection.py | `tests/test_data_collection.py` | Rewrite with pytest assertions |

**New Files:**
- `tests/conftest.py` - Shared fixtures
- `pytest.ini` or section in `pyproject.toml`

---

## Phase 5: Core Module Tests

### Branch: `test/core-modules`

**Priority:** HIGH
**Complexity:** Medium
**Dependencies:** Phase 4 (pytest setup)
**Estimated Tasks:** 6

Add comprehensive tests for all core modules.

| Task | File(s) | Change |
|------|---------|--------|
| Test WeatherDataProcessor | `tests/test_data_processor.py` | Feature engineering, scaling, edge cases |
| Test WeatherPredictor | `tests/test_weather_predictor.py` | Training, prediction, model persistence |
| Test WeatherCollector | `tests/test_weather_collector.py` | API mocking, database operations |
| Test config classes | `tests/test_config.py` | Configuration validation |
| Add edge case tests | `tests/` | Empty data, missing values, invalid input |
| Add integration tests | `tests/test_integration.py` | End-to-end pipeline test |

**Coverage Target:** 80%

---

## Phase 6: CI/CD Pipeline

### Branch: `feature/ci-cd-pipeline`

**Priority:** HIGH
**Complexity:** Medium
**Dependencies:** Phase 5 (core module tests)
**Estimated Tasks:** 4

Implement automated testing and code quality checks.

| Task | File(s) | Change |
|------|---------|--------|
| Create test workflow | `.github/workflows/test.yml` | Run pytest on push/PR |
| Add code coverage | `.github/workflows/test.yml` | Upload coverage to Codecov |
| Add linting (black) | `.github/workflows/lint.yml` | Check code formatting |
| Add type checking (mypy) | `.github/workflows/lint.yml` | Static type analysis |

**New Files:**
- `.github/workflows/test.yml`
- `.github/workflows/lint.yml`
- `.pre-commit-config.yaml` (optional, for local checks)

**Workflow Triggers:**
- On push to main
- On pull request to main

---

## Phase 7: API Resilience

### Branch: `feature/api-resilience`

**Priority:** MEDIUM
**Complexity:** Medium
**Dependencies:** Phase 1 (critical bugs)
**Estimated Tasks:** 4

Improve API reliability with retry logic and rate limiting.

| Task | File(s) | Change |
|------|---------|--------|
| Add retry with backoff | `src/data_collection/weather_collector.py` | Implement exponential backoff |
| Add rate limiting | `src/data_collection/weather_collector.py` | Track calls, throttle if needed |
| Add request timeout handling | `src/data_collection/weather_collector.py` | Graceful timeout handling |
| Log API metrics | `src/data_collection/weather_collector.py` | Track success/failure rates |

**Implementation Notes:**
- Use `tenacity` library or custom retry decorator
- Default: 3 retries with exponential backoff (1s, 2s, 4s)
- Rate limit: 60 calls/minute (OpenWeatherMap free tier)

---

## Phase 8: Data Quality

### Branch: `feature/data-quality`

**Priority:** MEDIUM
**Complexity:** Medium
**Dependencies:** Phase 1 (critical bugs)
**Estimated Tasks:** 3

Add data deduplication and validation.

| Task | File(s) | Change |
|------|---------|--------|
| Add deduplication | `src/data_collection/weather_collector.py` | UPSERT pattern for (city, timestamp) |
| Add response validation | `src/data_collection/weather_collector.py` | Validate API response structure |
| Add data quality checks | `src/data_processing/data_processor.py` | Flag anomalies, log warnings |

**Implementation Notes:**
- Use SQLite `INSERT OR REPLACE` or `ON CONFLICT` clause
- Validate required fields before database insert
- Add data quality report generation

---

## Phase 9: Docker Deployment

### Branch: `feature/docker-deployment`

**Priority:** MEDIUM
**Complexity:** Medium
**Dependencies:** Phase 6 (CI/CD pipeline)
**Estimated Tasks:** 4

Containerize the application for deployment.

| Task | File(s) | Change |
|------|---------|--------|
| Create Dockerfile | `Dockerfile` | Python 3.10-slim, multi-stage build |
| Create docker-compose.yml | `docker-compose.yml` | Dashboard service, volumes |
| Add .dockerignore | `.dockerignore` | Exclude unnecessary files |
| Add container healthcheck | `Dockerfile` | Streamlit health endpoint |

**New Files:**
- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`

**Docker Compose Services:**
```yaml
services:
  dashboard:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    env_file:
      - .env
```

---

## Phase 10: Model Versioning

### Branch: `feature/model-versioning`

**Priority:** MEDIUM
**Complexity:** Medium
**Dependencies:** Phase 2 (feature scaling)
**Estimated Tasks:** 4

Track model versions and metadata.

| Task | File(s) | Change |
|------|---------|--------|
| Add model metadata | `src/ml_models/weather_predictor.py` | Save training date, metrics, features |
| Create model registry | `src/ml_models/model_registry.py` | Track model versions |
| Display model info | `dashboard.py` | Show model version, training date |
| Add model comparison | `dashboard.py` | Compare model performance |

**New Files:**
- `src/ml_models/model_registry.py`
- `models/registry.json` (model metadata store)

---

## Phase 11: Dashboard Improvements

### Branch: `feature/dashboard-improvements`

**Priority:** LOW
**Complexity:** Medium
**Dependencies:** Phase 3 (dashboard prediction fix)
**Estimated Tasks:** 3

Improve dashboard UX and performance.

| Task | File(s) | Change |
|------|---------|--------|
| Improve error messages | `dashboard.py` | Specific errors with troubleshooting hints |
| Optimize auto-refresh | `dashboard.py` | Incremental updates, smarter caching |
| Add loading states | `dashboard.py` | Spinners for long operations |

---

## Phase 12: Documentation

### Branch: `docs/comprehensive-docs`

**Priority:** LOW
**Complexity:** Medium
**Dependencies:** Phase 9 (Docker deployment)
**Estimated Tasks:** 4

Complete project documentation.

| Task | File(s) | Change |
|------|---------|--------|
| API documentation | `docs/API.md` | Document all classes and methods |
| Deployment guide | `docs/DEPLOYMENT.md` | Local, Docker, cloud deployment |
| Database migration guide | `docs/DATABASE.md` | Schema versioning, migrations |
| Troubleshooting guide | `docs/TROUBLESHOOTING.md` | Common issues and solutions |

**New Files:**
- `docs/API.md`
- `docs/DEPLOYMENT.md`
- `docs/DATABASE.md`
- `docs/TROUBLESHOOTING.md`

---

## Phase 13: Security & Monitoring

### Branch: `feature/security-monitoring`

**Priority:** LOW
**Complexity:** Medium
**Dependencies:** Phase 6 (CI/CD pipeline)
**Estimated Tasks:** 3

Add security scanning and performance monitoring.

| Task | File(s) | Change |
|------|---------|--------|
| Add security scanning | `.github/workflows/security.yml` | Bandit, safety checks |
| Add dependency scanning | `.github/workflows/security.yml` | Check for vulnerabilities |
| Add performance metrics | `src/` | Track inference time, query performance |

**New Files:**
- `.github/workflows/security.yml`

---

## Phase 14: Advanced ML

### Branch: `feature/temporal-validation`

**Priority:** LOW
**Complexity:** Complex
**Dependencies:** Phase 5 (core module tests)
**Estimated Tasks:** 3

Implement proper time series validation.

| Task | File(s) | Change |
|------|---------|--------|
| Temporal train/test split | `src/ml_models/weather_predictor.py` | Time-based splitting |
| Walk-forward validation | `src/ml_models/weather_predictor.py` | Rolling window validation |
| Add data leakage checks | `tests/test_weather_predictor.py` | Verify no future data in training |

---

## Dependency Graph

```
Phase 1: fix/critical-bugs
    │
    ├──► Phase 2: fix/feature-scaling
    │        │
    │        └──► Phase 3: fix/dashboard-prediction
    │                  │
    │                  └──► Phase 11: feature/dashboard-improvements
    │
    ├──► Phase 4: test/pytest-setup
    │        │
    │        └──► Phase 5: test/core-modules
    │                  │
    │                  ├──► Phase 6: feature/ci-cd-pipeline
    │                  │        │
    │                  │        ├──► Phase 9: feature/docker-deployment
    │                  │        │        │
    │                  │        │        └──► Phase 12: docs/comprehensive-docs
    │                  │        │
    │                  │        └──► Phase 13: feature/security-monitoring
    │                  │
    │                  └──► Phase 14: feature/temporal-validation
    │
    ├──► Phase 7: feature/api-resilience
    │
    ├──► Phase 8: feature/data-quality
    │
    └──► Phase 10: feature/model-versioning (also depends on Phase 2)
```

---

## Timeline Summary

| Week | Phases | Branches |
|------|--------|----------|
| 1 | 1, 2, 3 | `fix/critical-bugs`, `fix/feature-scaling`, `fix/dashboard-prediction` |
| 2 | 4, 5 | `test/pytest-setup`, `test/core-modules` |
| 3 | 6, 7 | `feature/ci-cd-pipeline`, `feature/api-resilience` |
| 4 | 8, 9 | `feature/data-quality`, `feature/docker-deployment` |
| 5+ | 10-14 | Remaining features and documentation |

---

## Quick Reference: All Branches

| Phase | Branch Name | Priority | Complexity | Dependencies |
|-------|-------------|----------|------------|--------------|
| 1 | `fix/critical-bugs` | Critical | Simple | None |
| 2 | `fix/feature-scaling` | High | Medium | Phase 1 |
| 3 | `fix/dashboard-prediction` | Critical | Complex | Phase 2 |
| 4 | `test/pytest-setup` | High | Medium | Phase 1 |
| 5 | `test/core-modules` | High | Medium | Phase 4 |
| 6 | `feature/ci-cd-pipeline` | High | Medium | Phase 5 |
| 7 | `feature/api-resilience` | Medium | Medium | Phase 1 |
| 8 | `feature/data-quality` | Medium | Medium | Phase 1 |
| 9 | `feature/docker-deployment` | Medium | Medium | Phase 6 |
| 10 | `feature/model-versioning` | Medium | Medium | Phase 2 |
| 11 | `feature/dashboard-improvements` | Low | Medium | Phase 3 |
| 12 | `docs/comprehensive-docs` | Low | Medium | Phase 9 |
| 13 | `feature/security-monitoring` | Low | Medium | Phase 6 |
| 14 | `feature/temporal-validation` | Low | Complex | Phase 5 |

---

## Recommended Starting Point

**Start with:** `fix/critical-bugs` (Phase 1)

**Rationale:**
1. No dependencies - can start immediately
2. Simple complexity - quick wins
3. Fixes blocking issues (package imports, deprecation warnings)
4. Unblocks all other phases
5. Can be completed and merged in < 1 hour

**Command to begin:**
```bash
git checkout -b fix/critical-bugs
```

---

*This roadmap should be reviewed weekly and updated as tasks are completed.*
