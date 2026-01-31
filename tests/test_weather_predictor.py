"""
Tests for src/ml_models/weather_predictor.py
"""

import matplotlib
matplotlib.use("Agg")

import os
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch

from src.ml_models.weather_predictor import WeatherPredictor


# ---------------------------------------------------------------------------
# Class-scoped fixtures for expensive training operations
# ---------------------------------------------------------------------------

@pytest.fixture(scope="class")
def trained_temp_predictor(tmp_path_factory):
    """Train temperature models once for the whole class."""
    np.random.seed(99)
    n = 60
    df = pd.DataFrame({
        "temperature": np.random.normal(20, 5, n),
        "humidity": np.random.uniform(30, 90, n),
        "pressure": np.random.normal(1013, 5, n),
        "wind_speed": np.random.uniform(0, 10, n),
        "cloudiness": np.random.uniform(0, 100, n),
        "hour_sin": np.random.uniform(-1, 1, n),
        "hour_cos": np.random.uniform(-1, 1, n),
        "temperature_lag_1h": np.random.normal(20, 5, n),
        "humidity_lag_1h": np.random.uniform(30, 90, n),
        "temperature_rolling_mean_24h": np.random.normal(20, 3, n),
        "heat_index": np.random.normal(25, 5, n),
        "wind_chill": np.random.normal(18, 4, n),
        "temp_humidity_interaction": np.random.normal(1200, 300, n),
        "city_encoded": np.random.choice([0, 1], n),
        "weather_main_encoded": np.random.choice([0, 1, 2, 3], n),
    })
    df["temperature_future"] = df["temperature"] + np.random.normal(0, 2, n)
    df["will_rain"] = (np.random.rand(n) > 0.5).astype(int)

    model_dir = str(tmp_path_factory.mktemp("models")) + "/"
    predictor = WeatherPredictor(model_dir=model_dir)
    X, y = predictor.prepare_features(df, "temperature_future")
    results = predictor.train_temperature_models(X, y)
    return predictor, X, y, results, df


@pytest.fixture(scope="class")
def trained_rain_predictor(tmp_path_factory):
    """Train rain classifier once for the whole class."""
    np.random.seed(99)
    n = 60
    df = pd.DataFrame({
        "temperature": np.random.normal(20, 5, n),
        "humidity": np.random.uniform(30, 90, n),
        "pressure": np.random.normal(1013, 5, n),
        "wind_speed": np.random.uniform(0, 10, n),
        "cloudiness": np.random.uniform(0, 100, n),
        "hour_sin": np.random.uniform(-1, 1, n),
        "hour_cos": np.random.uniform(-1, 1, n),
        "temperature_lag_1h": np.random.normal(20, 5, n),
        "humidity_lag_1h": np.random.uniform(30, 90, n),
        "temperature_rolling_mean_24h": np.random.normal(20, 3, n),
        "heat_index": np.random.normal(25, 5, n),
        "wind_chill": np.random.normal(18, 4, n),
        "temp_humidity_interaction": np.random.normal(1200, 300, n),
        "city_encoded": np.random.choice([0, 1], n),
        "weather_main_encoded": np.random.choice([0, 1, 2, 3], n),
    })
    df["temperature_future"] = df["temperature"] + np.random.normal(0, 2, n)
    df["will_rain"] = (np.random.rand(n) > 0.5).astype(int)

    model_dir = str(tmp_path_factory.mktemp("rain_models")) + "/"
    predictor = WeatherPredictor(model_dir=model_dir)
    X, y = predictor.prepare_features(df, "will_rain")
    results = predictor.train_rain_classifier(X, y)
    return predictor, X, y, results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPrepareFeatures:
    def test_returns_tuple(self, predictor_instance, predictor_training_dataframe):
        X, y = predictor_instance.prepare_features(predictor_training_dataframe, "temperature_future")
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_excludes_non_feature_cols(self, predictor_instance, predictor_training_dataframe):
        X, _ = predictor_instance.prepare_features(predictor_training_dataframe, "temperature_future")
        excluded = {"temperature_future", "will_rain", "temp_change", "temp_change_category",
                     "city", "country", "timestamp", "weather_description", "time_of_day"}
        assert excluded.isdisjoint(set(X.columns))

    def test_no_nan_in_features(self, predictor_instance, predictor_training_dataframe):
        X, _ = predictor_instance.prepare_features(predictor_training_dataframe, "temperature_future")
        assert X.isna().sum().sum() == 0

    def test_stores_feature_columns(self, predictor_instance, predictor_training_dataframe):
        predictor_instance.prepare_features(predictor_training_dataframe, "temperature_future")
        assert predictor_instance.feature_columns is not None
        assert len(predictor_instance.feature_columns) > 0

    def test_scaler_fitted(self, predictor_instance, predictor_training_dataframe):
        predictor_instance.prepare_features(predictor_training_dataframe, "temperature_future")
        assert predictor_instance.scaler is not None
        assert hasattr(predictor_instance.scaler, "mean_")


@pytest.mark.slow
class TestTrainTemperatureModels:
    def test_all_six_models_present(self, trained_temp_predictor):
        _, _, _, results, _ = trained_temp_predictor
        expected = {"linear_regression", "ridge", "random_forest",
                    "gradient_boosting", "xgboost", "neural_network"}
        assert expected == set(results.keys())

    def test_metrics_keys(self, trained_temp_predictor):
        _, _, _, results, _ = trained_temp_predictor
        required = {"train_mse", "test_mse", "train_mae", "test_mae",
                     "train_r2", "test_r2", "cv_score_mean", "cv_score_std"}
        for name, res in results.items():
            assert required.issubset(set(res.keys())), f"{name} missing metric keys"

    def test_best_model_stored(self, trained_temp_predictor):
        predictor, _, _, _, _ = trained_temp_predictor
        assert "temperature" in predictor.best_models

    def test_feature_importance_stored(self, trained_temp_predictor):
        predictor, _, _, _, _ = trained_temp_predictor
        # At least tree-based models should have feature importance
        assert len(predictor.feature_importance) > 0

    def test_finite_metrics(self, trained_temp_predictor):
        _, _, _, results, _ = trained_temp_predictor
        for name, res in results.items():
            assert np.isfinite(res["test_mse"]), f"{name} test_mse not finite"
            assert np.isfinite(res["test_r2"]), f"{name} test_r2 not finite"

    def test_prediction_lengths(self, trained_temp_predictor):
        _, X, _, results, _ = trained_temp_predictor
        for name, res in results.items():
            # Predictions should be for the test split (~20% of 60 = 12)
            assert len(res["predictions"]) == len(res["actual"])


class TestTrainRainClassifier:
    def test_classifier_trained(self, trained_rain_predictor):
        predictor, _, _, results = trained_rain_predictor
        assert "rain" in predictor.best_models

    def test_classification_report(self, trained_rain_predictor):
        _, _, _, results = trained_rain_predictor
        assert "classification_report" in results
        report = results["classification_report"]
        assert "accuracy" in report

    def test_roc_auc_range(self, trained_rain_predictor):
        _, _, _, results = trained_rain_predictor
        assert 0 <= results["roc_auc"] <= 1

    def test_binary_predictions(self, trained_rain_predictor):
        _, _, _, results = trained_rain_predictor
        assert set(np.unique(results["predictions"])).issubset({0, 1})


@pytest.mark.slow
class TestHyperparameterTuning:
    def test_xgboost_best_params(self, predictor_instance, predictor_training_dataframe):
        X, y = predictor_instance.prepare_features(
            predictor_training_dataframe, "temperature_future"
        )
        results = predictor_instance.hyperparameter_tuning(X, y, "xgboost")
        assert "best_params" in results
        assert "n_estimators" in results["best_params"]

    def test_random_forest_best_params(self, predictor_instance, predictor_training_dataframe):
        X, y = predictor_instance.prepare_features(
            predictor_training_dataframe, "temperature_future"
        )
        results = predictor_instance.hyperparameter_tuning(X, y, "random_forest")
        assert "best_params" in results
        assert "max_depth" in results["best_params"]

    def test_tuned_model_predicts(self, predictor_instance, predictor_training_dataframe):
        X, y = predictor_instance.prepare_features(
            predictor_training_dataframe, "temperature_future"
        )
        results = predictor_instance.hyperparameter_tuning(X, y, "xgboost")
        preds = results["model"].predict(X)
        assert len(preds) == len(X)


class TestCreateEnsembleModel:
    def test_returns_results(self, predictor_instance, predictor_training_dataframe):
        X, y = predictor_instance.prepare_features(
            predictor_training_dataframe, "temperature_future"
        )
        results = predictor_instance.create_ensemble_model(X, y)
        assert isinstance(results, dict)

    def test_finite_mse(self, predictor_instance, predictor_training_dataframe):
        X, y = predictor_instance.prepare_features(
            predictor_training_dataframe, "temperature_future"
        )
        results = predictor_instance.create_ensemble_model(X, y)
        assert np.isfinite(results["test_mse"])

    def test_prediction_length(self, predictor_instance, predictor_training_dataframe):
        X, y = predictor_instance.prepare_features(
            predictor_training_dataframe, "temperature_future"
        )
        results = predictor_instance.create_ensemble_model(X, y)
        assert len(results["predictions"]) == len(results["actual"])

    def test_three_base_models(self, predictor_instance, predictor_training_dataframe):
        X, y = predictor_instance.prepare_features(
            predictor_training_dataframe, "temperature_future"
        )
        results = predictor_instance.create_ensemble_model(X, y)
        assert len(results["base_models"]) == 3


class TestPredict:
    def test_predict_after_training(self, trained_temp_predictor):
        predictor, _, _, _, df = trained_temp_predictor
        preds = predictor.predict(df, model_type="temperature")
        assert len(preds) == len(df)

    def test_value_error_without_model(self, predictor_instance, predictor_training_dataframe):
        with pytest.raises(ValueError, match="No trained model"):
            predictor_instance.predict(predictor_training_dataframe, model_type="temperature")

    def test_value_error_without_scaler(self, predictor_instance, predictor_training_dataframe):
        predictor_instance.best_models["temperature"] = "dummy"
        with pytest.raises(ValueError, match="No scaler"):
            predictor_instance.predict(predictor_training_dataframe, model_type="temperature")

    def test_output_shape(self, trained_temp_predictor):
        predictor, _, _, _, df = trained_temp_predictor
        preds = predictor.predict(df, model_type="temperature")
        assert isinstance(preds, np.ndarray)
        assert preds.ndim == 1


class TestSaveLoadModels:
    def test_pkl_files_created(self, trained_temp_predictor):
        predictor, _, _, _, _ = trained_temp_predictor
        predictor.save_models()
        files = os.listdir(predictor.model_dir)
        pkl_files = [f for f in files if f.endswith(".pkl")]
        assert len(pkl_files) >= 1

    def test_load_restores_models(self, trained_temp_predictor):
        predictor, _, _, _, _ = trained_temp_predictor
        predictor.save_models()

        new_predictor = WeatherPredictor(model_dir=predictor.model_dir)
        new_predictor.load_models()
        assert "temperature" in new_predictor.best_models

    def test_loaded_model_predicts_same(self, trained_temp_predictor):
        predictor, _, _, _, df = trained_temp_predictor
        predictor.save_models()

        original_preds = predictor.predict(df, model_type="temperature")

        new_predictor = WeatherPredictor(model_dir=predictor.model_dir)
        new_predictor.load_models()
        loaded_preds = new_predictor.predict(df, model_type="temperature")

        np.testing.assert_array_almost_equal(original_preds, loaded_preds)

    def test_scaler_and_features_persisted(self, trained_temp_predictor):
        predictor, _, _, _, _ = trained_temp_predictor
        predictor.save_models()

        new_predictor = WeatherPredictor(model_dir=predictor.model_dir)
        new_predictor.load_models()
        assert new_predictor.scaler is not None
        assert new_predictor.feature_columns is not None

    def test_backward_compat_bare_model(self, tmp_path):
        """Loading a bare model (not dict) should still work."""
        import joblib
        from sklearn.linear_model import LinearRegression

        model_dir = str(tmp_path / "compat_models") + "/"
        os.makedirs(model_dir)
        model = LinearRegression()
        model.fit([[1], [2], [3]], [1, 2, 3])
        joblib.dump(model, model_dir + "legacy_model.pkl")

        predictor = WeatherPredictor(model_dir=model_dir)
        predictor.load_models()
        assert "legacy" in predictor.best_models


class TestPlotMethods:
    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.show")
    def test_model_comparison_returns_figure(self, mock_show, mock_save, trained_temp_predictor):
        predictor, _, _, results, _ = trained_temp_predictor
        fig = predictor.plot_model_comparison(results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.show")
    def test_feature_importance_returns_figure(self, mock_show, mock_save, trained_temp_predictor):
        predictor, _, _, _, _ = trained_temp_predictor
        fig = predictor.plot_feature_importance()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPredictorEdgeCases:
    def test_high_null_cols_dropped(self, predictor_instance):
        np.random.seed(42)
        n = 20
        df = pd.DataFrame({
            "temperature": np.random.normal(20, 5, n),
            "humidity": np.random.uniform(30, 90, n),
            "mostly_null": [np.nan] * (n - 2) + [1.0, 2.0],
        })
        df["temperature_future"] = df["temperature"] + 1
        X, _ = predictor_instance.prepare_features(df, "temperature_future")
        assert "mostly_null" not in X.columns

    def test_constant_target(self, predictor_instance):
        np.random.seed(42)
        n = 60
        df = pd.DataFrame({
            "temperature": np.random.normal(20, 5, n),
            "humidity": np.random.uniform(30, 90, n),
            "pressure": np.random.normal(1013, 5, n),
        })
        df["temperature_future"] = 20.0  # constant
        X, y = predictor_instance.prepare_features(df, "temperature_future")
        # Should not crash even with constant target
        assert len(X) == n

    def test_missing_feature_col(self, trained_temp_predictor):
        predictor, _, _, _, df = trained_temp_predictor
        # Remove a feature column from input
        df_partial = df.drop(columns=[predictor.feature_columns[0]])
        with pytest.raises(KeyError):
            predictor.predict(df_partial, model_type="temperature")

    def test_save_with_no_models(self, predictor_instance):
        # Should not crash, just save nothing
        predictor_instance.save_models()
        files = os.listdir(predictor_instance.model_dir)
        pkl_files = [f for f in files if f.endswith(".pkl")]
        assert len(pkl_files) == 0
