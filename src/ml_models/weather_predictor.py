"""
Machine Learning Models for Weather Prediction
Includes multiple models for temperature and rain prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, roc_auc_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Any
import logging

logger = logging.getLogger(__name__)

class WeatherPredictor:
    """ML models for weather prediction"""
    
    def __init__(self, model_dir: str = "models/"):
        self.model_dir = model_dir
        self.models = {}
        self.best_models = {}
        self.feature_importance = {}
        self.scaler = None
        self.feature_columns = None
        
    def prepare_features(self, df: pd.DataFrame, 
                        target_col: str = 'temperature_future') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for model training"""
        
        # Features to exclude
        exclude_cols = [
            'id', 'timestamp', 'created_at', 'city', 'country',
            'weather_description', 'temperature_future', 'temp_change',
            'temp_change_category', 'will_rain', 'time_of_day'
        ]
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove columns with too many nulls
        null_threshold = 0.3
        for col in feature_cols.copy():
            if df[col].isnull().sum() / len(df) > null_threshold:
                feature_cols.remove(col)
                logger.info(f"Removed {col} due to high null percentage")
        
        X = df[feature_cols].fillna(0)
        y = df[target_col]

        self.feature_columns = list(X.columns)
        self.scaler = StandardScaler()
        X = pd.DataFrame(self.scaler.fit_transform(X), columns=self.feature_columns, index=X.index)

        logger.info(f"Prepared {X.shape[1]} features for training")
        return X, y
        
    def train_temperature_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train multiple regression models for temperature prediction"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        # Define models
        models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'neural_network': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42
            )
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                       scoring='neg_mean_squared_error')
            
            results[name] = {
                'model': model,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'cv_score_mean': -cv_scores.mean(),
                'cv_score_std': cv_scores.std(),
                'predictions': y_pred_test,
                'actual': y_test
            }
            
            logger.info(f"{name} - Test MSE: {test_mse:.4f}, Test R2: {test_r2:.4f}")
            
            # Store feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
        
        self.models['temperature'] = results
        
        # Select best model based on test R2 score
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
        self.best_models['temperature'] = results[best_model_name]['model']
        
        logger.info(f"Best temperature model: {best_model_name} with R2: {results[best_model_name]['test_r2']:.4f}")
        
        return results
    
    def train_rain_classifier(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train classification model for rain prediction"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training rain classifier. Class distribution: {y.value_counts().to_dict()}")
        
        # Train Random Forest Classifier
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        clf.fit(X_train, y_train)
        
        # Predictions
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results = {
            'model': clf,
            'classification_report': report,
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'prediction_proba': y_pred_proba,
            'actual': y_test
        }
        
        self.models['rain'] = results
        self.best_models['rain'] = clf
        
        # Store feature importance
        self.feature_importance['rain_classifier'] = pd.DataFrame({
            'feature': X.columns,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Rain classifier ROC-AUC: {roc_auc:.4f}")
        
        return results
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series, 
                            model_type: str = 'xgboost') -> Dict:
        """Perform hyperparameter tuning for best model"""
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        if model_type == 'xgboost':
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
            model = xgb.XGBRegressor(random_state=42)
            
        elif model_type == 'random_forest':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        logger.info(f"Starting hyperparameter tuning for {model_type}...")
        
        grid_search = GridSearchCV(
            model, param_grid, 
            cv=5, 
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_,
            'test_mse': mean_squared_error(y_test, y_pred),
            'test_r2': r2_score(y_test, y_pred),
            'model': best_model
        }
        
        logger.info(f"Best params: {results['best_params']}")
        logger.info(f"Test MSE: {results['test_mse']:.4f}, Test R2: {results['test_r2']:.4f}")
        
        return results
    
    def create_ensemble_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Create ensemble model combining multiple algorithms"""
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train base models
        base_models = [
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
            ('xgb', xgb.XGBRegressor(n_estimators=100, random_state=42))
        ]
        
        # Train base models and get predictions
        base_predictions_train = []
        base_predictions_test = []
        
        for name, model in base_models:
            model.fit(X_train, y_train)
            base_predictions_train.append(model.predict(X_train))
            base_predictions_test.append(model.predict(X_test))
        
        # Stack predictions
        stacked_train = np.column_stack(base_predictions_train)
        stacked_test = np.column_stack(base_predictions_test)
        
        # Train meta-model
        meta_model = LinearRegression()
        meta_model.fit(stacked_train, y_train)
        
        # Final predictions
        y_pred = meta_model.predict(stacked_test)
        
        results = {
            'base_models': base_models,
            'meta_model': meta_model,
            'test_mse': mean_squared_error(y_test, y_pred),
            'test_r2': r2_score(y_test, y_pred),
            'predictions': y_pred,
            'actual': y_test
        }
        
        logger.info(f"Ensemble model - Test MSE: {results['test_mse']:.4f}, Test R2: {results['test_r2']:.4f}")
        
        return results
    
    def predict(self, X: pd.DataFrame, model_type: str = 'temperature') -> np.ndarray:
        """Apply saved scaler and predict using a trained model"""
        if model_type not in self.best_models:
            raise ValueError(f"No trained model found for '{model_type}'")
        if self.scaler is None:
            raise ValueError("No scaler fitted. Train a model first or load one.")

        X_prepared = X[self.feature_columns].fillna(0)
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_prepared),
            columns=self.feature_columns, index=X_prepared.index
        )
        return self.best_models[model_type].predict(X_scaled)

    def save_models(self):
        """Save trained models to disk"""
        import os
        os.makedirs(self.model_dir, exist_ok=True)
        
        for name, model in self.best_models.items():
            filepath = f"{self.model_dir}{name}_model.pkl"
            artifact = {
                'model': model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }
            joblib.dump(artifact, filepath)
            logger.info(f"Saved {name} model to {filepath}")
    
    def load_models(self):
        """Load models from disk"""
        import os
        
        for filename in os.listdir(self.model_dir):
            if filename.endswith('_model.pkl'):
                name = filename.replace('_model.pkl', '')
                filepath = f"{self.model_dir}{filename}"
                loaded = joblib.load(filepath)
                if isinstance(loaded, dict):
                    self.best_models[name] = loaded['model']
                    self.scaler = loaded.get('scaler')
                    self.feature_columns = loaded.get('feature_columns')
                else:
                    # Backward compatibility: bare model from old saves
                    self.best_models[name] = loaded
                logger.info(f"Loaded {name} model from {filepath}")
    
    def plot_model_comparison(self, results: Dict):
        """Plot model performance comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # MSE comparison
        models = list(results.keys())
        train_mse = [results[m]['train_mse'] for m in models]
        test_mse = [results[m]['test_mse'] for m in models]
        
        ax = axes[0, 0]
        x = np.arange(len(models))
        width = 0.35
        ax.bar(x - width/2, train_mse, width, label='Train MSE')
        ax.bar(x + width/2, test_mse, width, label='Test MSE')
        ax.set_xlabel('Model')
        ax.set_ylabel('MSE')
        ax.set_title('Model MSE Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        
        # R2 comparison
        train_r2 = [results[m]['train_r2'] for m in models]
        test_r2 = [results[m]['test_r2'] for m in models]
        
        ax = axes[0, 1]
        ax.bar(x - width/2, train_r2, width, label='Train R2')
        ax.bar(x + width/2, test_r2, width, label='Test R2')
        ax.set_xlabel('Model')
        ax.set_ylabel('R2 Score')
        ax.set_title('Model R2 Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        
        # Best model predictions vs actual
        best_model = max(results.keys(), key=lambda x: results[x]['test_r2'])
        predictions = results[best_model]['predictions']
        actual = results[best_model]['actual']
        
        ax = axes[1, 0]
        ax.scatter(actual, predictions, alpha=0.5)
        ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Temperature')
        ax.set_ylabel('Predicted Temperature')
        ax.set_title(f'Best Model ({best_model}) - Predictions vs Actual')
        
        # Residuals plot
        residuals = predictions - actual
        ax = axes[1, 1]
        ax.hist(residuals, bins=50, edgecolor='black')
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        ax.set_title('Residuals Distribution')
        ax.axvline(x=0, color='r', linestyle='--')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_feature_importance(self, top_n: int = 20):
        """Plot feature importance for tree-based models"""
        fig, axes = plt.subplots(1, len(self.feature_importance), figsize=(15, 6))
        
        if len(self.feature_importance) == 1:
            axes = [axes]
        
        for idx, (name, importance_df) in enumerate(self.feature_importance.items()):
            top_features = importance_df.head(top_n)
            
            axes[idx].barh(range(len(top_features)), top_features['importance'])
            axes[idx].set_yticks(range(len(top_features)))
            axes[idx].set_yticklabels(top_features['feature'])
            axes[idx].set_xlabel('Importance')
            axes[idx].set_title(f'Feature Importance - {name}')
            axes[idx].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig

# Example usage
if __name__ == "__main__":
    from data_processing import WeatherDataProcessor
    
    # Load and process data
    processor = WeatherDataProcessor("data/weather.db")
    df = processor.process_pipeline()
    
    # Initialize predictor
    predictor = WeatherPredictor()
    
    # Prepare features for temperature prediction
    X_temp, y_temp = predictor.prepare_features(df, 'temperature_future')
    
    # Train temperature models
    temp_results = predictor.train_temperature_models(X_temp, y_temp)
    
    # Train rain classifier
    X_rain, y_rain = predictor.prepare_features(df, 'will_rain')
    rain_results = predictor.train_rain_classifier(X_rain, y_rain)
    
    # Hyperparameter tuning
    tuned_results = predictor.hyperparameter_tuning(X_temp, y_temp, 'xgboost')
    
    # Create ensemble model
    ensemble_results = predictor.create_ensemble_model(X_temp, y_temp)
    
    # Save models
    predictor.save_models()
    
    # Create visualizations
    predictor.plot_model_comparison(temp_results)
    predictor.plot_feature_importance()
    
    print("\nModel Training Complete!")
    print(f"Best Temperature Model R2: {max(temp_results.values(), key=lambda x: x['test_r2'])['test_r2']:.4f}")
    print(f"Rain Classifier ROC-AUC: {rain_results['roc_auc']:.4f}")