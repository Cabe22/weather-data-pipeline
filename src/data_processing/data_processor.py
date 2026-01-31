"""
Data Processing and Feature Engineering Module
Processes raw weather data and creates features for ML models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Tuple, List, Optional
import sqlite3
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class WeatherDataProcessor:
    """Process and engineer features from weather data"""
    
    def __init__(self, db_path: str = "data/weather.db"):
        self.db_path = db_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, 
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None) -> pd.DataFrame:
        """Load weather data from database"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM weather_data"
        conditions = []
        
        if start_date:
            conditions.append(f"timestamp >= '{start_date}'")
        if end_date:
            conditions.append(f"timestamp <= '{end_date}'")
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        query += " ORDER BY city, timestamp"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logger.info(f"Loaded {len(df)} records from database")
        return df
        
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df = df.copy()
        
        # Extract time components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Time of day categories
        df['time_of_day'] = pd.cut(df['hour'], 
                                   bins=[0, 6, 12, 18, 24],
                                   labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                   include_lowest=True)
        
        # Is weekend
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        logger.info("Created time-based features")
        return df
        
    def create_lag_features(self, df: pd.DataFrame, 
                           lag_hours: List[int] = [1, 3, 6, 12, 24]) -> pd.DataFrame:
        """Create lag features for time series analysis"""
        df = df.copy()
        df = df.sort_values(['city', 'timestamp'])
        
        # Features to lag
        lag_features = ['temperature', 'humidity', 'pressure', 'wind_speed']
        
        for feature in lag_features:
            for lag in lag_hours:
                col_name = f'{feature}_lag_{lag}h'
                df[col_name] = df.groupby('city')[feature].shift(lag)
                
        # Rolling statistics
        for feature in lag_features:
            # 24-hour rolling mean
            df[f'{feature}_rolling_mean_24h'] = (
                df.groupby('city')[feature]
                .rolling(window=24, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )
            
            # 24-hour rolling std
            df[f'{feature}_rolling_std_24h'] = (
                df.groupby('city')[feature]
                .rolling(window=24, min_periods=2)
                .std()
                .reset_index(0, drop=True)
            )
            
        logger.info("Created lag and rolling features")
        return df
        
    def create_weather_indices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific weather indices"""
        df = df.copy()
        
        # Heat Index (simplified version)
        df['heat_index'] = (
            -8.78469475556 + 
            1.61139411 * df['temperature'] + 
            2.33854883889 * df['humidity'] - 
            0.14611605 * df['temperature'] * df['humidity'] - 
            0.012308094 * df['temperature']**2 - 
            0.0164248277778 * df['humidity']**2 + 
            0.002211732 * df['temperature']**2 * df['humidity'] + 
            0.00072546 * df['temperature'] * df['humidity']**2 - 
            0.000003582 * df['temperature']**2 * df['humidity']**2
        )
        
        # Wind Chill (for temperatures below 10°C)
        mask = df['temperature'] < 10
        df.loc[mask, 'wind_chill'] = (
            13.12 + 0.6215 * df.loc[mask, 'temperature'] - 
            11.37 * df.loc[mask, 'wind_speed']**0.16 + 
            0.3965 * df.loc[mask, 'temperature'] * df.loc[mask, 'wind_speed']**0.16
        )
        df.loc[~mask, 'wind_chill'] = df.loc[~mask, 'temperature']
        
        # Discomfort Index
        df['discomfort_index'] = df['temperature'] - 0.55 * (1 - 0.01 * df['humidity']) * (df['temperature'] - 14.5)
        
        # Pressure change rate
        df['pressure_change'] = df.groupby('city')['pressure'].diff()
        
        # Temperature range
        df['temp_range'] = df['temp_max'] - df['temp_min']
        
        logger.info("Created weather indices")
        return df
        
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between variables"""
        df = df.copy()
        
        # Temperature-Humidity interaction
        df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
        
        # Wind-Temperature interaction
        df['wind_temp_interaction'] = df['wind_speed'] * df['temperature']
        
        # Pressure-Temperature interaction
        df['pressure_temp_interaction'] = df['pressure'] * df['temperature']
        
        # Cloud-Humidity interaction
        df['cloud_humidity_interaction'] = df['cloudiness'] * df['humidity']
        
        logger.info("Created interaction features")
        return df
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        df = df.copy()
        
        # Forward fill for time series continuity (within each city)
        time_cols = ['temperature', 'humidity', 'pressure', 'wind_speed']
        df[time_cols] = df.groupby('city')[time_cols].ffill(limit=3)
        
        # Fill remaining with interpolation
        df[time_cols] = df.groupby('city')[time_cols].apply(
            lambda x: x.interpolate(method='linear', limit_direction='both')
        )
        
        # Fill categorical with mode
        categorical_cols = ['weather_main', 'weather_description', 'time_of_day']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df.groupby('city')[col].fillna(
                    lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
                )
        
        # Drop rows with too many missing values
        df = df.dropna(thresh=len(df.columns) * 0.7)
        
        logger.info(f"Handled missing values. Remaining rows: {len(df)}")
        return df
        
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        df = df.copy()
        
        categorical_cols = ['city', 'country', 'weather_main', 'time_of_day']
        
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].fillna('Unknown'))
                else:
                    # Handle unseen categories
                    df[f'{col}_encoded'] = df[col].apply(
                        lambda x: self.label_encoders[col].transform([x])[0] 
                        if x in self.label_encoders[col].classes_ 
                        else -1
                    )
        
        logger.info("Encoded categorical features")
        return df
        
    def normalize_features(self, df: pd.DataFrame, 
                          exclude_cols: List[str] = None) -> pd.DataFrame:
        """Normalize numerical features"""
        df = df.copy()
        
        if exclude_cols is None:
            exclude_cols = ['id', 'timestamp', 'created_at']
            
        # Get numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        # Fit and transform
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        
        logger.info(f"Normalized {len(numerical_cols)} numerical features")
        return df
        
    def create_target_variable(self, df: pd.DataFrame, 
                               target_hours: int = 24) -> pd.DataFrame:
        """Create target variable for prediction"""
        df = df.copy()
        df = df.sort_values(['city', 'timestamp'])
        
        # Future temperature (24 hours ahead)
        df['temperature_future'] = df.groupby('city')['temperature'].shift(-target_hours)
        
        # Temperature change category
        df['temp_change'] = df['temperature_future'] - df['temperature']
        df['temp_change_category'] = pd.cut(
            df['temp_change'],
            bins=[-np.inf, -2, 2, np.inf],
            labels=['Decrease', 'Stable', 'Increase']
        )
        
        # Rain prediction (will it rain in next 24 hours?)
        df['will_rain'] = (
            df.groupby('city')['rain_1h']
            .rolling(window=target_hours, min_periods=1)
            .sum()
            .shift(-target_hours)
            .reset_index(0, drop=True) > 0
        ).astype(int)
        
        logger.info("Created target variables")
        return df
        
    def process_pipeline(self, 
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """Execute complete processing pipeline"""
        logger.info("Starting data processing pipeline")
        
        # Load data
        df = self.load_data(start_date, end_date)

        # Run quality checks on raw data
        self.run_quality_checks(df)

        # Create features
        df = self.create_time_features(df)
        df = self.create_lag_features(df)
        df = self.create_weather_indices(df)
        df = self.create_interaction_features(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Create target variable
        df = self.create_target_variable(df)
        
        # Remove rows without target
        df = df.dropna(subset=['temperature_future'])
        
        logger.info(f"Pipeline complete. Final dataset shape: {df.shape}")
        return df
        
    # Physically plausible ranges for weather measurements
    VALID_RANGES = {
        'temperature': (-60.0, 60.0),
        'feels_like': (-70.0, 70.0),
        'temp_min': (-60.0, 60.0),
        'temp_max': (-60.0, 60.0),
        'humidity': (0, 100),
        'pressure': (870, 1084),
        'wind_speed': (0.0, 120.0),
        'wind_deg': (0, 360),
        'cloudiness': (0, 100),
        'visibility': (0, 100000),
    }

    def run_quality_checks(self, df: pd.DataFrame) -> Dict:
        """Run data quality checks and return a report.

        Checks performed:
        - Duplicate (city, timestamp) rows
        - Missing values per column
        - Out-of-range values for known weather measurements
        - Rows with a high proportion of nulls

        Returns a dict summarising the issues found. Warnings are
        also emitted via the module logger.
        """
        report: Dict = {
            'total_rows': len(df),
            'duplicates': {},
            'missing': {},
            'out_of_range': {},
            'high_null_rows': 0,
        }

        # --- Duplicates -------------------------------------------------------
        if 'city' in df.columns and 'timestamp' in df.columns:
            dup_mask = df.duplicated(subset=['city', 'timestamp'], keep=False)
            dup_count = dup_mask.sum()
            if dup_count > 0:
                logger.warning(f"Found {dup_count} duplicate (city, timestamp) rows")
            report['duplicates'] = {'city_timestamp': int(dup_count)}

        # --- Missing values ----------------------------------------------------
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        for col, count in missing.items():
            pct = count / len(df) * 100
            logger.warning(f"Column '{col}' has {count} missing values ({pct:.1f}%)")
            report['missing'][col] = {'count': int(count), 'percent': round(pct, 1)}

        # --- Out-of-range values -----------------------------------------------
        for col, (lo, hi) in self.VALID_RANGES.items():
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors='coerce')
            below = (series < lo).sum()
            above = (series > hi).sum()
            if below > 0 or above > 0:
                total = below + above
                logger.warning(
                    f"Column '{col}': {total} values outside [{lo}, {hi}] "
                    f"({below} below, {above} above)"
                )
                report['out_of_range'][col] = {
                    'below': int(below),
                    'above': int(above),
                    'range': [lo, hi],
                }

        # --- High-null rows ----------------------------------------------------
        null_threshold = len(df.columns) * 0.3  # >30 % nulls in a row
        row_nulls = df.isnull().sum(axis=1)
        high_null = (row_nulls > null_threshold).sum()
        if high_null > 0:
            logger.warning(f"{high_null} rows have >30% null values")
        report['high_null_rows'] = int(high_null)

        if (not report['missing'] and not report['out_of_range']
                and report['duplicates'].get('city_timestamp', 0) == 0
                and high_null == 0):
            logger.info("Data quality checks passed — no issues found")
        else:
            logger.info("Data quality checks complete — see warnings above")

        return report

    def get_feature_importance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic feature statistics for importance analysis"""
        feature_cols = [col for col in df.columns 
                       if col not in ['id', 'timestamp', 'created_at', 'city', 
                                     'country', 'weather_description']]
        
        stats = []
        for col in feature_cols:
            if df[col].dtype in [np.float64, np.int64]:
                stats.append({
                    'feature': col,
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'nulls': df[col].isnull().sum(),
                    'correlation_with_temp': df[col].corr(df['temperature'])
                })
        
        return pd.DataFrame(stats)

# Example usage
if __name__ == "__main__":
    processor = WeatherDataProcessor("data/weather.db")
    
    # Process data
    processed_df = processor.process_pipeline()
    
    print(f"Processed dataset shape: {processed_df.shape}")
    print(f"Features created: {processed_df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(processed_df.head())
    
    # Get feature importance
    importance = processor.get_feature_importance_data(processed_df)
    print(f"\nTop correlated features with temperature:")
    print(importance.nlargest(10, 'correlation_with_temp')[['feature', 'correlation_with_temp']])