"""
Interactive Weather Dashboard using Streamlit
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List
import os
from src.data_processing.data_processor import WeatherDataProcessor
from src.ml_models.weather_predictor import WeatherPredictor
from src.ml_models.model_registry import ModelRegistry
import logging
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Weather Data Pipeline Dashboard",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stPlotlyChart {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(db_path: str = "data/weather.db", hours: int = 168) -> pd.DataFrame:
    """Load weather data from database"""
    conn = sqlite3.connect(db_path)
    
    query = f"""
        SELECT * FROM weather_data
        WHERE datetime(timestamp) > datetime('now', '-{hours} hours')
        ORDER BY timestamp DESC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df

@st.cache_resource
def load_predictor(model_dir: str = "models/") -> WeatherPredictor:
    """Load trained ML models via WeatherPredictor"""
    predictor = WeatherPredictor(model_dir=model_dir)
    if os.path.exists(model_dir):
        try:
            predictor.load_models()
        except Exception as e:
            logger.warning(f"Could not load models: {e}")
            st.warning(f"Could not load models from {model_dir}: {e}")
    return predictor

@st.cache_data(ttl=300)
def engineer_features_for_prediction(db_path: str = "data/weather.db") -> pd.DataFrame:
    """Run full feature engineering pipeline for prediction"""
    processor = WeatherDataProcessor(db_path=db_path)
    try:
        df = processor.load_data()
        if df.empty:
            return df
        df = processor.create_time_features(df)
        df = processor.create_lag_features(df)
        df = processor.create_weather_indices(df)
        df = processor.create_interaction_features(df)
        df = processor.handle_missing_values(df)
        df = processor.encode_categorical_features(df)
        df = processor.create_target_variable(df)
        return df  # DO NOT drop NaN target rows
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        return pd.DataFrame()

def create_temperature_chart(df: pd.DataFrame, city: str = None) -> go.Figure:
    """Create temperature time series chart"""
    
    if city:
        df_filtered = df[df['city'] == city]
    else:
        df_filtered = df
    
    fig = go.Figure()
    
    for city_name in df_filtered['city'].unique():
        city_data = df_filtered[df_filtered['city'] == city_name]
        
        fig.add_trace(go.Scatter(
            x=city_data['timestamp'],
            y=city_data['temperature'],
            mode='lines+markers',
            name=city_name,
            line=dict(width=2),
            marker=dict(size=4)
        ))
    
    fig.update_layout(
        title="Temperature Over Time",
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        hovermode='x unified',
        height=400
    )
    
    return fig

def create_weather_metrics(df: pd.DataFrame) -> Dict:
    """Calculate weather metrics"""
    
    latest_data = df.groupby('city').first().reset_index()
    
    metrics = {
        'avg_temp': df['temperature'].mean(),
        'max_temp': df['temperature'].max(),
        'min_temp': df['temperature'].min(),
        'avg_humidity': df['humidity'].mean(),
        'avg_pressure': df['pressure'].mean(),
        'cities_monitored': df['city'].nunique(),
        'total_records': len(df),
        'latest_update': df['timestamp'].max()
    }
    
    return metrics

def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create correlation heatmap"""
    
    numeric_cols = ['temperature', 'feels_like', 'humidity', 'pressure', 
                   'wind_speed', 'cloudiness', 'visibility']
    
    # Filter columns that exist in dataframe
    cols = [col for col in numeric_cols if col in df.columns]
    
    corr_matrix = df[cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Feature Correlation Matrix",
        height=500,
        width=600
    )
    
    return fig

def create_weather_distribution(df: pd.DataFrame) -> go.Figure:
    """Create weather condition distribution chart"""
    
    weather_counts = df['weather_main'].value_counts()
    
    fig = go.Figure(data=[
        go.Bar(
            x=weather_counts.index,
            y=weather_counts.values,
            marker_color='lightblue',
            text=weather_counts.values,
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Weather Condition Distribution",
        xaxis_title="Weather Condition",
        yaxis_title="Count",
        height=400
    )
    
    return fig

def create_city_comparison(df: pd.DataFrame) -> go.Figure:
    """Create city comparison chart"""
    
    city_stats = df.groupby('city').agg({
        'temperature': 'mean',
        'humidity': 'mean',
        'wind_speed': 'mean',
        'pressure': 'mean'
    }).round(2)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Average Temperature', 'Average Humidity', 
                       'Average Wind Speed', 'Average Pressure')
    )
    
    # Temperature
    fig.add_trace(
        go.Bar(x=city_stats.index, y=city_stats['temperature'], 
               marker_color='orange', showlegend=False),
        row=1, col=1
    )
    
    # Humidity
    fig.add_trace(
        go.Bar(x=city_stats.index, y=city_stats['humidity'], 
               marker_color='blue', showlegend=False),
        row=1, col=2
    )
    
    # Wind Speed
    fig.add_trace(
        go.Bar(x=city_stats.index, y=city_stats['wind_speed'], 
               marker_color='green', showlegend=False),
        row=2, col=1
    )
    
    # Pressure
    fig.add_trace(
        go.Bar(x=city_stats.index, y=city_stats['pressure'], 
               marker_color='purple', showlegend=False),
        row=2, col=2
    )
    
    fig.update_layout(height=600, title_text="City Weather Comparison")
    fig.update_xaxes(tickangle=45)
    
    return fig

def predict_temperature(predictor: WeatherPredictor,
                        engineered_df: pd.DataFrame, city: str) -> Dict:
    """Make temperature predictions using full feature pipeline"""
    if 'temperature' not in predictor.best_models:
        return {'error': 'No temperature model available. Train a model first.'}
    if engineered_df.empty:
        return {'error': 'No engineered feature data available.'}
    city_data = engineered_df[engineered_df['city'] == city]
    if city_data.empty:
        return {'error': f'No data available for city: {city}'}
    latest_row = city_data.sort_values('timestamp', ascending=False).iloc[0:1]
    if predictor.feature_columns is None:
        return {'error': 'Model metadata missing. Please retrain with updated pipeline.'}
    missing = [c for c in predictor.feature_columns if c not in latest_row.columns]
    if missing:
        return {'error': f'Feature mismatch: {len(missing)} expected features missing.'}
    try:
        prediction = predictor.predict(latest_row, model_type='temperature')[0]
        current_temp = latest_row['temperature'].values[0]
        return {
            'current_temp': current_temp,
            'predicted_temp': prediction,
            'change': prediction - current_temp,
            'features_used': len(predictor.feature_columns),
            'has_scaler': predictor.scaler is not None
        }
    except Exception as e:
        logger.error(f"Prediction failed for {city}: {e}")
        return {'error': f'Prediction failed: {str(e)}'}

# Main Dashboard
def main():
    st.markdown('<h1 class="main-header">🌤️ Weather Data Pipeline Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Dashboard Controls")
    
    # Time range selector
    time_range = st.sidebar.selectbox(
        "Select Time Range",
        options=[24, 48, 72, 168, 336],
        format_func=lambda x: f"Last {x} hours" if x < 168 else f"Last {x//24} days",
        index=2
    )
    
    # Load data
    df = load_data(hours=time_range)
    
    if df.empty:
        st.error("No data available. Please run the data collection script first.")
        st.stop()
    
    # Load models
    predictor = load_predictor()
    
    # City selector
    cities = df['city'].unique()
    selected_city = st.sidebar.selectbox("Select City", options=['All'] + list(cities))
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (every 60 seconds)")
    
    if auto_refresh:
        st.rerun()
    
    # Metrics
    st.header("📊 Key Metrics")
    metrics = create_weather_metrics(df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Average Temperature",
            value=f"{metrics['avg_temp']:.1f}°C",
            delta=f"Max: {metrics['max_temp']:.1f}°C"
        )
    
    with col2:
        st.metric(
            label="Average Humidity",
            value=f"{metrics['avg_humidity']:.0f}%"
        )
    
    with col3:
        st.metric(
            label="Cities Monitored",
            value=metrics['cities_monitored']
        )
    
    with col4:
        st.metric(
            label="Total Records",
            value=f"{metrics['total_records']:,}"
        )
    
    # Temperature Chart
    st.header("🌡️ Temperature Trends")
    
    if selected_city != 'All':
        temp_fig = create_temperature_chart(df, selected_city)
    else:
        temp_fig = create_temperature_chart(df)
    
    st.plotly_chart(temp_fig, use_container_width=True)
    
    # Two column layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("🌦️ Weather Conditions")
        weather_fig = create_weather_distribution(df)
        st.plotly_chart(weather_fig, use_container_width=True)
    
    with col2:
        st.header("🔗 Correlations")
        corr_fig = create_correlation_heatmap(df)
        st.plotly_chart(corr_fig, use_container_width=True)
    
    # City Comparison
    st.header("🏙️ City Comparison")
    city_comp_fig = create_city_comparison(df)
    st.plotly_chart(city_comp_fig, use_container_width=True)
    
    # ML Predictions Section
    if predictor.best_models:
        st.header("🤖 Machine Learning Predictions")

        col1, col2 = st.columns(2)

        with col1:
            pred_city = st.selectbox("Select city for prediction", cities)

        with col2:
            if st.button("Generate Prediction"):
                with st.spinner("Engineering features..."):
                    engineered_df = engineer_features_for_prediction()
                prediction = predict_temperature(predictor, engineered_df, pred_city)

                if 'error' in prediction:
                    st.error(prediction['error'])
                else:
                    st.success(f"**Prediction for {pred_city}:**")
                    st.write(f"Current Temperature: {prediction['current_temp']:.1f}°C")
                    st.write(f"Predicted Temperature (24h): {prediction['predicted_temp']:.1f}°C")

                    if prediction['change'] > 0:
                        st.write(f"Expected Change: 🔴 +{prediction['change']:.1f}°C")
                    else:
                        st.write(f"Expected Change: 🔵 {prediction['change']:.1f}°C")
                    st.caption(
                        f"Features used: {prediction['features_used']} | "
                        f"Scaler: {'active' if prediction['has_scaler'] else 'none'}"
                    )
    else:
        st.info("No trained models found. Train models to enable predictions.")

    # Model Information Section
    if predictor.model_metadata:
        st.header("Model Information")

        for model_type, meta in predictor.model_metadata.items():
            with st.expander(f"{model_type.title()} Model Details", expanded=False):
                info_cols = st.columns(3)
                trained_at = meta.get('trained_at', 'Unknown')
                if trained_at != 'Unknown':
                    try:
                        trained_at = datetime.fromisoformat(trained_at).strftime('%Y-%m-%d %H:%M UTC')
                    except ValueError:
                        pass
                info_cols[0].metric("Trained", trained_at)
                info_cols[1].metric("Algorithm", meta.get('best_algorithm', 'N/A'))
                info_cols[2].metric("Features", meta.get('num_features', 'N/A'))

                sample_cols = st.columns(2)
                sample_cols[0].metric("Training Samples", f"{meta.get('training_samples', 'N/A'):,}" if isinstance(meta.get('training_samples'), int) else 'N/A')
                sample_cols[1].metric("Test Samples", f"{meta.get('test_samples', 'N/A'):,}" if isinstance(meta.get('test_samples'), int) else 'N/A')

                metrics = meta.get('metrics', {})
                if metrics:
                    st.subheader("Performance Metrics")
                    metric_cols = st.columns(len(metrics))
                    for i, (k, v) in enumerate(metrics.items()):
                        label = k.replace('_', ' ').title()
                        metric_cols[i].metric(label, f"{v:.4f}" if isinstance(v, float) else str(v))

                # Per-algorithm breakdown (temperature models)
                all_models = meta.get('all_models', {})
                if all_models:
                    st.subheader("All Algorithms")
                    rows = []
                    for algo, m in all_models.items():
                        rows.append({
                            "Algorithm": algo,
                            "Test R2": f"{m['test_r2']:.4f}",
                            "Test MSE": f"{m['test_mse']:.4f}",
                            "Test MAE": f"{m['test_mae']:.4f}",
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Model Version History & Comparison
    registry_path = os.path.join("models", "registry.json")
    if os.path.exists(registry_path):
        registry = ModelRegistry(registry_path)
        all_versions = registry.list_versions()

        if all_versions:
            st.header("Model Version History")

            version_rows = []
            for v in all_versions:
                meta = v.get("metadata", {})
                metrics = meta.get("metrics", {})
                primary_metric = metrics.get("test_r2", metrics.get("roc_auc", ""))
                version_rows.append({
                    "Version": v["version"],
                    "Type": v["model_type"],
                    "Registered": v.get("registered_at", "")[:19].replace("T", " "),
                    "Algorithm": meta.get("best_algorithm", ""),
                    "Primary Metric": f"{primary_metric:.4f}" if isinstance(primary_metric, float) else str(primary_metric),
                })
            st.dataframe(pd.DataFrame(version_rows), use_container_width=True, hide_index=True)

            # Comparison selector
            if len(all_versions) >= 2:
                st.subheader("Compare Versions")
                version_labels = [f"{v['version']} ({v['model_type']})" for v in all_versions]
                comp_cols = st.columns(2)
                sel_a = comp_cols[0].selectbox("Version A", version_labels, index=len(version_labels) - 2)
                sel_b = comp_cols[1].selectbox("Version B", version_labels, index=len(version_labels) - 1)
                ver_a = sel_a.split(" ")[0]
                ver_b = sel_b.split(" ")[0]
                if ver_a != ver_b:
                    comparison = registry.compare(ver_a, ver_b)
                    if comparison and comparison["metrics"]:
                        rows = []
                        for metric_name, vals in comparison["metrics"].items():
                            label = metric_name.replace('_', ' ').title()
                            va = vals.get(ver_a)
                            vb = vals.get(ver_b)
                            delta = vals.get("delta")
                            rows.append({
                                "Metric": label,
                                ver_a: f"{va:.4f}" if isinstance(va, float) else str(va),
                                ver_b: f"{vb:.4f}" if isinstance(vb, float) else str(vb),
                                "Delta": f"{delta:+.4f}" if isinstance(delta, float) else "",
                            })
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    else:
                        st.info("No metrics available to compare.")
                else:
                    st.info("Select two different versions to compare.")

    # Data Table
    with st.expander("📋 View Raw Data"):
        # Filter data if city selected
        if selected_city != 'All':
            display_df = df[df['city'] == selected_city]
        else:
            display_df = df
        
        # Show recent records
        st.dataframe(
            display_df.head(100),
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name=f"weather_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"**Last Updated:** {metrics['latest_update'].strftime('%Y-%m-%d %H:%M:%S')}"
        if pd.notnull(metrics['latest_update']) else "No data"
    )

if __name__ == "__main__":
    main()