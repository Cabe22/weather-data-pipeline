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
import joblib
from typing import Dict, List
import os

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
def load_models(model_dir: str = "models/") -> Dict:
    """Load trained ML models"""
    models = {}
    
    if os.path.exists(model_dir):
        for filename in os.listdir(model_dir):
            if filename.endswith('_model.pkl'):
                name = filename.replace('_model.pkl', '')
                filepath = os.path.join(model_dir, filename)
                try:
                    models[name] = joblib.load(filepath)
                except:
                    st.warning(f"Could not load model: {name}")
    
    return models

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

def predict_temperature(models: Dict, df: pd.DataFrame, city: str) -> Dict:
    """Make temperature predictions"""
    
    if 'temperature' not in models or df.empty:
        return None
    
    # Get latest data for city
    city_data = df[df['city'] == city].iloc[0:1]
    
    # Simple feature preparation (you'd use the full processing pipeline in production)
    features = ['temperature', 'humidity', 'pressure', 'wind_speed', 'cloudiness']
    X = city_data[features].fillna(0)
    
    try:
        prediction = models['temperature'].predict(X)[0]
        return {
            'current_temp': city_data['temperature'].values[0],
            'predicted_temp': prediction,
            'change': prediction - city_data['temperature'].values[0]
        }
    except:
        return None

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
    models = load_models()
    
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
    if models:
        st.header("🤖 Machine Learning Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pred_city = st.selectbox("Select city for prediction", cities)
        
        with col2:
            if st.button("Generate Prediction"):
                prediction = predict_temperature(models, df, pred_city)
                
                if prediction:
                    st.success(f"**Prediction for {pred_city}:**")
                    st.write(f"Current Temperature: {prediction['current_temp']:.1f}°C")
                    st.write(f"Predicted Temperature (24h): {prediction['predicted_temp']:.1f}°C")
                    
                    if prediction['change'] > 0:
                        st.write(f"Expected Change: 🔴 +{prediction['change']:.1f}°C")
                    else:
                        st.write(f"Expected Change: 🔵 {prediction['change']:.1f}°C")
                else:
                    st.error("Could not generate prediction. Check if models are trained.")
    
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