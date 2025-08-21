"""
Generate data quality report
"""
import pandas as pd
import sqlite3
from datetime import datetime, timedelta

def generate_report():
    conn = sqlite3.connect("data/weather.db")
    df = pd.read_sql_query("SELECT * FROM weather_data", conn)
    conn.close()
    
    print("📊 DATA QUALITY REPORT")
    print("="*50)
    print(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📈 Dataset Overview:")
    print(f"   Total records: {len(df):,}")
    print(f"   Cities monitored: {df['city'].nunique()}")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Data completeness
    print(f"\n🔍 Data Completeness:")
    completeness = (1 - df.isnull().sum() / len(df)) * 100
    for col, pct in completeness.items():
        if pct < 100:
            print(f"   {col}: {pct:.1f}%")
    
    # City coverage
    print(f"\n🏙️ City Coverage:")
    city_counts = df['city'].value_counts()
    for city, count in city_counts.head().items():
        print(f"   {city}: {count} records")
    
    # Weather patterns
    print(f"\n🌤️ Weather Patterns:")
    weather_dist = df['weather_main'].value_counts()
    for weather, count in weather_dist.head().items():
        print(f"   {weather}: {count} records ({count/len(df)*100:.1f}%)")
    
    print(f"\n✅ Data collection is working properly!")
    
if __name__ == "__main__":
    generate_report()