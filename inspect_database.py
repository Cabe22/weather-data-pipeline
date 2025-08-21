"""
Database inspection and data quality checks
"""
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

def inspect_database(db_path="data/weather.db"):
    """Inspect database structure and content"""
    print("🔍 Inspecting Weather Database")
    print("="*50)
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Database info
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"📋 Tables: {[table[0] for table in tables]}")
        
        # Weather data table info
        cursor.execute("PRAGMA table_info(weather_data);")
        columns = cursor.fetchall()
        print(f"📊 Columns: {len(columns)}")
        
        # Data summary
        df = pd.read_sql_query("SELECT * FROM weather_data ORDER BY timestamp DESC", conn)
        
        if len(df) > 0:
            print(f"\n📈 Data Summary:")
            print(f"   Total records: {len(df):,}")
            print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"   Cities: {df['city'].nunique()} ({', '.join(df['city'].unique()[:5])}{'...' if df['city'].nunique() > 5 else ''})")
            print(f"   Temperature range: {df['temperature'].min():.1f}°C to {df['temperature'].max():.1f}°C")
            print(f"   Missing values: {df.isnull().sum().sum()}")
            
            # Recent data
            recent = df.head(10)
            print(f"\n📊 Recent Records:")
            print(recent[['city', 'timestamp', 'temperature', 'humidity', 'weather_main']].to_string(index=False))
            
        else:
            print("⚠️  No data found in database!")
        
        conn.close()
        return len(df) > 0
        
    except Exception as e:
        print(f"❌ Database inspection failed: {e}")
        return False

if __name__ == "__main__":
    inspect_database()