"""
Quick database verification script
"""
import sqlite3
import pandas as pd
from datetime import datetime

def check_database():
    """Check what's actually in the database"""
    print("🔍 Quick Database Check")
    print("="*40)
    
    try:
        conn = sqlite3.connect("data/weather.db")
        
        # Get total count
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM weather_data")
        total_count = cursor.fetchone()[0]
        print(f"📊 Total records: {total_count}")
        
        if total_count > 0:
            # Get latest records
            df = pd.read_sql_query("""
                SELECT city, timestamp, temperature, humidity, pressure, weather_main
                FROM weather_data 
                ORDER BY timestamp DESC 
                LIMIT 15
            """, conn)
            
            print(f"\n📈 Latest {len(df)} records:")
            print(df.to_string(index=False))
            
            # Get summary by city
            summary = pd.read_sql_query("""
                SELECT 
                    city, 
                    COUNT(*) as records,
                    AVG(temperature) as avg_temp,
                    MAX(timestamp) as latest_time
                FROM weather_data 
                GROUP BY city
                ORDER BY latest_time DESC
            """, conn)
            
            print(f"\n🏙️ Summary by city:")
            print(summary.to_string(index=False))
            
            # Check time range
            time_range = pd.read_sql_query("""
                SELECT 
                    MIN(timestamp) as earliest,
                    MAX(timestamp) as latest,
                    COUNT(DISTINCT city) as unique_cities
                FROM weather_data
            """, conn)
            
            print(f"\n⏰ Time range:")
            print(time_range.to_string(index=False))
            
        else:
            print("❌ No records found in database!")
            
        conn.close()
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")

if __name__ == "__main__":
    check_database()