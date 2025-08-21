"""
Comprehensive test for weather data collection
"""
import os
import sys
from dotenv import load_dotenv
import sqlite3

# Add src to path
sys.path.append('src')

from data_collection.weather_collector import WeatherCollector, WeatherConfig

def test_api_connection():
    """Test basic API connectivity"""
    print("🔍 Testing API Connection...")
    
    load_dotenv()
    api_key = os.getenv("OPENWEATHER_API_KEY")
    
    if not api_key:
        print("❌ No API key found in .env file!")
        return False
        
    # Fix: Provide cities parameter explicitly
    config = WeatherConfig(
        api_key=api_key,
        cities=["New York"]  # Add this line
    )
    collector = WeatherCollector(config)
    
    # Test single city
    data = collector.fetch_weather_data("New York")
    if data:
        print("✅ API connection successful!")
        print(f"   Temperature in New York: {data['main']['temp']}°C")
        return True
    else:
        print("❌ API connection failed!")
        return False

def test_database_setup():
    """Test database creation and schema"""
    print("\n🔍 Testing Database Setup...")
    
    # Fix: Provide both required parameters
    config = WeatherConfig(
        api_key="dummy_key",  # Add this line
        cities=["New York"],  # Add this line
        db_path="data/test_weather.db"
    )
    collector = WeatherCollector(config)
    
    # Check if database file exists
    if os.path.exists("data/test_weather.db"):
        print("✅ Database file created!")
        
        # Check table structure
        conn = sqlite3.connect("data/test_weather.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if ('weather_data',) in tables:
            print("✅ Weather data table created!")
            
            # Check columns
            cursor.execute("PRAGMA table_info(weather_data);")
            columns = [col[1] for col in cursor.fetchall()]
            
            required_cols = ['city', 'temperature', 'humidity', 'pressure']
            if all(col in columns for col in required_cols):
                print("✅ Database schema is correct!")
                conn.close()
                return True
            else:
                print("❌ Missing required columns!")
                conn.close()
                return False
        else:
            print("❌ Weather data table not found!")
            conn.close()
            return False
    else:
        print("❌ Database file not created!")
        return False

def test_data_collection():
    """Test full data collection process"""
    print("\n🔍 Testing Data Collection...")
    
    load_dotenv()
    config = WeatherConfig(
        api_key=os.getenv("OPENWEATHER_API_KEY"),
        cities=["New York", "Los Angeles", "Chicago"],
        db_path="data/test_weather.db"
    )
    
    collector = WeatherCollector(config)
    
    # Collect data for test cities
    print("🔄 Collecting data for test cities...")
    collector.collect_all_cities()
    
    # Debug: Check database directly
    import sqlite3
    import pandas as pd
    
    conn = sqlite3.connect("data/test_weather.db")
    df = pd.read_sql_query("SELECT * FROM weather_data ORDER BY timestamp DESC LIMIT 10", conn)
    conn.close()
    
    print(f"📊 Records in database: {len(df)}")
    if len(df) > 0:
        print("✅ Successfully collected and stored data!")
        print("\n📊 Sample data:")
        print(df[['city', 'temperature', 'humidity', 'pressure']].head())
        return True
    else:
        print("❌ No data found in database!")
        return False

def run_all_tests():
    """Run complete test suite"""
    print("🌤️  Starting Weather Data Collection Tests\n")
    
    tests = [
        ("API Connection", test_api_connection),
        ("Database Setup", test_database_setup),
        ("Data Collection", test_data_collection)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📋 TEST RESULTS SUMMARY")
    print("="*50)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! Data collection is ready!")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    run_all_tests()