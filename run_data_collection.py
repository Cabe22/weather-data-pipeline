"""
Production data collection runner
"""
import os
import sys
from dotenv import load_dotenv
import time

# Add src to path
sys.path.append('src')

from data_collection.weather_collector import WeatherCollector, WeatherConfig

def main():
    """Run production data collection"""
    print("🌤️  Starting Production Weather Data Collection")
    print("="*60)
    
    # Load configuration
    load_dotenv()
    
    config = WeatherConfig(
        api_key=os.getenv("OPENWEATHER_API_KEY"),
        cities=[
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
            "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"
        ],
        update_interval=1800,  # 30 minutes
        db_path="data/weather.db"
    )
    
    print(f"📍 Cities to monitor: {len(config.cities)}")
    print(f"⏱️  Update interval: {config.update_interval/60} minutes")
    print(f"🗄️  Database: {config.db_path}")
    print()
    
    # Initialize collector
    collector = WeatherCollector(config)
    
    # Run initial collection
    print("🔄 Running initial data collection...")
    start_time = time.time()
    
    collector.collect_all_cities()
    
    end_time = time.time()
    collection_time = end_time - start_time
    
    # Check results
    recent_data = collector.get_recent_data(hours=1)
    
    print(f"✅ Collection completed in {collection_time:.2f} seconds")
    print(f"📊 Records collected: {len(recent_data)}")
    
    if len(recent_data) > 0:
        print("\n📈 Latest data summary:")
        print(f"   Average temperature: {recent_data['temperature'].mean():.1f}°C")
        print(f"   Cities covered: {recent_data['city'].nunique()}")
        print(f"   Data freshness: {recent_data['timestamp'].max()}")
        
        print("\n🏙️ City temperatures:")
        city_temps = recent_data.groupby('city')['temperature'].first().sort_values(ascending=False)
        for city, temp in city_temps.head().items():
            print(f"   {city}: {temp:.1f}°C")
    
    print("\n✅ Production data collection setup complete!")
    print("💡 To run continuous collection, use: collector.run_scheduler()")

if __name__ == "__main__":
    main()