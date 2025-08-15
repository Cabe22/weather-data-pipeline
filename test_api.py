"""
Quick test to verify API connection works
"""
import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("OPENWEATHER_API_KEY")

print("🌤️  Testing Weather API Connection...")
print(f"API Key loaded: {'✅ Yes' if api_key else '❌ No'}")

if not api_key:
    print("❌ ERROR: No API key found. Check your .env file!")
    exit()

# Test API call
try:
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': 'New York',
        'appid': api_key,
        'units': 'metric'
    }
    
    print("🔄 Making API request...")
    response = requests.get(url, params=params, timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        temp = data['main']['temp']
        city = data['name']
        country = data['sys']['country']
        
        print("✅ API Connection Successful!")
        print(f"📍 Location: {city}, {country}")
        print(f"🌡️  Temperature: {temp}°C")
        print(f"☁️  Weather: {data['weather'][0]['description']}")
        
    else:
        print(f"❌ API Error: Status code {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"❌ Connection Error: {e}")
    print("Check your internet connection and API key!")