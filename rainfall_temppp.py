import requests
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class AgriWeatherEngine:
    def __init__(self, lookup_csv='E:\coding\ML_hackthon_agri\historical_agri_lookup.csv'):
        self.api_key = os.getenv("OPENWEATHER_API_KEY")
        self.base_url = "http://api.openweathermap.org/data/2.5"
        
        # Load the unified lookup (Rainfall + Temperature)
        self.lookup = pd.read_csv(lookup_csv)
        if self.lookup.empty:
                print(f"⚠️ WARNING: {lookup_csv} was found but it is EMPTY.")
        else:
            print(f"✅ Successfully loaded {len(self.lookup)} weather records.")
        self.lookup['State_Name'] = self.lookup['State_Name'].str.lower().str.strip()
        self.lookup['Crop_Type'] = self.lookup['Crop_Type'].str.lower().str.strip()

    def _get_planting_seasons(self):
        month = datetime.now().month
        if 6 <= month <= 10: return ['kharif', 'whole year']
        if 11 <= month <= 2 or month == 12: return ['rabi', 'whole year']
        if 3 <= month <= 5: return ['summer', 'whole year']
        return ['whole year']

    def get_weather_scenarios(self, city, state_name, soil_data):
        curr_url = f"{self.base_url}/weather?q={city}&appid={self.api_key}&units=metric"
        fore_url = f"{self.base_url}/forecast?q={city}&appid={self.api_key}&units=metric"

        try:
            curr_resp = requests.get(curr_url).json()
            fore_resp = requests.get(fore_url).json()

            if curr_resp.get("cod") != 200: return {"error": "City not found"}
            
            live_temp = curr_resp["main"]["temp"]
            live_7day_rain = sum(item.get("rain", {}).get("3h", 0) for item in fore_resp["list"])
            
            state_clean = state_name.lower().strip()
            seasons = self._get_planting_seasons()
            scenarios = []

            for season in seasons:
                # 1. Fetch Baselines from the Unified Lookup
                row = self.lookup[(self.lookup['State_Name'] == state_clean) & 
                                  (self.lookup['Crop_Type'] == season)]
                
                if row.empty:
                    baseline_rain = 1100.0 if season == 'whole year' else 600.0
                    baseline_temp = 25.0 # National average fallback
                else:
                    baseline_rain = row['rainfall'].values[0]
                    baseline_temp = row['temperature'].values[0]

                # 2. Rainfall Synthesis Logic
                expected_weekly = baseline_rain / 16
                if live_7day_rain > (expected_weekly * 1.5):
                    synth_rain = baseline_rain * 1.10 
                elif live_7day_rain < (expected_weekly * 0.5) and live_7day_rain < 5:
                    synth_rain = baseline_rain * 0.90
                else:
                    synth_rain = baseline_rain

                # 3. Build Feature Vector
                # CRITICAL: We use 'baseline_temp' for the model, but 'live_temp' for alerts
                features = {
                    "N": soil_data["N"], "P": soil_data["P"], "K": soil_data["K"],
                    "pH": soil_data["pH"], 
                    "temperature": round(baseline_temp, 2), # Model-safe data
                    "rainfall": round(synth_rain, 2),
                    "State_Name": state_clean,
                    "Crop_Type": season
                }
                scenarios.append(features)

            return {
                "scenarios": scenarios,
                "display_weather": {
                    "live_temp": live_temp, 
                    "humidity": curr_resp["main"]["humidity"],
                    "state_avg_temp": baseline_temp # Helpful for LLM context
                },
                "alerts": self._generate_alerts(live_temp, fore_resp)
            }

        except Exception as e:
            return {"error": str(e)}

    def _generate_alerts(self, temp, fore_resp):
        alerts = []
        if any(item.get("rain", {}).get("3h", 0) > 10 for item in fore_resp["list"][:4]):
            alerts.append("HEAVY_RAIN_WARNING: Postpone fertilization to prevent leaching.")
        if temp > 35:
            alerts.append("HEAT_STRESS_ALERT: Use mulching to preserve soil moisture.")
        elif temp < 10:
            alerts.append("COLD_STRESS_WARNING: Protect sensitive seedlings from frost.")
        return alerts