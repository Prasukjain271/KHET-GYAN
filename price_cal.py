import os
import json
import requests
import statistics
from datetime import datetime

class EconomicsAdvisor:
    def __init__(self, cache_file="price_cache.json"):
        self.cache_file = cache_file

        # VERIFIED OFFICIAL MSP 2025-26 (Rs. per Quintal)
        self.msp_data = {
            "Cotton": 7710, "Wheat": 2425, "Paddy(Common)": 2369, 
            "Maize": 2400, "Soyabean": 5328, "Mustard": 5950,
            "Jowar(Sorghum)": 3699, "Barley(Jau)": 1980,
            "Ragi (Finger Millet)": 4886, "Green Gram(Moong)(Whole)": 8768,
            "Sunflower": 7721, "Jute": 5650
        }

        self.crop_mapping = {
            "cotton": "Cotton", "horsegram": "Kulthi(Horse Gram)", "jowar": "Jowar(Sorghum)",
            "maize": "Maize", "moong": "Green Gram(Moong)(Whole)", "rice": "Paddy(Common)",
            "wheat": "Wheat", "sesamum": "Sesamum(Sesame,Gingelly,Til)", "soyabean": "Soyabean",
            "jute": "Jute", "rapeseed": "Mustard", "arecanut": "Arecanut(Betelnut/Supari)",
            "onion": "Onion", "potato": "Potato", "sweetpotato": "Sweet Potato",
            "tapioca": "Tapioca", "turmeric": "Turmeric", "barley": "Barley(Jau)",
            "banana": "Banana", "coriander": "Corriander seed", "garlic": "Garlic",
            "blackpepper": "Black pepper", "ragi": "Ragi (Finger Millet)",
            "sunflower": "Sunflower", "cashewnuts": "Cashewnuts"
        }
        self.price_cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except: return {}
        return {}

    def _save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.price_cache, f)

    def _get_api_price(self, api_crop, state, district=None):
        api_key = os.getenv("GOV_DATA_API_KEY")
        if not api_key:
            return None, [], "API Key Missing"

        resource_id = "9ef84268-d588-465a-a308-a864a43d0070"
        today_str = datetime.now().strftime('%Y-%m-%d')
        
        state_norm = state.strip().title()
        dist_norm = district.strip().title() if district else None
        cache_key = f"{api_crop}_{state_norm}_{dist_norm}".lower().replace(" ", "_")

        if cache_key in self.price_cache:
            entry = self.price_cache[cache_key]
            if entry.get("date") == today_str:
                return entry["price"], entry.get("history", []), "Live (Cached)"

        url = (f"https://api.data.gov.in/resource/{resource_id}?api-key={api_key}"
               f"&format=json&limit=50&filters[state]={state_norm}"
               f"&filters[commodity]={api_crop}")
        if dist_norm: url += f"&filters[district]={dist_norm}"

        try:
            response = requests.get(url, timeout=7)
            if response.status_code == 200:
                records = response.json().get('records', [])
                prices = []
                for r in records:
                    try:
                        val = r.get('Modal_x0020_Price')
                        if val is not None:
                            price = float(val)
                            if price > 0: prices.append(price)
                    except (ValueError, TypeError): continue
                
                if prices:
                    avg_price = sum(prices) / len(prices)
                    history = self.price_cache.get(cache_key, {}).get("history", [])
                    if not history or history[-1] != avg_price:
                        history.append(avg_price)
                    history = history[-10:]

                    self.price_cache[cache_key] = {"price": avg_price, "date": today_str, "history": history}
                    self._save_cache()
                    return avg_price, history, "Live (Fresh)"
        except: pass

        if cache_key in self.price_cache:
            old_entry = self.price_cache[cache_key]
            return old_entry["price"], old_entry.get("history", []), f"Offline (Last updated: {old_entry['date']})"

        return None, [], None

    def calculate_economics(self, crop, state, district, yield_tonnes, fert_cost, land_size):
        # 1. Standardize and Map
        clean_crop = str(crop).lower().strip()
        api_crop = self.crop_mapping.get(clean_crop)
        if not api_crop:
            api_crop = clean_crop.title()

        # 2. Waterfall API Call
        price, history, source = self._get_api_price(api_crop, state, district)
        if price is None:
            price, history, source = self._get_api_price(api_crop, state)
            if source:
                source = f"Regional Avg ({source})"

        # 3. SAFETY FALLBACK (For the Hackathon Demo)
        if price is None:
            price = self.msp_data.get(api_crop, 2500)
            source = "Estimated (Market Data Unavailable)"

        # 4. Trend Analysis
        trend = "Stable ➖"
        volatility = "Stable"
        if len(history) >= 3:
            avg_history = sum(history[:-1]) / (len(history) - 1)
            if price > avg_history * 1.02: trend = "Rising 📈"
            elif price < avg_history * 0.98: trend = "Falling 📉"
            
            std_dev = statistics.stdev(history)
            rel_std_dev = (std_dev / price) * 100
            if rel_std_dev > 10: volatility = "High Risk ⚠️"
            elif rel_std_dev > 5: volatility = "Moderate"

        # 5. MSP Advice
        msp_price = self.msp_data.get(api_crop)
        recommendation = "No official MSP (Horticultural Crop)"
        if msp_price:
            diff = price - msp_price
            if diff > 0:
                recommendation = f"Sell in Mandi (Current Profit: +₹{round(diff, 2)} over MSP)"
            else:
                recommendation = f"Consider MSP procurement (Market is ₹{round(abs(diff), 2)} below MSP)"

        # 6. Final Financial Calculations (Indentation Fixed)
        total_yield_qtl = float(yield_tonnes) * 10 * float(land_size) 
        total_sales = total_yield_qtl * price
        
        # Operational costs set to 0 as per request
       
        
        # Total investment now only tracks Fertilizer for ROI calculation
        total_fertilizer_investment = float(fert_cost) 
        profit = total_sales - total_fertilizer_investment

        return {
            "crop": api_crop,
            "data_source": source,
            "market_price_qtl": round(price, 2),
            "price_trend": trend,
            "volatility": volatility,
            "total_yield_qtl":total_yield_qtl,
            "strategic_advice": recommendation,
            "profit": round(profit, 2),
            'total_fertilizer_investment':total_fertilizer_investment,
            'total_sales':total_sales,
            "financial_note": (
                "Note: This profit calculation factors in fertilizer costs only. "
                "Operational expenses (labor, seeds, irrigation, and machinery) vary by region "
                "and have not been deducted from the final amount."
            )
        }