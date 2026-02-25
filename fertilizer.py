import pandas as pd

# 1. CENTRAL CONFIGURATION LAYER
# Strategic Reason: Deterministic pricing ensures sub-second response times 
# and 100% reliability in low-connectivity rural environments.
FERTILIZER_MARKET_CONFIG = {
    "UREA": {
        "price_per_50kg": 266.50,
        "source": "Dept. of Fertilizers, Rabi 2024-25",
        "last_verified": "2025-02-28"
    },
    "DAP": {
        "price_per_50kg": 1350.00,
        "source": "NBS Circular 2025",
        "last_verified": "2025-02-28"
    },
    "MOP": {
        "price_per_50kg": 1700.00,
        "source": "Market Benchmark 2025",
        "last_verified": "2025-02-28"
    }
}

class FertilizerNode:
    def __init__(self, data_path='E:\coding\ML_hackthon_agri\Crop_production_final.csv'):
        # Establish the "Source of Truth" for Nutrient Requirements
        df = pd.read_csv(data_path)
        self.crop_requirements = df.groupby('Crop')[['N', 'P', 'K']].median().to_dict('index')
        
        # Pull prices from the static config
        self.prices = {
            "Urea": FERTILIZER_MARKET_CONFIG["UREA"]["price_per_50kg"],
            "DAP": FERTILIZER_MARKET_CONFIG["DAP"]["price_per_50kg"],
            "MOP": FERTILIZER_MARKET_CONFIG["MOP"]["price_per_50kg"]
        }

    def get_fertilizer_advice(self, recommendations, user_soil, weather_alerts, land_area=1.0):
        """
        Calculates bags, costs, and safety status using Seasonal Baseline Normalization.
        """
        full_report = []
        is_heavy_rain = any("HEAVY_RAIN" in alert for alert in weather_alerts)
        
        for rec in recommendations:
            crop_name = rec['crop'].lower().strip()
            req = self.crop_requirements.get(crop_name)
            
            if not req: continue

            # Calculate Nutrient Gap (kg/ha)
           # Instead of: def_n = max(0, req['N'] - user_soil['N'])
# Use a 20% Maintenance Floor:
            def_n = max(req['N'] * 0.20, req['N'] - user_soil['N'])
            def_p = max(req['P'] * 0.20, req['P'] - user_soil['P'])
            def_k = max(req['K'] * 0.20, req['K'] - user_soil['K'])

            # Stoichiometric Conversion: Accounts for DAP providing both P and N
            # [Image of chemical composition of DAP and Urea fertilizers]
            dap_bags = (def_p / 0.46) / 50
            n_from_dap = (dap_bags * 50) * 0.18
            urea_bags = (max(0, def_n - n_from_dap) / 0.46) / 50
            mop_bags = (def_k / 0.60) / 50

            # Scale by User's Land Area
            total_urea = round(urea_bags * land_area, 1)
            total_dap = round(dap_bags * land_area, 1)
            total_mop = round(mop_bags * land_area, 1)

            # Financial Calculation
            total_cost = (total_urea * self.prices["Urea"]) + \
                         (total_dap * self.prices["DAP"]) + \
                         (total_mop * self.prices["MOP"])

            # Logic for Status Determination
            needs_fert = (total_urea + total_dap + total_mop) > 0

            if not needs_fert:
                status, msg = "OPTIMAL", "✅ Soil nutrients are optimal. No additional fertilizer required."
            elif is_heavy_rain:
                status, msg = "HOLD", "⚠️ HEAVY RAIN ALERT: Risk of Nitrogen leaching. Postpone application for 48h."
            else:
                status, msg = "PROCEED", "👍 Weather is clear. Proceed with application for max yield potential."

            full_report.append({
                "crop": rec['crop'],
                "nutrient_gap": {"N": round(def_n, 1), "P": round(def_p, 1), "K": round(def_k, 1)},
                "bags": {"Urea": total_urea, "DAP": total_dap, "MOP": total_mop},
                "financials": {
                    "estimated_investment_inr": round(total_cost, 2),
                    "source": FERTILIZER_MARKET_CONFIG["UREA"]["source"]
                },
                "guidance": {"status": status, "message": msg}
            })

        return full_report