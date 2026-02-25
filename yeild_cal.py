import joblib
import pandas as pd

class YieldCalculator:
    def __init__(self, model_path="production_models/"):
        """
        Initializes the 3 specialist models from the saved .pkl files.
        
        """
        try:
            # Loading the 3-Tier Specialists
            self.high_tonnage_model = joblib.load(f"{model_path}high_tonnage_model.pkl")
            self.precision_model = joblib.load(f"{model_path}precision_model.pkl")
            self.low_density_model = joblib.load(f"{model_path}low_density_model.pkl")
            print("✅ All 3 models loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading models: {e}")

        # Define the routing maps based on our statistical tiers
        self.high_tonnage_crops = ['banana', 'sugarcane', 'tapioca', 'potato', 'onion', 'sweetpotato']
        self.low_density_crops = ['moong', 'sesamum', 'rapeseed', 'soyabean', 'horsegram', 
                                  'sunflower', 'coriander', 'cashewnuts', 'blackpepper']
        # Tier 3 (Precision) now includes our cleaned Maize and Cotton
        self.precision_crops = ['wheat', 'rice', 'barley', 'jowar', 'ragi', 'arecanut', 
                                'garlic', 'turmeric', 'maize', 'cotton']

    def calculate_yield(self, input_data: dict):
        """
        Routes the input to the correct specialist and returns the prediction.
        """
        # 1. Convert the dictionary to a DataFrame for the preprocessor
        input_df = pd.DataFrame([input_data])
        # 🔍 DEBUG: Check for NaNs and Print them
        print("\n--- Yield Input Debug ---")
        print(input_df.isna().sum()) # Shows count of NaNs per column
        if input_df.isnull().values.any():
            print("🚨 NULL FOUND IN:")
            print(input_df.columns[input_df.isnull().any()].tolist())
            print("Raw Data Row:", input_data)
            # 2. Extract crop for routing (standardize to lowercase)
        crop = input_data.get('Crop', '').lower()
        
        # 3. Routing Logic (The If-Else Chain)
        if crop in self.high_tonnage_crops:
            prediction = self.high_tonnage_model.predict(input_df)
            tier = "High-Tonnage Specialist"
        elif crop in self.low_density_crops:
            prediction = self.low_density_model.predict(input_df)
            tier = "Low-Density Specialist"
        elif crop in self.precision_crops:
            prediction = self.precision_model.predict(input_df)
            tier = "Precision Specialist"
        else:
            # Default fallback if a rare crop appears
            prediction = self.precision_model.predict(input_df)
            tier = "Default (Precision) Model"

        # 4. Final Output Formatting
        result = round(float(prediction[0]), 4)
        print(f"--- Prediction Complete ---")
        print(f"Crop: {crop.capitalize()} | Specialist Used: {tier}")
        print(f"Predicted Yield: {result} t/ha")
        
        return result

