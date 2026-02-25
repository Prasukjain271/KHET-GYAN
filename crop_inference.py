import joblib
import pandas as pd
import numpy as np

class CropInferenceNode:
    def __init__(self, model_path="best_crop_recommendation_model.pkl"):
        """
        Initializes the Inference Node.
        The .pkl file contains the full Pipeline (Scaler + OneHotEncoder + RandomForest)
        """
        try:
            self.pipeline = joblib.load(model_path)
            print("✅ Crop Inference Pipeline loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def get_structured_recommendations(self, scenarios):
        """
        Takes the 'scenarios' list from AgriWeatherEngine and returns:
        - Top 3 for the specific season (Kharif/Rabi/Summer)
        - Top 2 for 'Whole Year'
        """
        final_output = {
            "seasonal_recommendations": [],
            "whole_year_recommendations": [],
            "metadata": {
                "total_crops_analyzed": len(self.pipeline.classes_),
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
            }
        }

        for scenario in scenarios:
            # Create DataFrame for the pipeline
            input_df = pd.DataFrame([scenario])
            
            # Get probabilities
            probs = self.pipeline.predict_proba(input_df)[0]
            
            # Identify logic based on Crop_Type
            crop_type = scenario['Crop_Type'].lower().strip()
            is_whole_year = (crop_type == 'whole year')
            
            # Set K limit based on your UX requirement (2 for Whole Year, 3 for Season)
            k_limit = 2 if is_whole_year else 3
            
            # Get indices of top K probabilities
            top_k_indices = np.argsort(probs)[-k_limit:][::-1]
            
            formatted_crops = []
            for idx in top_k_indices:
                formatted_crops.append({
                    "crop": self.pipeline.classes_[idx],
                    "confidence": f"{probs[idx] * 100:.2f}%",
                    "reasoning": self._generate_brief_reasoning(scenario, self.pipeline.classes_[idx])
                })
            
            if is_whole_year:
                final_output["whole_year_recommendations"] = formatted_crops
            else:
                final_output["seasonal_recommendations"] = formatted_crops
                final_output["active_season"] = crop_type

        return final_output

    def _generate_brief_reasoning(self, scenario, crop_name):
        """Simple rule-based reasoning for the LLM/UI to use"""
        # This is the 'Geek Factor' logic for the UI
        reason = f"Suitable for {scenario['State_Name']} in {scenario['Crop_Type']}."
        if scenario['rainfall'] > 1000:
            reason += " Tolerates high moisture."
        if scenario['temperature'] > 30:
            reason += " Heat resistant variety."
        return reason

# ==========================================
# Integration Test for Hackathon Demo
# ==========================================
if __name__ == "__main__":
    # Simulate the output from your AgriWeatherEngine
    mock_scenarios = [
        {
            "N": 90, "P": 40, "K": 40, "pH": 6.5, 
            "temperature": 27.23, "rainfall": 482.2, 
            "State_Name": "punjab", "Crop_Type": "rabi"
        },
        {
            "N": 90, "P": 40, "K": 40, "pH": 6.5, 
            "temperature": 27.23, "rainfall": 535.78, 
            "State_Name": "punjab", "Crop_Type": "whole year"
        }
    ]

    inference = CropInferenceNode()
    results = inference.get_structured_recommendations(mock_scenarios)

    print("\n" + "="*50)
    print(f"🌾 CROP ADVISORY FOR {mock_scenarios[0]['State_Name'].upper()}")
    print("="*50)
    
    print(f"\n[CURRENT SEASON: {results['active_season'].upper()} - TOP 3]")
    for rec in results['seasonal_recommendations']:
        print(f"-> {rec['crop'].title()}: {rec['confidence']} | {rec['reasoning']}")

    print(f"\n[LONG-TERM INVESTMENT: WHOLE YEAR - TOP 2]")
    for rec in results['whole_year_recommendations']:
        print(f"-> {rec['crop'].title()}: {rec['confidence']} | {rec['reasoning']}")