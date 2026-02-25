from typing import TypedDict, Any, List, Literal, Annotated
from abc import ABC, abstractmethod
import os
import re
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing import Literal, List, Optional
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
gemini_api_key = os.getenv("GEMINI_API_KEY") 
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite',google_api_key=gemini_api_key)
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
parser=StrOutputParser()

from rainfall_temp import AgriWeatherEngine
# 1. Create an instance of the class
from crop_inference import CropInferenceNode
from fertilizer import FertilizerNode
from price_cal import EconomicsAdvisor 
from yeild_cal import YieldCalculator
class state(TypedDict):
    user_input:dict
    scenarios:dict
    weather_alerts:dict
    crop_recommendations:dict
    fertilizer_report:dict
    yeild_report:dict
    final_report:dict
    government_schemes: list
    final_llm_answer:dict

def make_proper_input(state: state) -> dict[str, Any]:
    weather_engine = AgriWeatherEngine()
    city, state_name, soil_data = [state["user_input"].get(k) for k in ("city", "state_name", "soil_data")]
    result=weather_engine.get_weather_scenarios(city, state_name, soil_data)
    if "error" in result:
        print(f"❌ Weather Engine Error: {result['error']}")    
    scenarios = result.get("scenarios")
    if scenarios is None:
        # Fallback logic so the graph doesn't die
        print("⚠️ Warning: No scenarios found in weather result. ")
    # CHECK FOR ERROR FIRST

    return {
             "scenarios": result["scenarios"],
            "weather_alerts": result["alerts"]
            }
def suitable_crops(state: state) -> dict[str, Any]:
    predictor=CropInferenceNode()
    result=predictor.get_structured_recommendations(state['scenarios'])
    return{'crop_recommendations':result}
def fertilizer(state: state) -> dict[str, Any]:
    fertilizer=FertilizerNode()
    soil_data = state["user_input"].get("soil_data")
    area = state["user_input"].get("area")
    # Assuming 'data' is your JSON dictionary
    data = state['crop_recommendations'] # Your JSON input

    # 1. Combine both lists and add the 'rec_type' tag
    merged_reccs = []

    # Process Seasonal
    for item in data.get("seasonal_recommendations", []):
        new_item = item.copy()
        new_item["rec_type"] = "seasonal"
        merged_reccs.append(new_item)

    # Process Whole Year
    for item in data.get("whole_year_recommendations", []):
        new_item = item.copy()
        new_item["rec_type"] = "whole_year"
        merged_reccs.append(new_item)

    # 2. Final List of 5 Dictionaries
    # merged_reccs now contains all 5 crops with their confidence, reasoning, and type
    print(merged_reccs[:5])
    fertilizer_report=fertilizer.get_fertilizer_advice(merged_reccs,soil_data,state['weather_alerts'],area)    #def get_fertilizer_advice(self, recommendations, user_soil, weather_alerts, land_area=1.0):
    return {'fertilizer_report':fertilizer_report}
def yeild_cal(state: state) -> dict[str, Any]:
    cal = YieldCalculator()
    
    # 1. ALWAYS extract your data first
    scenarios_list = state.get("scenarios", [])
    if not scenarios_list:
        return {'yeild_report': {"seasonal": [], "annual": []}}
    
    scenario = scenarios_list[0] 
    recommendations = state.get("crop_recommendations", {})
    
    # --- DEFINE THESE BEFORE THE HELPER FUNCTION CALLS ---
    seasonal_list = recommendations.get("seasonal_recommendations", [])
    annual_list = recommendations.get("whole_year_recommendations", [])
    
    # 2. Define the helper function
    def process_crop_list(crop_objects):
        reports = []
        for item in crop_objects:
            model_input = {
                'State_Name': scenario.get('State_Name'),
                'Crop_Type': scenario.get('Crop_Type'),
                'Crop': item.get("crop").lower(),
                'N': scenario.get('N'),
                'P': scenario.get('P'),
                'K': scenario.get('K'),
                'pH': scenario.get('pH'),
                'rainfall': scenario.get('rainfall'),
                'temperature': scenario.get('temperature')
            }
            predicted_tonnage = cal.calculate_yield(model_input)
            reports.append({**item, "predicted_yield": predicted_tonnage})
        return reports

    # 3. NOW it is safe to call the function with the defined lists
    final_reports = {
        "seasonal": process_crop_list(seasonal_list),
        "annual": process_crop_list(annual_list)
    }
    
    return {'yeild_report': final_reports}
def economics_node(state: dict):
    # 1. Setup the Advisor and get User Inputs
    advisor = EconomicsAdvisor() # Or however you initialized it
    user_input = state.get("user_input", {})
    
    # Extract static values
    state_name = user_input.get("state_name")
    district = user_input.get("city")
    land_size = user_input.get("area", 1.0)
    
    # 2. Get the reports generated by previous nodes
    # 'final_report' from yeild_cal and 'fertilizer_report' from fertilizer node
    yield_data = state.get("yeild_report", {})
    fert_report = state.get("fertilizer_report", []) 
    
    # Create a quick lookup for fertilizer costs by crop name
    fert_costs = {item['crop']: item['financials']['estimated_investment_inr'] 
                  for item in fert_report if 'financials' in item}

    # 3. Process the logic
    final_economics = {"seasonal": [], "annual": []}
    
    def process_tier(crop_list):
        processed = []
        for item in crop_list:
            crop_name = item['crop']
            # Get the specific cost for this crop, default to 0 if missing
            current_fert_cost = fert_costs.get(crop_name, 0)
            
            # --- THE FIX: Pass all 6 required arguments ---
            econ_result = advisor.calculate_economics(
                crop=crop_name,
                state=state_name,
                district=district,
                yield_tonnes=item['predicted_yield'],
                fert_cost=current_fert_cost,
                land_size=land_size
            )
            
            # Merge everything into one final object
            processed.append({**item, "economics": econ_result})
        return processed

    # 4. Map back to state
    final_economics["seasonal"] = process_tier(yield_data.get("seasonal", []))
    final_economics["annual"] = process_tier(yield_data.get("annual", []))

    return {"final_report": final_economics}
    
    #def calculate_economics(self, crop, state, district, yield_tonnes, fert_cost, land_size):

def dynamic_scheme_research(state: state) -> dict:
    # Initialize the updated tool
    search = TavilySearchResults(k=3)
    
    crop = state['final_report']['seasonal'][0]['crop']
    state_name = state['user_input']['state_name']
    
    # 🟢 Refined Query: Uses exact matching and excludes news sites for higher authority
    query = f'site:gov.in "{crop}" "{state_name}" subsidy "2026" -site:news.google.com'
    
    try:
        # Fetching raw results (Tavily naturally returns content and URL)
        results = search.run(query)
        return {"government_schemes": results}
    except Exception as e:
        print(f"Research Node Error: {e}")
        return {"government_schemes": []}
    
# 1. Define the LLM Response Node
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Any

def generate_response(state: dict) -> dict[str, Any]:
    # 1. Extract EVERYTHING from the state
    user_input = state.get("user_input", {})
    scenarios = state.get("scenarios", [])
    weather_alerts = state.get("weather_alerts", [])
    crop_recs = state.get("crop_recommendations", {})
    fert_report = state.get("fertilizer_report", [])
    final_report = state.get("final_report", {})
    schemes_data = state.get("government_schemes", [])
    
    user_location = f"{user_input.get('city')}, {user_input.get('state_name')}"
    farm_area = user_input.get("area", 1.0)

    # 2. Enhanced System Prompt
    system_prompt = SystemMessage(content=(
    """You are 'Khet-Gyan AI', a specialized Agricultural Business Advisor for Indian farmers. 
    Your goal is to synthesize ML predictions, market data, and scientific research into a strategic farming plan.

    STRICT UNIT & INTEGRITY RULES:
    1.DO NOT MAKE STUFF ON YOUR OWN .ONLY READ THE SPEECH GIVEN    
    2. NO CONFIDENCE SCORES: Do not mention 'confidence scores' or 'model percentages' (e.g., 59.50%). 
       Instead, describe the strength of the recommendation using natural language like 'highly reliable', 
       'strongly recommended', or 'shows great potential'.

    STRICT RESPONSE STRUCTURE:

    1. Polite Greeting & Context: Start with a respectful, professional, and encouraging 'Kisan Bhai' tone. 
       Acknowledge the current season and market conditions.

    2. The 'Best Fit' Report: Recommend the single best crop Based on the ECONOMIC ANALYSIS . Explain WHY (soil health, market demand, 
       or resilience) without using technical ML jargon.
    ** do the 3. and 4. for all the secondary options in the final_report
    3. Alternative Crop Options: PRINT all the secondary options in the data .

    4. Financial & Yield Summary Table: Provide a clear table summarizing all mentioned crops for the 
       TOTAL farm area (Hectare ). Columns:
       - Crop Name
       - Predicted Yield (Total Quintals)
       - Estimated Total Sales (INR)
       - Estimated Fertilizer Cost (INR)

    5. 🎁 Government Benefits & Subsidies: 
       - Use the 'LIVE RESEARCH DATA' provided to identify the most relevant schemes.
       - Explain how the farmer can benefit (e.g., seed subsidy, low-interest loan).
       - MANDATORY: Provide the official source URL for each scheme so the farmer can apply.


    TONE GUIDELINES:
    Always use a professional yet encouraging 'Kisan Bhai' tone. Use simple, clear language 
    while maintaining scientific and financial accuracy."""
        ))
    # 3. Comprehensive Human Content (The "State Dump")
    human_content = f"""
    ### FARM CONTEXT
    - Location: {user_location}
    - Farm Area: {farm_area} Acres
    - Soil Data: {user_input.get('soil_data')}
    
    ### PREDICTIVE DATA
    - Seasonal Recommendations: {crop_recs.get('seasonal_recommendations')}
    - Economic Analysis: {final_report.get('seasonal')}
    - Fertilizer Requirements: {fert_report}
    
    ### EXTERNAL FACTORS
    - Weather Alerts: {weather_alerts}
    - Environmental Scenarios: {scenarios}
    
    Please provide a concise summary, the required table, and a strategic explanation for the farmer.
    """
    
    # 4. Run the LLM
    response = llm.invoke([system_prompt, HumanMessage(content=human_content)])
    
    # Return the updated state
    return {"final_llm_answer": response.content}
graph= StateGraph(state)

graph.add_node("make_proper_input",make_proper_input)
graph.add_node("suitable_crops",suitable_crops)
graph.add_node("fertilizer",fertilizer)
graph.add_node("economics_node",economics_node)
graph.add_node("dynamic_scheme_research", dynamic_scheme_research)
graph.add_node("yeild_cal",yeild_cal)
graph.add_node("generate_response", generate_response)


#graph.add_node("fertilizer",fertilizer)
graph.add_edge(START,"make_proper_input")


graph.add_edge("make_proper_input","suitable_crops")

graph.add_edge("suitable_crops","fertilizer")
graph.add_edge("suitable_crops","yeild_cal")

graph.add_edge("fertilizer","economics_node")
graph.add_edge("yeild_cal","economics_node")
graph.add_edge("economics_node", "dynamic_scheme_research")
graph.add_edge("dynamic_scheme_research", "generate_response")
graph.add_edge("generate_response", END)

workflow=graph.compile()
