import streamlit as st
import os
from faq import research_app  # The Pinecone/RAG workflow
from langgg import workflow      # The ML/Economics workflow

# --- UI CONFIGURATION (White Background, Black Text) ---
# --- ENHANCED UI CONFIGURATION ---
st.set_page_config(page_title="Khet-Gyan AI", layout="wide")

st.markdown("""
    <style>
    /* Main background and global text color */
    .stApp {
        background-color: #FFFFFF;
        color: #1A1A1A;
    }
    
    /* Ensure all headers and labels are crisp black */
    h1, h2, h3, p, span, label {
        color: #1A1A1A !important;
        font-weight: 500 !important;
    }

    /* Style the tabs for better visibility */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: #F8F9FA;
        padding: 10px;
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px;
        color: #1A1A1A;
    }

    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        border-bottom: 3px solid #2E7D32 !important; /* Green highlight */
    }

    /* Input fields: Light grey background with dark text */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: #F1F3F4 !important;
        color: #1A1A1A !important;
        border: 1px solid #DADCE0 !important;
    }

    /* The 'Search' and 'Generate' Buttons */
    div.stButton > button {
        background-color: #2E7D32 !important; /* Professional Green */
        color: #FFFFFF !important;
        width: 100%;
        border-radius: 8px;
        border: none;
        height: 3em;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    div.stButton > button:hover {
        background-color: #1B5E20 !important; /* Darker green on hover */
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    /* Result boxes for visibility */
    .stAlert {
        background-color: #F8F9FA;
        border: 1px solid #2E7D32;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.title("🚜 Khet-Gyan AI: Smart Farming Assistant")
    
    tab1, tab2 = st.tabs(["📋 Farm Planning", "🔍 Agricultural Research"])

    # --- TAB 1: ML PREDICTIONS ---
    with tab1:
        st.header("Predictive Farm Planning")
        col1, col2 = st.columns(2)
        
        with col1:
            city = st.text_input("Enter City", "Patiala")
            state_name = st.text_input("Enter State", "Punjab")
            area = st.number_input("Farm Area (Acres)", min_value=0.1, value=1.0)
            
        with col2:
            st.write("Soil Data (NPK & pH)")
            n = st.number_input("Nitrogen (N)", value=90)
            p = st.number_input("Phosphorus (P)", value=42)
            k = st.number_input("Potassium (K)", value=43)
            ph = st.number_input("Soil pH", value=6.5)

        if st.button("Generate Farming Plan"):
            with st.spinner("Analyzing soil, weather, and market data..."):
                inputs = {
                    "user_input": {
                        "city": city,
                        "state_name": state_name,
                        "area": area,
                        "soil_data": {"N": n, "P": p, "K": k, "pH": ph}
                    }
                }
                # Running the lng.py workflow
                final_state = workflow.invoke(inputs)
                st.markdown("### 📝 Your Strategic Plan")
                st.write(final_state["final_llm_answer"])

    # --- TAB 2: RAG RESEARCH ---
    with tab2:
        st.header("Expert Knowledge Base")
        st.info("Ask about pest control, weed management, or specific crop diseases.")
        
        user_query = st.text_input("What is your question? (e.g., 'Wheat pest control')")
        
        if st.button("Search Database"):
            if user_query:
                with st.spinner("Consulting agricultural records..."):
                    # Running the faq.py workflow
                    research_state = research_app.invoke({"question": user_query})
                    st.markdown("### 🔬 Research Report")
                    st.write(research_state["answer"])
            else:
                st.warning("Please enter a question first.")

if __name__ == "__main__":
    main()