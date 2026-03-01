# 🌾 Khet-Gyan: The AI Advisory for Farmers

**Built for the AI for Sustainability Hackathon (Organized by Canadian University Dubai)**

Khet-Gyan is an end-to-end AI farm assistant designed to bridge the "Data Black Hole" in traditional farming. By shifting away from intuition-based decisions, Khet-Gyan empowers smallholder and marginal farmers with data-driven crop recommendations, yield forecasting, live market integrations, and an autonomous subsidy discovery system.

---

## 🚀 The Problem & Solution

* **The Crisis:** Traditional farming is failing against climate volatility and soil (NPK) depletion.
* **The Solution:** An integrated AI Agronomist that handles everything from optimal crop selection to live sales projections, supported by a 24/7 LangGraph-powered conversational knowledge base.

---

## 🛠️ Tech Stack

* **Orchestration:** LangChain, LangGraph (Agentic Workflows)
* **Machine Learning:** Scikit-Learn (Random Forest Ensembles), Pandas, NumPy
* **Vector Database:** Pinecone (Metadata Filtering)
* **LLMs & Embeddings:** Google Gemini, Sentence-Transformers (MiniLM-L6-v2)
* **Real-time APIs:** Tavily Search (Govt. Schemes), Live Mandi API (Pricing)
* **Frontend:** Streamlit

---

## ✨ Key Features & Architecture

### 1. Agentic Agri-Advisor (RAG Pipeline)
A 24/7 conversational interface that simplifies complex agricultural protocols.
* **Architecture:** LangGraph workflow routing through Brain (Analyzer), Expansion (Multi-Query), Search (Vector Retrieval), and Synthesis (Mentor) nodes.
* **Technology:** Uses Pinecone for metadata filtering and a Citation Engine that maps responses to verified sources like Vikaspedia.

### 2. Intelligent Crop & Yield Prediction
* **Crop Recommendation ML Node:** A high-speed Random Forest ensemble (300 decision trees) processing an 8-D feature vector (NPK, pH, Climate, Location) to recommend top seasonal and annual crops.
* **Yield Specialist:** Uses a 3-tier routing system to prevent averaging errors, treating different biological crop groups (High-Tonnage vs. Low-Density) independently.

### 3. Productivity & Resource Calculators
* **Fertilizer Intelligence:** Conducts nutrient gap analysis, accounting for DAP's dual N+P content to prevent overspending.
* **Environmental Synthesis Layer:** Blends live forecasts with historical baselines. Includes **Weather-Guard**, which actively halts fertilization advice during heavy rain, heat, or frost.

### 4. Business Intelligence & Policy Research
* **Financial Foresight:** Calculates profitability by weighing predicted yield against fertilizer investment, benchmarked against live Mandi prices and MSP (Minimum Support Price).
* **Autonomous Subsidy Discovery:** Leverages the Tavily Search API to bypass static AI data, actively scanning authorized `.gov.in` domains to surface 2026 government incentives.

---

## 🗂️ Project Structure

* `app.py`: Main UI application entry point
* `crop_inference.py`: ML inference script for crop recommendation
* `fertilizer.py`: Nutrient gap and fertilizer logic
* `price_cal.py`: Mandi API and financial forecasting
* `preprocess_rag.py`: Vector embedding and knowledge base setup
* `requirements.txt`: Python dependencies
* `/models/`: Contains serialized `.pkl` models
* `/data/`: Historical datasets and RAG text documents

---

## 👥 Team & Contributions

### **Samaira Jain**
* **Agentic Node Logic:** Engineered the core decision-making logic for the **LangGraph nodes** (Brain, Expansion, and Synthesis) and integrated the **Tavily Search API** for real-time 2026 government subsidy discovery.
* **Financial Foresight Engine:** Developed the **Price-Cal module** which triggers **Live Mandi API** fetches and benchmarks against **MSP (Minimum Support Price)** to determine real-time profitability and sales projections.
* **Predictive ML Development:** Built and trained the **Crop Recommendation Engine** using a Random Forest ensemble to process 8-D soil and climate feature vectors for optimal seasonal planting.
* **Sustainability & Climate Logic:** Designed the **Fertilizer Intelligence** module and the **Environmental Synthesis Layer** (Weather-Guard) to actively halt fertilization advice during high-risk climate extremes.

### **Prasuk Jain**
* **Workflow Architecture:** Designed the high-level **LangGraph state machine** and directed the structural flow of information between the agentic layers, memory buffers, and external tool-calling environments.
* **Yield Specialist Node:** Engineered the **3-tier routing system** for yield forecasting, optimizing predictions for distinct biological crop groups to prevent averaging errors in high-tonnage vs. low-density crops.
* **Knowledge Architecture:** Architected the **RAG (Retrieval-Augmented Generation) Pipeline**, integrating Pinecone for metadata filtering and MiniLM-L6-v2 for high-density semantic vector search.
* **System Integration & Sourcing:** Managed the end-to-end data pipeline between the vector database and the LangGraph environment while curating verified scientific datasets from sources like Vikaspedia.

---

## 🌟 Future Scope

* **Satellite Imagery Integration:** Incorporate multispectral satellite data to detect early-stage pest infestations and moisture stress before they are visible to the naked eye.
* **Multilingual Voice Interface:** Expand the "Mentor" node to support local Indian dialects via STT (Speech-to-Text) to assist farmers with lower literacy rates.
* **Edge IoT Integration:** Sync with on-field soil sensors to provide automated irrigation alerts directly through the LangGraph workflow.
* **Carbon Credit Tracking:** Implement a module to help farmers calculate and monetize carbon sequestration based on sustainable tilling practices.

---

## 🛠️ Setup & Installation

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/Khet-Gyan.git](https://github.com/yourusername/Khet-Gyan.git)
cd Khet-Gyan
