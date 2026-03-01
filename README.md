# 🌾 Khet-Gyan: The AI Advisory for Farmers

**Built for the AI for Sustainability Hackathon (Organized by Canadian University Dubai)**

Khet-Gyan is an end-to-end AI farm assistant designed to bridge the "Data Black Hole" in traditional farming. By shifting away from intuition-based decisions, Khet-Gyan empowers smallholder and marginal farmers with data-driven crop recommendations, yield forecasting, live market integrations, and an autonomous subsidy discovery system.

---

## 🚀 The Problem & Solution

* **The Crisis:** Traditional farming is failing against climate volatility and soil (NPK) depletion.
* **The Solution:** An integrated AI Agronomist that handles everything from optimal crop selection to live sales projections, supported by a 24/7 LangGraph-powered conversational knowledge base.

---

## ✨ Key Features & Architecture

### 1. Agentic Agri-Advisor (RAG Pipeline)
A 24/7 conversational interface that simplifies complex agricultural protocols.
* **Architecture:** LangGraph workflow routing through Brain (Analyzer), Expansion (Multi-Query), Search (Vector Retrieval), and Synthesis (Mentor) nodes.
* **Technology:** Uses Pinecone for metadata filtering, MiniLM embeddings for semantic search, and a Citation Engine that maps responses to verified sources like Vikaspedia.

### 2. Intelligent Crop & Yield Prediction
* **Crop Recommendation ML Node:** A high-speed Random Forest ensemble (300 decision trees) processing an 8-D feature vector (NPK, pH, Climate, Location) to recommend top seasonal and annual crops.
* **Yield Specialist:** Uses a 3-tier routing system to prevent averaging errors, treating different biological crop groups (High-Tonnage vs. Low-Density) independently.

### 3. Productivity & Resource Calculators
* **Fertilizer Intelligence:** Conducts nutrient gap analysis, accounting for DAP's dual N+P content to prevent overspending.
* **Environmental Synthesis Layer:** Blends live forecasts with historical baselines. Includes **Weather-Guard**, which actively halts fertilization advice during heavy rain, heat, or frost.

### 4. Business Intelligence & Policy Research
* **Financial Foresight:** Calculates profitability by weighing predicted yield against fertilizer investment, benchmarked against live Mandi prices and MSP (Minimum Support Price).
* **Autonomous Subsidy Discovery:** Leverages the Tavily Search API to bypass static AI data, actively scanning authorized `.gov.in` domains to surface 2026 government incentives, equipment loans, and seed subsidies.

---

## 🗂️ Project Structure

* `app.py`: Main UI application entry point
* `crop_inference.py`: ML inference script for crop recommendation
* `fertilizer.py`: Nutrient gap and fertilizer logic
* `price_cal.py`: Mandi API and financial forecasting
* `preprocess_rag.py`: Vector embedding and knowledge base setup
* `requirements.txt`: Python dependencies
* `/models/`: Contains serialized `.pkl` models (e.g., Random Forest)
* `/data/`: Historical datasets and RAG text documents (`.txt`)
* `/notebooks/`: Jupyter notebooks for EDA and model training

---

## 🛠️ Setup & Installation

**1. Clone the repository**
```bash
git clone [git clone https://github.com/KHET-GYAN/KHET-GYAN.git](git clone https://github.com/KHET-GYAN/KHET-GYAN.git)
cd Khet-Gyan
