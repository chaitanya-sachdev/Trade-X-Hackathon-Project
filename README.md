# Trade-X-Hackathon-Project
An AI-powered tariff optimization engine using a local Llama-3 8B &amp; ChromaDB RAG architecture to automate HS code classification and optimize global supply chains.
# Trade-X | AI Tariff Optimization

Trade-X is a privacy-first intelligence engine that automates HS code classification and supply chain optimization. By running local inference, it allows enterprises to classify proprietary product data without third-party API exposure.

## Project Scope
Current trade compliance is reactive and manual. Trade-X transforms this into a proactive financial advantage by identifying Free Trade Agreement (FTA) savings and recommending lower-tariff sourcing routes in real-time.

## Technical Architecture
The system utilizes a **Tiered Retrieval Architecture** to ensure sub-50ms latency and high accuracy:

* **Tier 1: Semantic Cache** – Instant lookup for repetitive queries using ChromaDB.
* **Tier 2: Vector RAG** – Contextual grounding in official Indian Trade Law and Tariff Schedules.
* **Tier 3: Local LLM** – Deep-reasoning classification via a fine-tuned Llama-3 (8B) model running natively on an NVIDIA RTX 4070.

## Core Features
* **AI Classifier:** Converts complex product descriptions into verified HS Codes.
* **Sourcing Optimizer:** Recommends alternative origins (e.g., Vietnam, India PLI) to mitigate duty exposure.
* **Audit Trail:** Generates immutable reports for legal and customs compliance.

## Setup
1. **Backend:** Initialize the FastAPI server in the `backend` directory to load the local Llama-3 weights.
2. **Frontend:** Run the React-Vite dashboard in the `frontend` directory to access the Classification Terminal.
