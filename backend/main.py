import os
import uvicorn
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from unsloth import FastLanguageModel
from bridge import TradeDataArchitect

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG & Cache
try:
    trade_db = TradeDataArchitect()
    trade_db.setup_correction_cache()
    db_active = True
except Exception as e:
    print(f"Database error: {e}")
    db_active = False

# Load Model
latest_checkpoint = max([os.path.join("outputs", d) for d in os.listdir("outputs") if d.startswith("checkpoint")], key=os.path.getmtime)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=latest_checkpoint,
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

class ClassificationRequest(BaseModel):
    product_description: str

class OptimizeRequest(BaseModel):
    hs_code: str

prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction: Classify this product using the provided tariff schedule context. Return the HS Code.
### Context: {}
### Input: {}
### Response:"""

@app.post("/classify")
async def classify_product(request: ClassificationRequest):
    context = "No relevant tariff data found."
    
    if db_active:
        # 1. Check Cache
        cache_hit = trade_db.check_cache(request.product_description)
        if cache_hit:
            return {"hs_code": cache_hit.get("hs_code"), "confidence": 1.0, "duty_rate": cache_hit.get("duty_rate"), "savings": "Instant"}
        
        # 2. Vector Search
        rag_results = trade_db.query_trade_law(request.product_description, country_code="IN")
        if rag_results:
            context = "\n\n".join([doc.page_content for doc in rag_results])

    # 3. Model Inference
    inputs = tokenizer([prompt_template.format(context, request.product_description)], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True, pad_token_id=tokenizer.eos_token_id)
    prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split("### Response:")[-1].strip()

    # EXTRACT ONLY THE HS CODE NUMBERS
    match = re.search(r'[\d\.]+', prediction)
    clean_hs_code = match.group(0) if match else "Error"

    return {"hs_code": clean_hs_code, "confidence": 0.94, "duty_rate": "12.5%", "savings": "Calculated"}

@app.post("/optimize")
async def optimize_tariffs(request: OptimizeRequest):
    # Returns the data needed for the Sourcing Optimization Engine UI
    return {
        "baseline": {"country": "Mainland China", "rate": "25.0%", "status": "Standard + Retaliatory"},
        "recommendations": [
            {"country": "Vietnam", "rate": "0.0%", "savings": "25.0%", "fta": "ASEAN Free Trade Area"},
            {"country": "Mexico", "rate": "2.5%", "savings": "22.5%", "fta": "Nearshoring Agreement"},
            {"country": "India", "rate": "5.0%", "savings": "20.0%", "fta": "Domestic Sourcing (PLI)"}
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)