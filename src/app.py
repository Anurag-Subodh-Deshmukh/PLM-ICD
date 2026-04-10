"""
FastAPI backend for PLM-ICD with RAG Evidence-Gated Fusion.

Pipeline:
  1. Tokenize input text -> chunks
  2. RoBERTa encoder -> hidden states
  3. LAAT attention -> label-wise vectors (d_l)   [pretrained weights]
  4. RAG retrieval -> evidence vectors (e_l)      [real FAISS search]
  5. Gated fusion: z_l = g_l * d_l + (1-g_l)*e_l [g_l hardcoded to 1]
  6. Final linear layer (third_linear) -> logits  [pretrained weights]
  7. Sigmoid -> probabilities
"""

import os
import sys
import json
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoConfig, AutoTokenizer

from modeling_roberta import RobertaForMultilabelClassification

# ── Global state ────────────────────────────────────────────────────────────────
MODEL = None
TOKENIZER = None
ICD_CODES = None
ICD_DESCRIPTIONS = None
CHUNK_SIZE = 128

app = FastAPI(title="PLM-ICD Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str
    top_k: int = 15
    threshold: float = 0.3


class CodeResult(BaseModel):
    code: str
    description: str
    probability: float


class PredictResponse(BaseModel):
    codes: list[CodeResult]


# ── Model loading ───────────────────────────────────────────────────────────────
def load_everything():
    global MODEL, TOKENIZER, ICD_CODES, ICD_DESCRIPTIONS

    base = os.path.dirname(__file__)

    # 1. Config & model
    config_dir = os.path.join(base, '..', 'additional_files')
    weights_path = os.path.join(base, '..', 'pretrained_weights', 'pytorch_model.bin')

    config = AutoConfig.from_pretrained(config_dir)
    if not hasattr(config, 'model_mode'):
        config.model_mode = 'laat'

    model = RobertaForMultilabelClassification(config)
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    MODEL = model
    print(f"[OK] Loaded PLM-ICD  (num_labels={config.num_labels}, hidden={config.hidden_size})")

    # 2. Tokenizer
    TOKENIZER = AutoTokenizer.from_pretrained(config_dir)
    print("[OK] Loaded tokenizer")

    # 3. ICD codes list
    codes_path = os.path.join(config_dir, 'ALL_CODES.txt')
    with open(codes_path, 'r') as f:
        ICD_CODES = [line.strip() for line in f if line.strip()]
    print(f"[OK] Loaded {len(ICD_CODES)} ICD codes")

    # 4. ICD descriptions (real CMS data)
    desc_path = os.path.join(base, 'icd9_descriptions.json')
    if os.path.exists(desc_path):
        with open(desc_path, 'r', encoding='utf-8') as f:
            ICD_DESCRIPTIONS = json.load(f)
        print(f"[OK] Loaded {len(ICD_DESCRIPTIONS)} ICD-9 descriptions (CMS)")
    else:
        print("[WARN] icd9_descriptions.json not found. Run download_icd9_descriptions.py first.")
        ICD_DESCRIPTIONS = {}


# ── Inference helpers ───────────────────────────────────────────────────────────
def tokenize_and_chunk(text: str):
    tokens = TOKENIZER.encode(text, add_special_tokens=False, max_length=4096, truncation=True)
    if len(tokens) % CHUNK_SIZE != 0:
        pad_len = CHUNK_SIZE - (len(tokens) % CHUNK_SIZE)
        tokens += [TOKENIZER.pad_token_id] * pad_len

    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).view(1, -1, CHUNK_SIZE)
    attention_mask = (input_ids != TOKENIZER.pad_token_id).long()
    return input_ids, attention_mask


def run_prediction(text: str, top_k: int = 15, threshold: float = 0.3):
    input_ids, attention_mask = tokenize_and_chunk(text)

    with torch.no_grad():
        outputs = MODEL(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    # Collect codes above threshold
    above = np.where(probabilities > threshold)[0]
    top_indices = np.argsort(probabilities)[::-1][:max(top_k, len(above))]

    results = []
    for idx in top_indices:
        prob = float(probabilities[idx])
        if prob < threshold:
            break
        code = ICD_CODES[idx] if idx < len(ICD_CODES) else f"IDX_{idx}"
        desc = ICD_DESCRIPTIONS.get(code, "Description not available")
        results.append(CodeResult(code=code, description=desc, probability=round(prob, 4)))

    return results


# ── Endpoint ────────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Clinical text cannot be empty.")

    try:
        codes = run_prediction(req.text, req.top_k, req.threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictResponse(codes=codes)


@app.get("/health")
async def health():
    return {"status": "ok", "num_labels": len(ICD_CODES) if ICD_CODES else 0}


# ── Startup ─────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    load_everything()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
