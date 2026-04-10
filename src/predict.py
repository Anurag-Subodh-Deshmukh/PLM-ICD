"""
PLM-ICD Inference with RAG Fusion
Loads the original pretrained weights and runs prediction on a clinical note.

Pipeline:
  1. Tokenize input text → chunks
  2. RoBERTa encoder → hidden states
  3. LAAT attention → label-wise vectors (d_l)  [pretrained weights]
  4. RAG retrieval → evidence vectors (e_l)     [real FAISS search]
  5. Gated fusion: z_l = g_l * d_l + (1-g_l) * e_l  [g_l hardcoded to 1]
  6. Final linear layer (third_linear) → logits  [pretrained weights]
  7. Sigmoid → probabilities
"""

import os
import sys
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer

from modeling_roberta import RobertaForMultilabelClassification


def load_model():
    """Load the PLM-ICD model with actual pretrained weights."""
    
    config_path = os.path.join(os.path.dirname(__file__), '..', 'additional_files')
    weights_path = os.path.join(os.path.dirname(__file__), '..', 'pretrained_weights', 'pytorch_model.bin')
    
    # Load config from the additional_files directory
    config = AutoConfig.from_pretrained(config_path)
    # Ensure model_mode is set (it should be 'laat' from the config)
    if not hasattr(config, 'model_mode'):
        config.model_mode = 'laat'
    
    print(f"Config: model_mode={config.model_mode}, num_labels={config.num_labels}, hidden_size={config.hidden_size}")
    
    # Instantiate model architecture (this creates random weights initially)
    model = RobertaForMultilabelClassification(config)
    
    # Load the actual pretrained weights from pytorch_model.bin
    print(f"Loading pretrained weights from: {os.path.basename(weights_path)}")
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
    
    # Load only the original PLM-ICD weights (ignore rag_extractor/rag_fusion keys which are new)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    print(f"Loaded pretrained weights successfully.")
    if missing:
        print(f"  Missing keys (expected - these are RAG fusion layers, not pretrained): {[k for k in missing if 'rag' in k]}")
        non_rag_missing = [k for k in missing if 'rag' not in k]
        if non_rag_missing:
            print(f"  WARNING - Missing non-RAG keys: {non_rag_missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")
    
    model.eval()
    return model, config


def load_tokenizer():
    """Load the tokenizer from additional_files."""
    tokenizer_path = os.path.join(os.path.dirname(__file__), '..', 'additional_files')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer


def load_icd_codes():
    """Load the ICD code list."""
    codes_path = os.path.join(os.path.dirname(__file__), '..', 'additional_files', 'ALL_CODES.txt')
    with open(codes_path, 'r') as f:
        codes = [line.strip() for line in f if line.strip()]
    return codes


def tokenize_and_chunk(text, tokenizer, chunk_size=128, max_length=4096):
    """Tokenize text and format it into chunks for the model."""
    tokens = tokenizer.encode(text, add_special_tokens=False, max_length=max_length, truncation=True)
    
    # Pad to multiple of chunk_size
    if len(tokens) % chunk_size != 0:
        pad_len = chunk_size - (len(tokens) % chunk_size)
        tokens = tokens + [tokenizer.pad_token_id] * pad_len
    
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # [1, total_len]
    input_ids = input_ids.view(1, -1, chunk_size)  # [1, num_chunks, chunk_size]
    
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    return input_ids, attention_mask


def predict(text, model, tokenizer, icd_codes, top_k=15, threshold=0.3):
    """
    Run full prediction pipeline:
      text → tokenize → RoBERTa → LAAT (d_l) → RAG fusion (z_l = d_l since g=1)
      → third_linear → logits → sigmoid → probabilities → top predictions
    """
    input_ids, attention_mask = tokenize_and_chunk(text, tokenizer)
    
    print(f"\nInput shape: {input_ids.shape}  (batch=1, chunks={input_ids.shape[1]}, chunk_size={input_ids.shape[2]})")
    
    with torch.no_grad():
        # Forward pass through the full model:
        # RoBERTa → LAAT attention → weighted_output (d_l) → RAG fusion → z_l → third_linear → logits
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [1, num_labels]
        
        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # [num_labels]
    
    # Get predictions above threshold
    above_threshold = np.where(probabilities > threshold)[0]
    
    # Also get top-k predictions
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    
    print(f"\n{'='*60}")
    print(f"TOP-{top_k} ICD CODE PREDICTIONS")
    print(f"{'='*60}")
    print(f"{'Rank':<6} {'ICD Code':<12} {'Probability':<12}")
    print(f"{'-'*30}")
    for rank, idx in enumerate(top_indices, 1):
        code = icd_codes[idx] if idx < len(icd_codes) else f"IDX_{idx}"
        prob = probabilities[idx]
        marker = " *" if prob > threshold else ""
        print(f"{rank:<6} {code:<12} {prob:<12.4f}{marker}")
    
    print(f"\n{len(above_threshold)} codes above threshold {threshold}")
    
    return probabilities, top_indices


def main():
    print("="*60)
    print("PLM-ICD with RAG Evidence-Gated Fusion")
    print("="*60)
    
    # Load everything
    print("\n[1/4] Loading model with pretrained weights...")
    model, config = load_model()
    
    print("\n[2/4] Loading tokenizer...")
    tokenizer = load_tokenizer()
    
    print("\n[3/4] Loading ICD codes...")
    icd_codes = load_icd_codes()
    print(f"  Loaded {len(icd_codes)} ICD codes")
    
    # Sample clinical note for demonstration
    sample_note = """
    DISCHARGE SUMMARY
    
    Patient is a 65-year-old male admitted for chest pain and shortness of breath.
    History of hypertension, type 2 diabetes mellitus, and coronary artery disease.
    ECG showed ST elevation in leads V1-V4 consistent with acute myocardial infarction.
    Patient underwent emergent cardiac catheterization with stent placement.
    Post-procedure course was complicated by acute kidney injury and atrial fibrillation.
    Patient was started on heparin drip and transitioned to warfarin.
    Blood cultures were negative. Creatinine peaked at 2.8 and trended down.
    Patient was discharged in stable condition on aspirin, metoprolol, lisinopril,
    metformin, and warfarin.
    
    DIAGNOSES:
    1. Acute ST-elevation myocardial infarction
    2. Coronary artery disease
    3. Essential hypertension
    4. Type 2 diabetes mellitus
    5. Acute kidney injury
    6. Atrial fibrillation
    """
    
    print(f"\n[4/4] Running inference...")
    print(f"  Input text length: {len(sample_note)} characters")
    
    probabilities, top_indices = predict(sample_note, model, tokenizer, icd_codes)


if __name__ == '__main__':
    main()
