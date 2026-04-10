import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

def build_note_index():
    print("Loading NOTEEVENTS dataset...")
    csv_path = os.path.join(os.path.dirname(__file__), '../dataset/NOTEEVENTS_random.csv')
    
    # Load a tiny subset of the dataset so it processes almost instantly
    df = pd.read_csv(csv_path, nrows=10)
    
    # Identify the text column
    if 'TEXT' in df.columns:
        texts = df['TEXT'].dropna().tolist()
    else:
        text_col = [c for c in df.columns if 'text' in c.lower()]
        if len(text_col) > 0:
            texts = df[text_col[0]].dropna().tolist()
        else:
            texts = df.iloc[:, -1].dropna().astype(str).tolist()

    print(f"Loaded {len(texts)} texts. Generating embeddings using all-MiniLM-L6-v2...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    print("Building FAISS index for retrieved notes...")
    dimension = embeddings.shape[1]
    # We use an IndexFlatL2 for exact L2 search
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Use relative path to avoid Unicode issues in absolute paths on Windows
    # out_index_path = os.path.join(os.path.dirname(__file__), 'note_index.faiss')
    out_index_path = 'note_index.faiss'
    faiss.write_index(index, out_index_path)
    
    out_csv_path = 'indexed_notes.csv'
    pd.DataFrame({'text': texts}).to_csv(out_csv_path, index=False)
    
    print(f"Successfully saved {out_index_path} and {out_csv_path} in current directory.")

if __name__ == '__main__':
    build_note_index()
