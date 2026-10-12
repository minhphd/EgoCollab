"""Sentence embeddings with all-MiniLM-L6-v2 (unit vectors, so a dot product is the cosine similarity).

Loads from the Hugging Face cache; on CHPC set HF_HOME=<repo>/models/hf HF_HUB_OFFLINE=1.

    from helper.llm.embed import embed
    E = embed(["pick up the spatula", "Pan scraped"])    # (2, 384) float32
"""

import numpy as np

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def embed(texts: list[str], batch: int = 256) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok, model = AutoTokenizer.from_pretrained(MODEL), AutoModel.from_pretrained(MODEL).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=64, return_tensors="pt")
            h, m = model(**enc).last_hidden_state, enc["attention_mask"][..., None].float()
            out.append(torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).numpy())
    return np.concatenate(out) if out else np.zeros((0, 384), np.float32)
