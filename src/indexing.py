import json
import os
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
from openai import OpenAI

from .config import OPENAI_API_KEY, EMBEDDING_MODEL, FAISS_INDEX_PATH, PASSAGES_PATH, KB_DIR

client = OpenAI(api_key=OPENAI_API_KEY)


def load_documents() -> List[Dict]:
    """Load all files from the KB directory."""
    kb_path = Path(KB_DIR)
    docs = []

    if not kb_path.exists():
        raise FileNotFoundError(f"KB directory not found: {kb_path.resolve()}")

    for path in kb_path.glob("**/*"):
        if path.suffix.lower() in {".md", ".txt"}:
            text = path.read_text(encoding="utf-8")
            docs.append({"id": str(path), "text": text})
    return docs


def paragraph_chunk(text: str, max_chars: int = 800) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    for p in paragraphs:
        is_heading = p.startswith("## ") or p.startswith("### ")

        # force a new chunk at each new heading (section boundary)
        if is_heading and current.strip():
            chunks.append(current.strip())
            current = ""

        # normal size-based grouping
        if len(current) + len(p) + 2 > max_chars:
            if current.strip():
                chunks.append(current.strip())
            current = p
        else:
            current = (current + "\n\n" + p).strip()

    if current.strip():
        chunks.append(current.strip())

    return chunks



def embed_texts(texts: List[str]) -> List[List[float]]:
    
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in resp.data]


def build_index():
    docs = load_documents()
    passages = []
    vectors = []

    # 1) Create passages (chunks) from docs
    for doc in docs:
        chunks = paragraph_chunk(doc["text"])
        for i, chunk in enumerate(chunks):
            passages.append({
                "id": f"{doc['id']}#chunk={i}",
                "source": doc["id"],
                "chunk_index": i,
                "text": chunk,
            })
        
        print("=" * 80)
        print("FILE:", doc["id"])
        print("Num chunks:", len(chunks))
        print("Chunk headings preview:")
        for c in chunks[:5]:
            first_line = c.splitlines()[0] if c.strip() else ""
            print("  ", first_line[:120])
        print("=" * 80)
    print(f"[indexing] Total chunks: {len(passages)}")
    print(f"[indexing] Total documents: {len(docs)}")
 

    # 2) Embed in batches
    batch_size = 32
    for i in range(0, len(passages), batch_size):
        batch = passages[i:i + batch_size]
        texts = [p["text"] for p in batch]
        embs = embed_texts(texts)
        vectors.extend(embs)
        print(f"[indexing] Embedded {min(i + batch_size, len(passages))}/{len(passages)} chunks")

    # 3) Create FAISS index
    vectors_np = np.array(vectors, dtype="float32")
    dim = vectors_np.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors_np)

    # 4) Save index & metadata
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(PASSAGES_PATH, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p) + "\n")

    print(f"[indexing] Saved FAISS index to {FAISS_INDEX_PATH}")
    print(f"[indexing] Saved passages metadata to {PASSAGES_PATH}")


if __name__ == "__main__":
    # This lets you test THIS MODULE separately:
    #   python -m src.knowledge_assistant.indexing
    build_index()
