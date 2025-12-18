# src/rag.py
from __future__ import annotations

import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# FAISS must exist in Docker image
import faiss

# OpenAI client (new SDK style)
from openai import OpenAI


# ---------- Paths (robust in Docker) ----------
# This file lives at: <repo>/src/rag.py
# Repo root is one directory up from src/
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FAISS_INDEX_PATH = os.path.join(REPO_ROOT, "data", "faiss", "index.faiss")
PASSAGES_PATH = os.path.join(REPO_ROOT, "data", "faiss", "passages.jsonl")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")

# ---------- Globals (loaded once) ----------
_index: Optional[faiss.Index] = None
_passages: Optional[List[Dict[str, Any]]] = None

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def init_rag() -> None:
    """
    Load FAISS index + passages once. Safe to call multiple times.
    """
    global _index, _passages

    if _index is not None and _passages is not None:
        return

    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at: {FAISS_INDEX_PATH}")

    if not os.path.exists(PASSAGES_PATH):
        raise FileNotFoundError(f"Passages file not found at: {PASSAGES_PATH}")

    _index = faiss.read_index(FAISS_INDEX_PATH)

    passages: List[Dict[str, Any]] = []
    with open(PASSAGES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            passages.append(json.loads(line))
    _passages = passages


def _embed_text(text: str) -> np.ndarray:
    """
    Embed a query text to a float32 numpy vector.
    Uses cosine similarity by normalizing vectors.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Pass it into Docker env.")

    client = OpenAI(api_key=api_key)

    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    vec = np.array(resp.data[0].embedding, dtype=np.float32)
    # normalize for cosine similarity
    norm = np.linalg.norm(vec) + 1e-12
    return vec / norm


def search(q: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Vector search against FAISS. Returns list of hits with score/source/text.
    """
    init_rag()
    assert _index is not None and _passages is not None

    qv = _embed_text(q).reshape(1, -1)

    scores, ids = _index.search(qv, top_k)

    results: List[Dict[str, Any]] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        p = _passages[idx]
        results.append({
            "score": float(score),
            "source": p.get("source", ""),
            "chunk_id": p.get("chunk_id", idx),
            "text": p.get("text", "")
        })
    return results


def extract_reference(hit: Dict[str, Any]) -> str:
    """
    Prefer the first '## ' heading inside the chunk for stable references.
    """
    text = hit.get("text", "") or ""
    m = re.search(r"^\s*##\s+(.*)$", text, flags=re.MULTILINE)
    if m:
        section = m.group(1).strip()
        return f"{hit.get('source','')}##{section}"

    # fallback: first non-empty line
    for line in text.splitlines():
        line = line.strip()
        if line:
            return f"{hit.get('source','')}#{line[:80]}"
    return f"{hit.get('source','')}#Unknown"


def _strip_json_fences(raw: str) -> str:
    """
    Your model sometimes returns ```json ... ```. Remove fences.
    """
    s = (raw or "").strip()
    s = re.sub(r"^\s*```json\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*```\s*", "", s)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def answer_query_json(q: str, top_k: int = 3, max_refs: int = 1) -> Dict[str, Any]:
    """
    RAG answer in strict JSON + references.
    """
    hits = search(q, top_k=top_k)

    if not hits:
        return {
            "answer": "I don't know based on the provided context.",
            "action_required": "none",
            "references": []
        }


    references = [extract_reference(h) for h in hits[:max_refs]]

    context = "\n\n---\n\n".join(
        f"Source: {h['source']}\n{h['text']}" for h in hits
    )

    prompt = f"""You are a policy support assistant.

Your task is to answer the user's question using ONLY the provided context.
The context consists of policy documents related to domains, billing, WHOIS validation, abuse handling, suspension, and reinstatement.

Answer the question ONLY if the context explicitly contains
the information required to answer it.

Return STRICT JSON with exactly:
- answer (string)
- action_required (string)

Valid action_required values:
- none
- update_whois
- contact_support
- escalate_to_abuse_team

If the question refers to a specific case (e.g., "my domain") and the context
contains only general policy information:

- Do NOT guess the exact cause.
- Summarize the possible reasons explicitly mentioned in the context.
- Recommend the best next step as action_required (often contact_support).
- Do NOT infer, assume, or speculate beyond what is directly stated.

If the context does not provide sufficient information to even summarize
possible reasons or next steps, respond with:
"I don't know based on the provided context."
Question:
{q}

Context:
{context}

Return ONLY raw JSON (no ```json fences, no markdown, no extra text).
""".strip()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Pass it into Docker env.")

    client = OpenAI(api_key=api_key)

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    raw = resp.choices[0].message.content
    raw = _strip_json_fences(raw)

    try:
        model_output = json.loads(raw)
    except json.JSONDecodeError:
        # as a fallback, try to extract the JSON object region
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            model_output = json.loads(raw[start:end+1])
        else:
            raise

    # attach trusted references
    model_output["references"] = references

    # simple post-fix: if it still says none for "my domain", set contact_support
    if model_output.get("action_required") == "none" and "my domain" in q.lower():
        model_output["action_required"] = "contact_support"

    return model_output
