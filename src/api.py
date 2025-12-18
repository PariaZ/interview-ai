# source/api.py
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

# ✅ You must provide these in rag_core.py (see notes below)
from src.rag_core import init_rag, answer_query_json


app = FastAPI(title="Policy RAG API", version="1.0.0")


class QueryRequest(BaseModel):
    q: str = Field(..., description="User question")
    top_k: int = Field(3, ge=1, le=20, description="How many chunks to retrieve")
    max_refs: int = Field(1, ge=1, le=10, description="How many references to return")


class QueryResponse(BaseModel):
    answer: str
    action_required: str
    references: list[str]


@app.on_event("startup")
def _startup() -> None:
    """
    Load FAISS + passages once on startup so requests are fast.
    """
    init_rag()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> Dict[str, Any]:
    try:
        result = answer_query_json(req.q, top_k=req.top_k, max_refs=req.max_refs)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Index files not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
