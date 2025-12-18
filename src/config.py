import os


# ======================
# API Keys
# ======================

# OpenAI API key is read from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Please set it as an environment variable."
    )

# ======================
# Embedding configuration
# ======================

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small"
)

# ======================
# Paths
# ======================

# Directory containing markdown knowledge base documents
KB_DIR = os.getenv("KB_DIR", "kb")

# FAISS index path (binary vector index)
FAISS_INDEX_PATH = os.getenv(
    "FAISS_INDEX_PATH",
    "data/faiss/index.faiss"
)

# Passage metadata path (JSONL mapping vectors -> text)
PASSAGES_PATH = os.getenv(
    "PASSAGES_PATH",
    "data/faiss/passages.jsonl"
)
