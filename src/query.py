# src/query.py
from __future__ import annotations

import json
from src.rag import answer_query_json


def main() -> None:
    q = input("Enter your question: ").strip()
    if not q:
        print("Empty question.")
        return

    result = answer_query_json(q, top_k=3, max_refs=1)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

