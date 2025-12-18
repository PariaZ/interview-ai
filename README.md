# Policy Knowledge Assistant (**RAG + MCP**)

This project implements a **retrieval-augmented knowledge assistant** for 
answering customer questions about **domain-related policies**, including 
**billing**, **renewal**, **WHOIS validation**, **abuse handling**, 
**suspension**, and **reinstatement**.

The system is designed to be **grounded**, **non-hallucinatory**, and 
**policy-faithful**, following **Model Context Protocol (MCP)** principles 
with **strict prompt constraints** and **structured JSON outputs**.

## 📌 Scope of Knowledge

The assistant answers questions **only** based on the following **policy 
documents**:

- **Billing and Renewal Policy**
- **Domain Suspension Policy**
- **WHOIS Validation Requirements**
- **Abuse Escalation Guidelines**
- **Domain Transfer Policy**

Questions outside this scope (e.g., **website performance**, **account 
creation**, **SEO**, **internal systems**) are intentionally **refused**.

## 🧠 System Architecture

**Core components:**

- **RAG pipeline**
  - Markdown documents are chunked by section (`##`)
  - Chunks are **embedded** and **indexed using FAISS**
  - **Top-K** relevant chunks are retrieved per query

- **Prompt-constrained LLM reasoning**
  - The model is **explicitly restricted** to the retrieved context
  - **Inference and speculation are forbidden**
  - If the context does not explicitly support an answer, the model must 
**refuse**

- **Structured outputs**
  - All responses are valid **JSON** with a **fixed schema**

**Interfaces:**

- **FastAPI REST endpoint** (for testing and evaluation)
- **MCP server** (STDIO mode) to demonstrate **MCP-aligned tool design**



## 🔐 Model Context Protocol (MCP) Integration

This project follows **MCP principles** by exposing the **RAG logic as a 
tool** with a **strict contract**.

### Prompt Design (**MCP-aligned**)

The prompt explicitly defines:

- **Role**
  - You are a **policy support assistant**.

- **Context**
  - Only **policy documents** related to **domains**, **billing**, 
**WHOIS**, **abuse**, **suspension**, and **reinstatement**.

- **Task**
  - Answer the question **only if** the context **explicitly contains** 
the required information.

- **Constraints**
  - **No inference**, **guessing**, or **speculation**
  - **No account-specific assumptions**
  - “**My domain**” questions are handled by **summarizing policy-level 
reasons only**

### Output Schema (**strict JSON**)

```json
{
  "answer": "...",
  "references": ["document.md##section"],
  "action_required": "none | update_whois | contact_support | 
escalate_to_abuse_team"
}
```

If the context does not clearly support an answer, the system responds 
with:


```json
{
  "answer": "I don't know based on the provided context.",
  "references": [],
  "action_required": "none"
}
```


## 🚫 Hallucination Prevention

The assistant intentionally **refuses to answer**:

- Questions about **website performance** (e.g., “Why is my site slow?”)
- Questions about **account creation or eligibility**
- Questions about **internal systems**, **SLAs**, or **decision makers**
- Questions requiring **private or account-specific data**
- Questions referencing **non-existent policies**

This refusal behavior is a **feature**, not a limitation, and demonstrates 
**strict context grounding**.


## ▶️ How to Run the Project

### 1. Build the Docker image

```bash
docker build -t interview-ai .

