 ## 📌 Project Overview and Design Choices

This project implements a **retrieval-augmented knowledge assistant** for 
answering customer questions about **domain-related policies**, including 
**billing and renewal**, **WHOIS validation**, **abuse handling**, 
**domain suspension**, **reinstatement**, and **domain transfer**.

To simulate a realistic support scenario, I created five **synthetic 
policy documents** that resemble the kind of policy content a **domain 
registrar or hosting provider** would publish. The documents cover:

- **Billing and renewal** rules  
- **Suspension and reinstatement** conditions  
- **WHOIS validation** requirements  
- **Abuse escalation** and investigation procedures  
- **Domain transfer** policies  

Even though the documents are synthetic, they are written to closely match 
what customers would typically read in practice.

---

 ## 🧠 System Architecture

**Core components:**

- **RAG pipeline**
  - Policy documents are written in **Markdown** and chunked primarily by 
section (`##`)
  - For long sections, an additional **paragraph-level split** with a soft 
size limit is applied to preserve semantic coherence
  - Chunks are **embedded** using OpenAI’s **small embedding model**
  - All embeddings are **indexed using FAISS** for efficient 
nearest-neighbor retrieval
  - **Top-K** relevant chunks are retrieved for each query

- **Prompt-constrained LLM reasoning**
  - The system uses **GPT-4.1-mini** as the underlying language model, 
selected to balance **answer quality**, **latency**, and **computational 
efficiency**
  - Retrieved chunks are injected into a **strictly constrained prompt**
  - The model is **explicitly restricted** to the retrieved context
  - **Inference and speculation are forbidden**
  - If the context does not explicitly support an answer, the model must 
**refuse**

- **Structured outputs**
  - All responses are returned as valid **JSON** with a fixed schema to 
support determinism and evaluation

**Interfaces:**

- **FastAPI REST endpoint** (for testing and evaluation)
- **MCP server** (STDIO mode) to demonstrate **MCP-aligned tool design**



## 🔐 Model Context Protocol (MCP) Integration

This project follows **MCP principles** by exposing the **RAG logic as a 
tool** with a **strict contract**.

### Prompt Design (**MCP-aligned**)
The system follows **Model Context Protocol (MCP)** principles through a 
carefully designed prompt that explicitly defines:

- the model’s **role** as a policy support assistant,  
- the **allowed context** (retrieved policy documents only),  
- the **task** (answering or refusing based on explicit textual support), 
and  
- a **strict JSON output schema**.

An **MCP server** is implemented in **STDIO mode** to expose the 
retrieval-augmented reasoning logic as a tool with a clear input/output 
contract. No MCP client is required for evaluation; the server is included 
to demonstrate **MCP-aligned design** and modularity.

The codebase is structured in a **modular and maintainable** way, 
separating indexing, retrieval, prompting, API handling, and MCP 
integration. This separation improves **reliability**, **debuggability**, 
and **extensibility**.

The entire application is containerized using **Docker**, ensuring 
reproducibility and ease of execution across environments.

 The prompt 
explicitly defines:

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

The assistant is intentionally designed to **refuse** questions that fall 
outside the retrieved policy context.  
This includes queries requiring external knowledge, private or 
account-specific information, or undefined policies.

Refusal behavior is a **deliberate design choice** to ensure strict 
grounding and prevent hallucination.


## 🧪 Testing and Evaluation

A curated set of test questions is used to evaluate the system, including:

- questions that should be answered directly from policy context,  
- “specific-case” questions where the assistant must summarize 
**policy-level reasons** without guessing, and  
- **out-of-scope questions** that must be explicitly refused to prevent 
hallucination.

This testing strategy demonstrates the robustness of the RAG pipeline, the 
effectiveness of the MCP-aligned prompt, and the system’s ability to 
remain **strictly grounded** in its knowledge base.



## 🧪 Example Test Questions

**In-scope (answered):**

- “Why was my domain suspended?”
- “What happens if I don’t verify my WHOIS information?”
- “My domain was suspended due to a chargeback. What should I do?”

**Out-of-scope (refused):**

- “Why is my website loading slowly?”
- “What are the policies to have an account?”
- “Who reported my domain for abuse?”


 ## ▶️ How to Run th eProject

### 1. Build the Docker image

```bash
docker build -t interview-ai .
```

### 2. Run the FastAPI app (for testing)

```bash
docker run --env-file .env -p 8000:8000 interview-ai

```

### 3. Run the MCP server (STDIO mode)

```bash
docker run --init -it --env-file .env interview-ai python mcp_server.py

```

Open in your browser:

```text
http://localhost:8000/docs

```
Use POST /query with input:

```json

{
  "q": "Why was my domain suspended?",
  "top_k": 3,
  "max_refs": 1
}

```
### 3. Run the MCP server (STDIO mode)

```bash
 
docker run --init -it --env-file .env interview-ai python mcp_server.py

```
Expected output:



```text
MCP server running (stdio). Waiting for client...

```

## 🔮 Future Improvements

- With access to **larger and more realistic policy datasets**, the 
assistant could support a wider range of questions while still remaining 
strictly grounded in source documents.
- Collecting **real user-like sample questions** would make it possible to 
further **refine the prompts** and improve accuracy, especially for edge 
cases.
- As the policy data grows and becomes more **interconnected**, more 
advanced retrieval approaches such as **GraphRAG** could be used to better 
capture relationships across documents.
- For larger corpora and more complex queries, the system could be 
upgraded to models such as **GPT-4.1** or **GPT-4o**, which are better 
suited for long-context and multi-step reasoning.
- More advanced prompt techniques (**structured or constrained 
reasoning flows**) could also be explored, while still preserving **MCP 
compliance** and safety guarantees.

