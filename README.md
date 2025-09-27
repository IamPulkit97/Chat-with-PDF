# 📚 PDF Chat Assistant

A Streamlit-based PDF Q&A assistant using **Retrieval-Augmented Generation (RAG)**. Upload a PDF, ask questions, and get answers with page citations.

---

## **Setup & Installation**

1. **Clone the repository:**
```bash
git clone <repo-url>
cd <repo-directory>
```
2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

3.**Install dependencies:**
```bash
pip install -r requirements.txt
```

4.**Set environment variables:**
```bash
export OPENAI_API_KEY="your_openai_api_key"  # macOS/Linux
setx OPENAI_API_KEY "your_openai_api_key"    # Windows
```

5.**Run the app:**
```bash
streamlit run app.py
```

## Architecture & Components
1. PDF Processing
   Uses PyPDF2 to extract text from each page.
   Chunking strategy: RecursiveCharacterTextSplitter
     Chunk size: 1000 characters
     Chunk overlap: 200 characters
     Preserves semantic context across paragraphs.
   Each chunk retains page metadata for citation.

2. Vector Database
   FAISS is used for local vector indexing.
   Embedded chunks are stored for fast similarity search.
   Embeddings are generated using OpenAI Embedding model:
     text-embedding-3-small

4. RAG Chain
   Question answering pipeline using OpenAI Chat LLM (gpt-3.5-turbo).
   Retrieves top-k relevant chunks from FAISS (k=4) for each query.
   Integrates context + conversation history into prompts.
   Returns structured answers with sources (page numbers + snippet).

---

## Conversation History

1. Maintained internally in the RAGChain class.
2. Stores the last N exchanges (default: 6) to provide context for follow-up questions.
3. Supports multi-turn dialogue:
    User asks a question
    Model generates an answer referencing previous context
4. Streamlit session state also keeps chat messages for UI display.

**Models & APIs Used**

    Embeddings :	OpenAI text-embedding-3-small
    LLM / Answer Gen :	OpenAI Chat LLM gpt-3.5-turbo
    Vector DB :	FAISS (local)
    Chunking Strategy :	RecursiveCharacterTextSplitter

**Known Limitations**

1. PDF extraction depends on PyPDF2:
  May fail on scanned PDFs (no OCR support).
  Complex layouts may affect text order.
2. FAISS runs locally:
  Large PDFs may consume significant RAM.
  No distributed search support.
3. Answers rely on retrieved chunks:
  Model may hallucinate if retrieval misses relevant context.
  Citations are limited to chunk metadata; very large pages may truncate snippets.
4. No automatic support for multilingual PDFs.
5. LLM API key must be provided via environment variable.

**Future Enhancements**

1. Add OCR for scanned PDFs.
2. Support multi-file uploads and combined search.
3. Highlight keywords in source passages.
4. Optionally integrate cloud vector databases for scaling.
5. No Mulitilingual Support
