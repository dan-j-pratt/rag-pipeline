# Local Rag Assistant

A fully local Retrieval-Augmented Generation (RAG) pipeline that lets you chat with saved documents keeping your data on your machine

Tools include **Ollama**, **LangChain**, **ChromaDB**, and **Streamlit**

---

## What is RAG?

It is a LLM that only knows what it was trained on. RAG improves this by giving the model access to your docuements. When you ask a question:

1. Your question is converted into a vector
2. ChromaDB searches for the most similar chunks from your documents
3. Those chunks are then injected into the prompt as context
4. The LLM then generates an answer based on the context given

Everything runs locally on your machine

---

## Stack
| Tool | Function |
|---|---|
| [Ollama](https:/ollama.ai) | Runs LLM and embedding|
| ChromaDB | Stores and searches document embeddings |
| Streamlit | Cheat interface/UI |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/yourname/rag-pipeline.git
cd rag-pipeline
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Mac/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama

Download from [ollama.ai](https://ollama.ai) and install for your OS.

Then pull the required models:

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

---

## Usage

### Step 1 — Add your documents

Copy any PDF files you want to chat with into the `/docs` folder:

```
docs/
├── research_paper.pdf
├── lecture_notes.pdf
└── resume.pdf
```

### Step 2 — Ingest your documents

Run this once to chunk, embed, and store your documents in ChromaDB:

```bash
python ingest.py
```

You should see output like:

```
Checking for new documents...
Found 3 new file(s)
Loaded 24 page(s) from 3 new file(s)
Split into 87 chunks
Vector store saved to /chroma_db
Done! Tracking file updated.
```

### Step 3 — Start the app

```bash
streamlit run app.py
```

The app opens automatically in your browser at `http://localhost:8501`.

### Step 4 — Ask questions

Type any question about your documents in the chat box. The assistant will answer using only the content of your documents and show which files it pulled from.

---

## Adding New Documents

Just drop new PDFs into `/docs` and run `ingest.py` again. The pipeline tracks which files have already been processed and only ingests new ones — existing documents are not re-embedded.

```bash
python ingest.py
```

---

## How it Works

```
INGEST PIPELINE
───────────────────────────────────────────────
/docs  →  Load  →  Chunk  →  Embed  →  ChromaDB

QUERY PIPELINE
───────────────────────────────────────────────
Question  →  Embed  →  Search ChromaDB
                            ↓
                      Top 3 matching chunks
                            ↓
                    Inject into prompt
                            ↓
                      Ollama (Llama 3)
                            ↓
                    Answer + source citations
```

---

## Configuration

Key settings you can tweak in `ingest.py`:

| Setting | Default | Description |
|---|---|---|
| `chunk_size` | `1024` | Characters per chunk |
| `chunk_overlap` | `100` | Overlap between chunks |
| Embedding model | `nomic-embed-text` | Local embedding model |

Key settings in `query.py`:

| Setting | Default | Description |
|---|---|---|
| `k` | `3` | Number of chunks retrieved per query |
| `search_type` | `similarity` | ChromaDB search strategy |
| LLM model | `llama3` | Model used to generate answers |

---

## Known Limitations

- Currently supports PDF files only
- Deleting a file from `/docs` does not remove it from ChromaDB — rebuild the vector store manually if needed by deleting the `/chroma_db` folder and re-running `ingest.py`
- Response speed depends on your hardware — a GPU will significantly improve inference time

---

## Future Improvements

- Support for `.txt` and `.md` file types
- Automatic removal of deleted documents from ChromaDB
- Similarity score threshold to filter low-confidence answers
- Docker support for easy deployment

