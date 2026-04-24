<div align="center">

# 🤖 Intelligent PDF Question-Answering System
### Powered by LangChain · ChromaDB · Sentence Transformers · RAG

<br/>

<br/><br/>

<!-- Tech Stack -->
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6B35?style=for-the-badge&logo=databricks&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

<br/>

<!-- Status & Meta -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-blueviolet?style=flat-square)
![Maintained](https://img.shields.io/badge/Maintained-Yes-success?style=flat-square)
[![GitHub Stars](https://img.shields.io/github/stars/saifullah857/Intelligent-PDF-question-answering-system-with-LangChain-?style=flat-square&color=yellow)](https://github.com/saifullah857/Intelligent-PDF-question-answering-system-with-LangChain-/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/saifullah857/Intelligent-PDF-question-answering-system-with-LangChain-?style=flat-square&color=orange)](https://github.com/saifullah857/Intelligent-PDF-question-answering-system-with-LangChain-/network)

<br/>

<!-- LLM Backends -->
![Gemini](https://img.shields.io/badge/Google-Gemini%202.5%20Flash-4285F4?style=flat-square&logo=google&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Qwen3--32B-F55036?style=flat-square)
![DeepSeek](https://img.shields.io/badge/DeepSeek-Chat-0EA5E9?style=flat-square)

<br/><br/>

> 💡 **Upload any PDF. Ask anything. Get precise, context-aware answers.**
> A full RAG pipeline — chunk, embed, store, retrieve, generate — built with LangChain, ChromaDB, Sentence Transformers, and your choice of LLM.

<br/>

[🏗️ Architecture](#️-pipeline-architecture) &nbsp;·&nbsp;
[🚀 Quick Start](#-quick-start) &nbsp;·&nbsp;
[💡 Usage](#-usage) &nbsp;·&nbsp;
[🤖 LLM Backends](#-llm-backends) &nbsp;·&nbsp;
[🔧 Configuration](#-configuration) &nbsp;·&nbsp;
[🤝 Contributing](#-contributing)

</div>

---

## 📌 Table of Contents

- [✨ Features](#-features)
- [🏗️ Pipeline Architecture](#️-pipeline-architecture)
- [📁 Project Structure](#-project-structure)
- [⚙️ Tech Stack](#️-tech-stack)
- [🚀 Quick Start](#-quick-start)
- [🔐 Environment Setup](#-environment-setup)
- [💡 Usage](#-usage)
- [🤖 LLM Backends](#-llm-backends)
- [🔧 Configuration](#-configuration)
- [📊 How It Works](#-how-it-works)
- [🗺️ Roadmap](#️-roadmap)
- [🙋 FAQ](#-faq)
- [🤝 Contributing](#-contributing)
- [👤 Author](#-author)
- [📄 License](#-license)

---

## ✨ Features

<div align="center">

| 🚀 Feature | 📝 Description |
|:---|:---|
| 📄 **Multi-PDF Ingestion** | Batch-load entire folders of PDFs automatically in one pipeline call |
| 📝 **Text File Support** | Load plain `.txt` files via `TextLoader` alongside PDFs |
| ✂️ **Intelligent Chunking** | Recursive character splitting that preserves semantic boundaries with overlap |
| 🧠 **Semantic Embeddings** | `all-MiniLM-L6-v2` dense vectors for meaning-based retrieval, not just keywords |
| 💾 **Persistent Vector Store** | ChromaDB stores embeddings on disk — no re-processing on restart |
| 🔍 **Cosine Similarity Search** | Score-filtered, ranked top-k retrieval for the most relevant context |
| 🤖 **Multi-LLM Q&A** | Plug in **Gemini 2.5 Flash**, **Groq Qwen3-32B**, or **DeepSeek Chat** |
| 💬 **Grounded Generation** | Answers built from retrieved context + LLM knowledge with graceful fallback |
| 🔌 **Fully Modular Design** | Swap any component: loader, embedder, vector store, or LLM with zero friction |
| 📊 **Source Attribution** | Every result traces back to the originating PDF page and document |

</div>

---

## 🏗️ Pipeline Architecture

```
╔═════════════════════════════════════════════════════════════════════════╗
║                      📥  INGESTION PIPELINE                            ║
╠═════════════════════════════════════════════════════════════════════════╣
║                                                                         ║
║   📂 PDF Folder / .txt File                                             ║
║         │                                                               ║
║         ▼                                                               ║
║   📄 PyPDFLoader / TextLoader    →   LangChain Document objects         ║
║         │                                                               ║
║         ▼                                                               ║
║   ✂️  RecursiveCharacterSplitter →   500-char chunks, 50-char overlap   ║
║         │                                                               ║
║         ▼                                                               ║
║   🧠 SentenceTransformer          →   384-dim semantic vectors          ║
║         │                                                               ║
║         ▼                                                               ║
║   💾 ChromaDB (Persistent)        →   Vectors + metadata saved to disk  ║
║                                                                         ║
╠═════════════════════════════════════════════════════════════════════════╣
║                   🔍  RETRIEVAL & GENERATION PIPELINE                  ║
╠═════════════════════════════════════════════════════════════════════════╣
║                                                                         ║
║   ❓ User Question                                                      ║
║         │                                                               ║
║         ▼                                                               ║
║   🧠 Embed Query                  →   Same model, same vector space     ║
║         │                                                               ║
║         ▼                                                               ║
║   📐 Cosine Similarity Search     →   Top-K nearest chunks              ║
║         │                                                               ║
║         ▼                                                               ║
║   📋 Ranked Context Chunks        →   Filtered by score threshold       ║
║         │                                                               ║
║         ▼                                                               ║
║   🤖 LLM (Gemini / Groq / DeepSeek) →  Context-grounded generation     ║
║         │                                                               ║
║         ▼                                                               ║
║   💬 Final Answer                 →   Precise, source-attributed reply  ║
║                                                                         ║
╚═════════════════════════════════════════════════════════════════════════╝
```

---

## 📁 Project Structure

```
📦 Intelligent-PDF-question-answering-system-with-LangChain/
│
├── 📂 data/
│   ├── 📂 pdfs/                    ← 📥 Drop your PDF documents here
│   ├── 📂 vector_store/            ← 🗃️  Auto-generated ChromaDB storage
│   └── 📄 Python.txt               ← 📝 Sample text for quick testing
│
├── 📓 main.ipynb                   ← 🚀 End-to-end pipeline notebook
├── 📄 .env                         ← 🔐 API keys (never commit this!)
├── 📄 requirements.txt             ← 📦 All Python dependencies
├── 📄 .gitignore
└── 📘 README.md
```

---

## ⚙️ Tech Stack

<div align="center">

| Layer | Badge | Library | Role |
|:---:|:---:|:---:|:---|
| **Orchestration** | ![LangChain](https://img.shields.io/badge/-LangChain-1C3C3C?style=flat-square&logo=chainlink&logoColor=white) | `langchain-community` | Pipeline orchestration & document loaders |
| **Text Splitting** | ![Split](https://img.shields.io/badge/-TextSplitter-6C63FF?style=flat-square) | `langchain-text-splitters` | Recursive character-based chunking |
| **Embeddings** | ![HF](https://img.shields.io/badge/-SentenceTransformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black) | `sentence-transformers` | Semantic dense vector encoding |
| **Vector Store** | ![Chroma](https://img.shields.io/badge/-ChromaDB-FF6B35?style=flat-square&logo=databricks&logoColor=white) | `chromadb` | Persistent vector database |
| **PDF Parsing** | ![PDF](https://img.shields.io/badge/-PyMuPDF%20%2F%20PyPDF-00897B?style=flat-square) | `pymupdf` / `pypdf` | High-fidelity PDF text extraction |
| **Similarity** | ![SK](https://img.shields.io/badge/-Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) | `scikit-learn` | Cosine similarity computation |
| **LLM — Gemini** | ![Gemini](https://img.shields.io/badge/-Gemini%202.5%20Flash-4285F4?style=flat-square&logo=google&logoColor=white) | `langchain-google-genai` | Google's multimodal LLM |
| **LLM — Groq** | ![Groq](https://img.shields.io/badge/-Groq%20Qwen3--32B-F55036?style=flat-square) | `langchain-groq` | Ultra-fast LPU inference |
| **LLM — DeepSeek** | ![DS](https://img.shields.io/badge/-DeepSeek%20Chat-0EA5E9?style=flat-square) | `langchain-deepseek` | High-capability open-weight model |

</div>

---

## 🚀 Quick Start

### 1️⃣ &nbsp;Clone the Repository

```bash
git clone https://github.com/saifullah857/Intelligent-PDF-question-answering-system-with-LangChain-.git
cd Intelligent-PDF-question-answering-system-with-LangChain-
```

### 2️⃣ &nbsp;Create & Activate Virtual Environment *(recommended)*

```bash
python -m venv venv

# ── Windows ──────────────────────────────
venv\Scripts\activate

# ── macOS / Linux ────────────────────────
source venv/bin/activate
```

### 3️⃣ &nbsp;Install All Dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>📋 <b>Or install packages manually</b></summary>

<br/>

```bash
# Core RAG dependencies
pip install langchain langchain-core langchain-community \
            pypdf pymupdf sentence-transformers          \
            chromadb scikit-learn langchain-text-splitters \
            python-dotenv

# LLM backends — install only what you need
pip install langchain-google-genai      # Gemini 2.5 Flash
pip install langchain-groq              # Groq (Qwen3-32B)
pip install langchain-deepseek          # DeepSeek Chat
```

</details>

### 4️⃣ &nbsp;Add Your PDF Documents

```
📂 data/pdfs/
  ├── research_paper.pdf
  ├── company_handbook.pdf
  └── any_document.pdf       ← Drop any PDFs here
```

### 5️⃣ &nbsp;Launch the Notebook

```bash
# Option A — VS Code (recommended)
code .
# Open main.ipynb → Run All

# Option B — Jupyter
jupyter notebook main.ipynb
```

---

## 🔐 Environment Setup

Create a `.env` file in the **project root** with API keys for the LLM backends you intend to use:

```env
# ── Google Gemini ─────────────────────────────────
GEMANAI_API_KEY=your_gemini_api_key_here

# ── Groq ──────────────────────────────────────────
GROQ_API_KEY=your_groq_api_key_here

# ── DeepSeek ──────────────────────────────────────
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```



Get your API keys here:

| Provider | Link |
|:---|:---|
| 🌐 Google Gemini | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) |
| ⚡ Groq | [console.groq.com/keys](https://console.groq.com/keys) |
| 🐋 DeepSeek | [platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys) |

---

## 💡 Usage

### 🔧 Initialize All Components

```python
# ── Embeddings ─────────────────────────────────────────────
embedding_manager = EmbeddingManager(model_name="all-MiniLM-L6-v2")

# ── Vector Store ───────────────────────────────────────────
vector_store = VectorStoreManager(
    persist_directory="data/vector_store",
    collection_name="pdf_documents"
)

# ── Retriever ──────────────────────────────────────────────
rag_retriever = RAGRetriever(embedding_manager, vector_store)
```

### 📥 Ingest Your PDFs

```python
# Step 1 — Load
all_docs = load_all_pdfs()
print(f"✅ Loaded {len(all_docs)} pages")

# Step 2 — Chunk
chunks = split_docs(all_docs, chunk_size=500, chunk_overlap=50)
print(f"✅ Created {len(chunks)} chunks")

# Step 3 — Embed
texts = [doc.page_content for doc in chunks]
embeddings = embedding_manager.generate_embeddings(texts)
print(f"✅ Generated embeddings: {embeddings.shape}")

# Step 4 — Store
vector_store.add_documents(chunks, embeddings)
print(f"✅ Stored in ChromaDB — ready to query!")
```

### ❓ Retrieve Relevant Chunks

```python
query = "What is the encoder-decoder architecture?"

results = rag_retriever.retrieve(query=query, top_k=5, score_threshold=0.3)

for doc in results:
    print(f"📌 Rank #{doc['rank']}  |  Score: {doc['similarity_score']:.4f}")
    print(f"📄 Source : {doc['metadata'].get('source', 'N/A')}")
    print(f"📃 Page   : {doc['metadata'].get('page', 'N/A')}")
    print(f"\n{doc['document'][:300]}...\n{'─'*60}")
```

### 💬 Generate a Full Answer

```python
answer = generate_output(
    query="What is the encoder-decoder architecture?",
    retriever=rag_retriever,
    llm=llm,        # Plug in any configured LLM
    top_k=3
)
print(answer)
```

**How `generate_output` works:**

| Scenario | Behavior |
|:---|:---|
| ✅ Context retrieved | Builds a `Context + Query` prompt — grounded, factual answer |
| ⚠️ No context found | Falls back to LLM's general knowledge gracefully |

---

## 🤖 LLM Backends

This system supports **three interchangeable LLM backends**. Configure your `.env` and initialize the one you need — all share the same `generate_output()` interface.

---

### 🌟 Google Gemini 2.5 Flash

[![Gemini](https://img.shields.io/badge/Google-Gemini%202.5%20Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://aistudio.google.com)

```python
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os

load_dotenv(".env")
api_key = os.getenv("GEMANAI_API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key

llm = ChatGoogleGenerativeAI(
    api_key=api_key,
    model="gemini-2.5-flash",
    temperature=0.1,
    max_output_tokens=2048
)

answer = generate_output("AI/ML Road Map", rag_retriever, llm)
print(answer)
```

> ✅ Best for: **precise, low-temperature factual Q&A** over technical documents.

---

### ⚡ Groq — Qwen3-32B

[![Groq](https://img.shields.io/badge/Groq-Qwen3--32B-F55036?style=for-the-badge)](https://console.groq.com)

```python
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

load_dotenv(".env")
api_key = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = api_key

llm = ChatGroq(
    api_key=api_key,
    model_name="qwen/qwen3-32b",
    temperature=0.7,
    max_tokens=1024
)

answer = generate_output("AI/ML Road Map", rag_retriever, llm)
print(answer)
```

> ✅ Best for: **ultra-fast inference** — Groq's dedicated LPU hardware delivers near-instant responses.

---

### 🐋 DeepSeek Chat

[![DeepSeek](https://img.shields.io/badge/DeepSeek-Chat-0EA5E9?style=for-the-badge)](https://platform.deepseek.com)

```python
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
import os

load_dotenv(".env")
api_key = os.getenv("DEEPSEEK_API_KEY")
os.environ["DEEPSEEK_API_KEY"] = api_key

model = ChatDeepSeek(
    api_key=api_key,
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=1024
)

answer = generate_output("What is RAG?", rag_retriever, model)
print(answer)
```

> ✅ Best for: **cost-effective, high-capability** generation with strong reasoning performance.

---

### 🔄 LLM Comparison

| Feature | Gemini 2.5 Flash | Groq Qwen3-32B | DeepSeek Chat |
|:---|:---:|:---:|:---:|
| Speed | ⚡⚡ | ⚡⚡⚡ | ⚡⚡ |
| Reasoning Quality | ★★★★☆ | ★★★★☆ | ★★★★★ |
| Cost | Low | Free tier available | Very Low |
| Best For | Precision Q&A | Low-latency apps | Complex reasoning |

---

## 🔧 Configuration

```python
# ╔═══════════════════════════════════════════════════════════╗
# ║             ⚙️  PIPELINE CONFIGURATION                   ║
# ╠═══════════════════════════════════════════════════════════╣

# ── Document Chunking ────────────────────────────────────────
CHUNK_SIZE      = 500    # Max characters per chunk
CHUNK_OVERLAP   = 50     # Overlap to preserve cross-boundary context

# ── Embedding Model ──────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 384-dim | CPU-friendly | Fast

# ── ChromaDB Vector Store ────────────────────────────────────
PERSIST_DIR     = "data/vector_store"
COLLECTION_NAME = "pdf_documents"

# ── Retrieval Parameters ─────────────────────────────────────
TOP_K           = 5      # Chunks to retrieve per query
SCORE_THRESHOLD = 0.0    # Min cosine similarity (raise to 0.3–0.5 for precision)

# ── Generation ───────────────────────────────────────────────
GENERATE_TOP_K  = 3      # Context chunks passed to LLM prompt

# ╚═══════════════════════════════════════════════════════════╝
```

> 💡 **Pro Tip:** Start with `score_threshold=0.0` to inspect all results, then raise it to `0.4` for tighter, higher-quality answers on technical documents.

---

## 📊 How It Works

<details>
<summary><b>📄 Step 1 — Document Loading</b></summary>
<br/>
<code>PyPDFLoader</code> reads every <code>.pdf</code> in <code>data/pdfs/</code>. Each page becomes a LangChain <code>Document</code> with <code>page_content</code> and <code>metadata</code> (source path, page number). <code>TextLoader</code> handles plain <code>.txt</code> files. Multiple sources are merged into a single document list automatically.
</details>

<details>
<summary><b>✂️ Step 2 — Intelligent Chunking</b></summary>
<br/>
<code>RecursiveCharacterTextSplitter</code> splits using a separator hierarchy: <code>\n\n</code> → <code>\n</code> → <code>" "</code>. This preserves paragraph and sentence structure far better than naive character splitting. The 50-character overlap ensures context is never severed at boundaries.
</details>

<details>
<summary><b>🧠 Step 3 — Semantic Embedding</b></summary>
<br/>
Each chunk is encoded into a 384-dimensional dense vector by <code>all-MiniLM-L6-v2</code>. Trained for semantic textual similarity, this model places related chunks geometrically close in vector space — regardless of exact word matches.
</details>

<details>
<summary><b>💾 Step 4 — Persistent Vector Storage</b></summary>
<br/>
ChromaDB persists each vector with its raw text and metadata to a local database. UUIDs prevent duplicate entries. The store survives session restarts — no re-embedding required on subsequent runs.
</details>

<details>
<summary><b>🔍 Step 5 — Retrieval & Ranking</b></summary>
<br/>
The user's question is embedded with the same model. ChromaDB performs approximate nearest-neighbor search to find top-k candidates. Cosine similarity re-scores results, which are filtered by threshold and returned sorted by relevance rank.
</details>

<details>
<summary><b>💬 Step 6 — Context-Grounded Generation</b></summary>
<br/>
Retrieved chunks are injected into a structured prompt alongside the user's question. The LLM (Gemini / Groq / DeepSeek) generates a final answer grounded in your documents. If no relevant context is found, the system gracefully falls back to the LLM's general knowledge — no crash, no empty response.
</details>

---

## 🗺️ Roadmap

- [x] PDF & TXT document ingestion
- [x] Recursive text chunking with overlap
- [x] Local sentence-transformer embeddings
- [x] ChromaDB persistent vector store
- [x] Cosine similarity retrieval with ranking
- [x] Gemini 2.5 Flash LLM backend
- [x] Groq (Qwen3-32B) LLM backend
- [x] DeepSeek Chat LLM backend
- [x] Graceful fallback when no context is retrieved
- [ ] Streamlit / Gradio chat UI
- [ ] Multi-collection & multi-tenant support
- [ ] Hybrid search (BM25 + semantic)
- [ ] Document re-ranking (cross-encoder)
- [ ] Chat history & multi-turn conversation
- [ ] REST API with FastAPI

---

## 🙋 FAQ

<details>
<summary><b>❓ Do I need a GPU to run this?</b></summary>
<br/>
No — <code>all-MiniLM-L6-v2</code> runs efficiently on CPU. A GPU will significantly speed up embedding for large-scale ingestion (thousands of pages).
</details>

<details>
<summary><b>❓ Can I swap the embedding model?</b></summary>
<br/>
Yes. Pass any HuggingFace sentence-transformers model to <code>EmbeddingManager(model_name="...")</code>. Good alternatives: <code>all-mpnet-base-v2</code> (higher accuracy, slower) or <code>multi-qa-MiniLM-L6-cos-v1</code> (optimized for Q&A).
</details>

<details>
<summary><b>❓ Which LLM backend should I use?</b></summary>
<br/>
Use <strong>Gemini</strong> for precise factual Q&A at low temperature. Use <strong>Groq</strong> when response speed matters most. Use <strong>DeepSeek</strong> for complex reasoning tasks at low cost. All three use the same <code>generate_output()</code> interface — just swap the <code>llm</code> object.
</details>

<details>
<summary><b>❓ Will my vector store persist between sessions?</b></summary>
<br/>
Yes — ChromaDB's <code>PersistentClient</code> saves everything to <code>data/vector_store/</code>. Just re-initialize <code>VectorStoreManager</code> and your embeddings are immediately ready to query.
</details>

<details>
<summary><b>❓ How do I add more PDFs after initial ingestion?</b></summary>
<br/>
Drop new PDFs into <code>data/pdfs/</code> and re-run the ingestion cells. ChromaDB assigns new UUIDs — existing documents are not overwritten, only new ones are appended.
</details>

---


---


---

