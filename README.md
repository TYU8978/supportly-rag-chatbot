# Supportly: Document‑Aware AI Assistant (RAG Chatbot)

**Supportly** is an AI‑powered, document‑aware assistant for healthcare and customer support workflows. Users upload PDFs, DOCX, TXT, or CSV files and then chat naturally; the system retrieves the most relevant passages and generates clear, empathetic answers grounded in those sources. Built with **Python + Streamlit**, **LangChain**, **OpenAI GPT**, and **FAISS** vector search, it demonstrates a practical Retrieval‑Augmented Generation (RAG) pipeline with intent handling and a clean web UI.

---

## Table of Contents
- [Why this Project](#why-this-project)
- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Data & Datasets](#data--datasets)
- [Setup](#setup)
- [Running the App](#running-the-app)
- [Usage Guide](#usage-guide)
- [Configuration](#configuration)
- [Security & Privacy Notes](#security--privacy-notes)
- [Limitations & Roadmap](#limitations--roadmap)
- [Project Structure](#project-structure)
- [Acknowledgments](#acknowledgments)

---

## Why this Project

The app targets **closed‑domain** assistance where accuracy and safety matter (e.g., health support and customer service). Instead of scripted rules, it retrieves answers directly from user‑provided materials and synthesizes helpful responses with an LLM—making it useful for **intake, FAQs, triage support, and policy explanation**.

---

## Key Features

- **Multi‑format ingestion**: PDF, DOCX, TXT, and CSV accepted via a simple Streamlit UI.
- **Efficient chunking**: Text split into ~1000‑character chunks with 100 overlap to balance recall and context.
- **Vector search (FAISS)**: Documents embedded (OpenAIEmbeddings) and indexed; top‑k similarity search powers retrieval.
- **RAG prompting**: Retrieved snippets are injected into the system prompt to ground answers; the same context is shown in the sidebar for transparency.
- **Intent handling** (design): DistilBERT fine‑tuning is described for intent classification to route/help interpret queries (in the academic write‑up).
- **Streaming chat UI**: Conversational interface with persisted history and streamed GPT responses.

---

## Architecture Overview

**High‑level flow**:
1) **Upload & parse** files → 2) **Split** into chunks → 3) **Embed** chunks → 4) **Index** in FAISS → 5) **User asks a question** → 6) **Similar chunks retrieved (k=5)** → 7) **RAG prompt built with context** → 8) **GPT generates response** → 9) **Show answer + top snippets**.

**Why RAG?**
RAG improves factual grounding and traceability, reducing hallucinations by conditioning LLM responses on retrieved domain text. The accompanying paper explains the hybrid design and evaluation choices (BLEU/ROUGE, user feedback loops).

---

## Tech Stack

- **Frontend**: Streamlit for a fast, accessible web UI.
- **Parsing**: PyMuPDF (`fitz`) for PDF, `docx2txt` for Word, direct read for TXT/CSV.
- **Text Processing**: LangChain `RecursiveCharacterTextSplitter` for chunking (size=1000, overlap=100).
- **Embeddings**: `OpenAIEmbeddings` for vectorization.
- **Vector Store**: FAISS for similarity search; persisted locally with `save_local("faiss_index")`.
- **LLM**: OpenAI Chat Completions API (`gpt-3.5-turbo` in the reference code; configurable).
- **(Design)**: DistilBERT intent classifier described and evaluated in the academic report.

---

## Data & Datasets

The academic report discusses training/evaluation materials including **Ubuntu Dialogue Corpus**, **Bitext customer support dataset**, and **MakTek FAQs**, which informed dialogue realism and intent coverage. (These are **not required** to run the demo app.)

---

## Setup

### 1) Environment
- Python 3.10+ recommended
- Create and activate a virtual environment (`venv` or `conda`).

### 2) Install dependencies
```bash
pip install streamlit pymupdf docx2txt pandas langchain langchain-community langchain-openai openai faiss-cpu
```
(Use `faiss-gpu` if you have CUDA.)

### 3) Configure API key
Set your OpenAI API key as an environment variable:
```bash
# macOS/Linux
export OPENAI_API_KEY="sk-..."
# Windows (PowerShell)
setx OPENAI_API_KEY "sk-..."
```
The app uses `OpenAI()` client and `OpenAIEmbeddings`, which read from the environment.

---

## Running the App
From the project root:
```bash
streamlit run streamlit_app_chat_main.py
```
This launches the web UI.

---

## Usage Guide

1. **Upload documents**: Click **“Process & Index Documents”** after selecting PDF/TXT/DOCX/CSV files. The app extracts text, splits into chunks, embeds, and builds a FAISS index. You’ll see a success message with the number of indexed chunks.
2. **Ask questions**: Use the chat box. The app retrieves the **top 5** similar chunks and shows them in the **sidebar**. The same context is injected into the system prompt to ground the answer.
3. **Review answers**: Responses stream in real time. Chat history is preserved for context during the session.

**Tip**: Start with specific, factual questions tied to your uploaded materials (e.g., “What’s the intake policy for minors?”).

---

## Configuration

Common parameters (adjust in `streamlit_app_chat_main.py`):
- **Chunking**: `chunk_size=1000`, `chunk_overlap=100`.
- **Retrieval depth**: `similarity_search(..., k=5)`.
- **Model**: `model="gpt-3.5-turbo"` (change to `gpt-4o` or enterprise model if available).
- **Persistence**: FAISS saved to `faiss_index`; can be re‑loaded for faster startup.

---

## Security & Privacy Notes

- Uploaded files are processed in memory and temporarily written to secure temp files for parsing; FAISS index is stored locally by default. Review retention policies if deploying to cloud.
- For PHI or sensitive content, add **PII scrubbing**, **role‑based access**, **encryption at rest/in‑transit**, and **audit logging** before production use. (Design guidance based on the project’s healthcare orientation.)

---

## Limitations & Roadmap

- **Latency & cost**: External LLM calls can add delay and cost; consider response caching and smaller/faster models for some queries.
- **Scale**: Very large corpora may degrade retrieval quality; consider hierarchical indexes, metadata filtering, or hybrid search.
- **Planned**: Evaluate *TextSearchBERT* for improved retrieval precision; optional intent router; support for local models; advanced memory.

---

## Project Structure

```
.
├── streamlit_app_chat_main.py   # Streamlit UI + RAG pipeline (upload, chunk, embed, FAISS, chat)
├── faiss_index/                 # Local FAISS index (created at runtime)
└── README.md                    # (this file)
```


---

## Acknowledgments

- Academic project report and methodology: closed‑domain chatbot, datasets, RAG design and evaluation.
- Implementation details: Streamlit UI, file parsing, chunking, embeddings, FAISS, OpenAI Chat Completions with streaming.

---

> **Quick blurb for job applications**
> Built a document‑aware RAG assistant for healthcare/customer support: Streamlit UI, multi‑format ingestion, FAISS vector search, OpenAI GPT chat with injected context, and transparent retrieval snippets. Designed and evaluated in a closed‑domain setting with intent classification and roadmap toward improved retrieval and safety.
