# CSC525 Module 8: Portfolio Project
# Created by Jacinto Jimenez
# Created: 7/06/2025
import os
import streamlit as st
import tempfile
from pathlib import Path
import fitz  # PyMuPDF for PDF handling
import docx2txt
import pandas as pd
import json
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

# Page configuration
st.set_page_config(page_title="📚 Document Chatbot", layout="wide")
st.title("📚 Supportly Tech Support Chatbot")

# --- File upload / processing UI ---
st.header("1. Upload Documents")
files = st.file_uploader(
    label="Choose PDF, TXT, DOCX, or CSV files",
    type=["pdf", "txt", "docx", "csv"],
    accept_multiple_files=True,
)
process = st.button("Process & Index Documents")

def extract_text(streamlit_file):
    suffix = Path(streamlit_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(streamlit_file.read())
        tmp_path = tmp.name
    try:
        if suffix == ".pdf":
            doc = fitz.open(tmp_path)
            text = "".join(page.get_text("text") for page in doc)
        elif suffix == ".docx":
            text = docx2txt.process(tmp_path)
        elif suffix in {".txt", ".csv"}:
            text = open(tmp_path, "r", encoding="utf-8", errors="ignore").read()
        else:
            text = ""
    except Exception:
        text = ""
    return text

# Process uploads and build vectorstore
if process and files:
    all_docs = []
    for uploaded in files:
        raw = extract_text(uploaded)
        if not raw:
            continue
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(raw)
        docs = [
            Document(page_content=chunk, metadata={"source": uploaded.name})
            for chunk in chunks
        ]
        all_docs.extend(docs)

    if all_docs:
        embeddings = OpenAIEmbeddings()
        vs = FAISS.from_documents(all_docs, embeddings)
        vs.save_local("faiss_index")
        st.session_state.vectorstore = vs
        st.session_state.embeddings = embeddings
        st.success(f"Processed and indexed {len(all_docs)} chunks.")
    else:
        st.warning("No valid text extracted from uploaded files.")

# Ensure session state exists
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# --- Main Chat Interface ---
st.header("2. Chat with Your Documents")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input via chat UI
user_input = st.chat_input("Ask a question about the uploaded documents...")
if user_input:
    # 1) Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 2) Retrieve relevant document snippets, display in sidebar, and inject into the prompt
    if st.session_state.vectorstore:
        vs = st.session_state.vectorstore
        # fetch top 5 chunks
        top_chunks = vs.similarity_search(user_input, k=5)

        # Show them in the sidebar
        with st.sidebar:
            st.markdown("### 🔍 Top Matching Snippets")
            for i, chunk in enumerate(top_chunks, start=1):
                src = chunk.metadata.get("source", "unknown")
                snippet = chunk.page_content
                st.markdown(f"**{i}. Source:** {src}")
                st.write(snippet[:200] + ("…" if len(snippet) > 200 else ""))

        # Inject exactly the same context into your system prompt
        context = "\n\n".join([chunk.page_content for chunk in top_chunks])
        st.session_state.messages.append(
            {
                "role": "system",
                "content": f"Use the following context to answer the question:\n\n{context}",
            }
        )
    else:
        st.session_state.messages.append(
            {"role": "system", "content": "No documents indexed yet."}
        )

    # 3) Generate assistant response with streaming
    client = OpenAI()  # make sure your OPENAI_API_KEY is set in env
    with st.chat_message("assistant"):
        try:
            response_stream = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=st.session_state.messages,
                stream=True,
            )
            assistant_response = st.write_stream(response_stream)
        except Exception as e:
            assistant_response = f"Error: {e}"
            st.error(assistant_response)

    # 4) Save assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_response}
    )
