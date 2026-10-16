# 🚀 RAG Document Assistant (FastAPI, ChromaDB & Gemini API)

An end-to-end **Retrieval-Augmented Generation (RAG)** application built to ingest, index, and query PDF documents in real-time. Powered by Google's GenAI models (`gemini-embedding-001` and `gemini-2.5-flash`), FastAPI, ChromaDB, and Streamlit.

---

## 📌 Features

- **PDF Ingestion & Processing**: Extract text from PDF documents cleanly using PyMuPDF.
- **Vector Indexing**: Generate high-dimensional vector embeddings with Google's `gemini-embedding-001` model and persist them locally in **ChromaDB**.
- **Context-Grounded Answers**: Connect query endpoints with **Gemini 2.5 Flash** to synthesize precise, hallucination-reduced answers accompanied by source page citations.
- **Asynchronous REST API**: Asynchronous backend built with **FastAPI** to handle file upload, chunking, and search endpoints.
- **Interactive UI**: User-friendly frontend built with **Streamlit** for uploading documents, adjusting retrieval settings, and chatting in real-time.
- **Security & Best Practices**: Environment configuration (`.env`) for safe API key management and robust `.gitignore`.

---

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **Backend**: FastAPI, Uvicorn
- **LLM & Embeddings**: Google GenAI SDK (`gemini-2.5-flash`, `gemini-embedding-001`)
- **Vector Database**: ChromaDB
- **Frontend**: Streamlit
- **PDF Processing**: PyMuPDF (`fitz`)
- **Version Control**: Git & GitHub

---

## 📁 Repository Structure

```text
rag-pdf-fastapi-streamlit/
├── step1_basics/       # Early experiments with embeddings & basic LLM calls
│   ├── embedding.py
│   └── rag.py
├── step2/              # Single-file RAG testing with PyMuPDF
│   ├── CV_David_Stocco.pdf
│   └── ragpdf.py
├── step3/              # Complete Full-Stack Application
│   ├── backend.py      # FastAPI server handling ingestion & RAG endpoints
│   └── frontend.py     # Streamlit UI for user interaction
├── .gitignore          # Excludes venv, .env, chroma_db/
└── README.md           # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/davidstocco2024-cell/rag-pdf-fastapi-streamlit.git
cd rag-pdf-fastapi-streamlit
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install fastapi uvicorn streamlit chromadb google-genai pymupdf python-dotenv
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

---

## 🚀 How to Run the Application

### Step 1: Start the FastAPI Backend
```bash
uvicorn step3.backend:app --reload
```
The API will be available at `http://127.0.0.1:8000`. You can explore the interactive docs at `http://127.0.0.1:8000/docs`.

### Step 2: Launch the Streamlit Frontend
In a new terminal window (with the virtual environment activated):
```bash
streamlit run step3/frontend.py
```

---

## 👨‍💻 Author

**Ignacio David Stocco**
- **GitHub**: [@davidstocco2024-cell](https://github.com/davidstocco2024-cell)
- **LinkedIn**: [Ignacio David Stocco](https://www.linkedin.com)
