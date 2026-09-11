# CarnetLM

A private, local-first AI research assistant for organizing sources, asking questions, and taking notes.

CarnetLM lets you group PDFs, articles, web pages, and video transcripts into separate subject notebooks. You can search and chat with them using semantic-hybrid RAG with page citations, and study the material with an interactive, Keep-style spaced repetition deck.

---

## Key Features

- **Local Privacy**: Uploaded PDFs, vector databases, SQLite metadata, and text memories are stored in the git-ignored `data/` directory. No telemetry or cloud storage. A fresh clone starts with an empty environment.
- **Multi-Notebook Vaults**: Keep projects, courses, or research themes in separate workspaces. Notebooks can be password-protected (verified client-side) to lock sensitive files.
- **Multi-Format Source Ingestion**:
  - Documents: local PDF, TXT, and Markdown files.
  - OCR: text extraction from uploaded images.
  - Web scraping: article extraction from a URL.
  - YouTube transcripts: pulled directly from video links.
  - Clipboard: paste text in as a source.
- **Grounded RAG Q&A with Citations**: Ask questions in plain language and get streaming answers grounded in your sources, with citations pointing back to specific paragraphs and pages.
- **Flashcard Study Decks**:
  - Auto-generate flashcards from a notebook using AI.
  - Drag-and-drop card ordering in a Keep-style grid.
  - Leitner spaced repetition: grade reviews to move cards through boxes 1 to 5.
  - Sidebar showing study progress, with an option to reset counts.
- **Integrated Editor with AI Assist**: Write synthesis documents in an editor workspace, with AI help to fix grammar, simplify sentences, expand paragraphs, write definitions, or rewrite text.
- **Local Text-to-Speech**: Reads syntheses and chat answers aloud using a local Orpheus TTS integration (health-checked).
- **One-Click Startup**: A Windows `run.bat` script sets up a Python virtual environment, installs dependencies, and launches the app.

---

## Tech Stack

### Frontend (Single Page Application)

- **Structure**: Vanilla HTML5 with SVG icons (`icons.js`).
- **Styling**: A dark-themed responsive CSS layout (`theme.css`) using CSS grid and transitions.
- **Interactivity**: HTML5 Drag-and-Drop API, canvas previews, client-side password hashing.
- **State Management**: A dependency-free JavaScript client, synced with `localStorage` for tab and active-notebook state.

### Backend (FastAPI API Server)

- **Web Server**: FastAPI, served by Uvicorn.
- **Vector Search**: Milvus Lite, an embedded file-based database (`data/carnetlm.db`) that needs no separate server.
- **Embeddings**: Local FastEmbed library running the `BAAI/bge-small-en-v1.5` model.
- **Hybrid Search**: BM25 keyword matching via `bm25s`.
- **Data Storage**: SQLite (`data/memory.db`) for notebook settings, study decks, notes, and password credentials.
- **Document Processing**: `pymupdf` for PDF text extraction, `easyocr` for OCR.
- **Web Scraping**: `httpx`, `beautifulsoup4`, and `trafilatura` for article extraction.
- **YouTube Extraction**: `yt-dlp` for subtitle parsing.
- **LLM Engine**: Local Ollama (recommended model: `qwen2.5`), with an optional Google Gemini 2.5 Flash API client for cloud acceleration.

---

## Directory Structure

```
CarnetLM/
├── backend/            # FastAPI main router, lifespan managers, and models
│   ├── helpers.py      # Prompt templates, export formats, synthesis functions
│   └── main.py         # Main webapp route definitions & processing logic
├── src/                # Modular Python sub-packages
│   ├── discovery/      # Web search tools (DuckDuckGo integration)
│   ├── document_processing/ # PDF text extraction and OCR image support
│   ├── embeddings/     # Local FastEmbed transformer wrapper
│   ├── generation/     # RAG prompt generation, routing and retrieval
│   ├── ingest/         # Content chunking and background ingest pipelines
│   ├── llm/            # LLMRouter supporting Gemini API & local Ollama fallback
│   ├── memory/         # Local SQLite storage (notes, flashcards, notebooks metadata)
│   ├── tts/            # Local Orpheus TTS connector client
│   └── vector_database/# Local Milvus Lite wrapper
├── static/             # Responsive frontend Single Page Application
│   ├── index.html      # Main app viewport, modals, study desks and tabs
│   ├── theme.css       # Dark theme palette, styles, transitions, grid layout
│   └── icons.js        # Lucide vector icon components
├── run.bat             # Automated Windows setup and startup file
├── pyproject.toml      # Project packaging metadata and dependencies
└── requirements.txt    # Standard package-requirements manifest
```

---

## Quick Start

### 1. Ollama Setup (Required for Local-First)

CarnetLM runs entirely on your local machine:

1. Download and install [Ollama](https://ollama.ai).
2. Make sure Ollama is running in the background.
3. Pull the recommended model (Qwen 2.5):
   ```bash
   ollama pull qwen2.5
   ```

### 2. Configure Environment

1. Copy `.env.example` to create your `.env` file:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and configure your settings:
   - `OLLAMA_MODEL=qwen2.5` is set by default.
   - Optional: add a `GEMINI_API_KEY` to use Gemini 2.5 Flash instead. Leave it blank to run entirely offline on Ollama.

### 3. Start the Application

#### Windows Setup (Automatic)

Double-click `run.bat` in the root directory. It checks for the `uv` tool manager, runs `uv sync`, and opens the browser at **http://localhost:8000** automatically.

#### macOS / Linux Setup (Manual)

1. Make sure you have **Python 3.11+** installed.
2. Install the [uv package manager](https://docs.astral.sh/uv/):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. Install dependencies:
   ```bash
   uv sync
   ```
4. Start the FastAPI server:
   ```bash
   uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```
5. Open **http://localhost:8000** in your browser.

---

## API Endpoints Directory

Routes defined in `backend/main.py`:

| Method                                  | Endpoint                                               | Description                                                            |
| :-------------------------------------- | :----------------------------------------------------- | :--------------------------------------------------------------------- |
| **Health & Configuration**              |                                                        |                                                                        |
| `GET`                                   | `/api/health`                                          | Diagnostic state of LLM models, Milvus, SQLite db, and TTS engine      |
| `GET`                                   | `/api/models`                                          | List all available local Ollama and cloud Gemini models                |
| `POST`                                  | `/api/models/select`                                   | Set the active model selection for the session                         |
| `GET`                                   | `/api/chunking/profiles`                               | Fetch chunking profiles (paragraphs, words, tokens)                    |
| `GET`                                   | `/api/settings/chunking`                               | Retrieve current notebook chunking setup parameters                    |
| `PUT`                                   | `/api/settings/chunking`                               | Save/modify chunk tokens and overlap tolerances                        |
| `GET`                                   | `/api/settings/performance`                            | Query performance settings (Quality vs. Fast speed profiles)           |
| `PUT`                                   | `/api/settings/performance`                            | Save performance settings profile                                      |
| `GET`                                   | `/api/settings/discover`                               | Query web discovery preferences                                        |
| `PUT`                                   | `/api/settings/discover`                               | Modify web discovery settings                                          |
| **Web Discovery Search**                |                                                        |                                                                        |
| `POST`                                  | `/api/discover/search`                                 | Search online using DuckDuckGo search providers                        |
| `POST`                                  | `/api/discover/ingest`                                 | Download and ingest search result target pages                         |
| **Ingestion Pipeline**                  |                                                        |                                                                        |
| `GET`                                   | `/api/ingest/jobs`                                     | Get active document-parsing job queue lists                            |
| `GET`                                   | `/api/ingest/jobs/{job_id}`                            | Check status details on an ongoing file ingestion job                  |
| `POST`                                  | `/api/upload`                                          | Upload PDF, TXT, MD documents (supports OCR image extract)             |
| `POST`                                  | `/api/url`                                             | Extract and ingest content from website articles                       |
| `POST`                                  | `/api/youtube`                                         | Crawl transcripts from YouTube videos                                  |
| `GET`                                   | `/api/sources`                                         | Get list of all source references in the current notebook              |
| `DELETE`                                | `/api/sources/{source_id}`                             | Remove a source reference and clear its vector store chunks            |
| `GET`                                   | `/api/sources/{source_id}/content`                     | Retrieve raw text contents of an ingested document                     |
| `PUT`                                   | `/api/sources/{source_id}/content`                     | Modify or write edits back to an ingested source                       |
| `POST`                                  | `/api/sources/refresh`                                 | Re-run ingestion crawl pipelines on a document                         |
| **Notebook Workspaces**                 |                                                        |                                                                        |
| `GET`                                   | `/api/notebooks`                                       | Fetch list of active subject notebooks                                 |
| `POST`                                  | `/api/notebooks`                                       | Create a new subject notebook (supports password protection)           |
| `POST`                                  | `/api/notebooks/{notebook_id}/verify`                  | Authenticate credentials for password-locked notebooks                 |
| `PUT`                                   | `/api/notebooks/{notebook_id}`                         | Rename notebook workspaces                                             |
| `DELETE`                                | `/api/notebooks/{notebook_id}`                         | Drop a notebook and remove all associated local sources and db indices |
| **Workspace Chat & Search**             |                                                        |                                                                        |
| `POST`                                  | `/api/chat/stream`                                     | Stream chatbot query answers using semantic-hybrid RAG                 |
| `GET`                                   | `/api/history`                                         | Retrieve full chat history for the active notebook                     |
| `DELETE`                                | `/api/history`                                         | Wipe chat log history clean                                            |
| `POST`                                  | `/api/search`                                          | Execute quick semantic queries over local vector databases             |
| `POST`                                  | `/api/chat/export`                                     | Export chat logs as Markdown or Text transcripts                       |
| **Synthesis & Research**                |                                                        |                                                                        |
| `GET`                                   | `/api/summary`                                         | Query active synthesis summaries generated from local source contexts  |
| `POST`                                  | `/api/summary`                                         | Force-generate a brand-new notebook synthesis summary                  |
| `POST`                                  | `/api/compare`                                         | Synthesize side-by-side comparisons of multiple sources                |
| `POST`                                  | `/api/clipboard`                                       | Quick-ingest pasted clipboard texts as source references               |
| **Notes & Document Workspace**          |                                                        |                                                                        |
| `GET`                                   | `/api/notebooks/{notebook_id}/notes`                   | Fetch notes written inside the active notebook workspace               |
| `POST`                                  | `/api/notebooks/{notebook_id}/notes`                   | Create a new note                                                      |
| `GET`                                   | `/api/notes/{note_id}`                                 | Retrieve individual note details                                       |
| `PUT`                                   | `/api/notes/{note_id}`                                 | Update note title and details                                          |
| `POST`                                  | `/api/notes/{note_id}/append`                          | Append text strings directly to a note                                 |
| `DELETE`                                | `/api/notes/{note_id}`                                 | Delete notes from SQLite databases                                     |
| `POST`                                  | `/api/notes/{note_id}/index`                           | Toggle indexing note content into the RAG vector search database       |
| `GET`                                   | `/api/notebooks/{notebook_id}/document`                | Get compiled synthesis document content                                |
| `PUT`                                   | `/api/notebooks/{notebook_id}/document`                | Write compiler edits back to active synthesis documents                |
| **AI Assist & Document Export**         |                                                        |                                                                        |
| `POST`                                  | `/api/ai/assist`                                       | Run AI editor operations (simplify, define, expand, grammar, rewrite)  |
| `POST`                                  | `/api/export`                                          | Export notebooks to DOCX, PDF, TXT, or Markdown formats                |
| **Flashcard Concepts (Leitner System)** |                                                        |                                                                        |
| `POST`                                  | `/api/concepts/generate`                               | Trigger AI generation of study flashcards from ingested sources        |
| `GET`                                   | `/api/concepts`                                        | Fetch flashcards deck list for the active notebook workspace           |
| `POST`                                  | `/api/concepts`                                        | Manually insert a newly designed card (prepended to index 0)           |
| `POST`                                  | `/api/concepts/reorder`                                | Persist customized Keep-style drag-and-drop index sorting              |
| `POST`                                  | `/api/concepts/grade`                                  | Grade card reviews to progress boxes (Leitner levels 1 to 5)           |
| `DELETE`                                | `/api/concepts/{concept_id}`                           | Remove a concept card from study decks                                 |
| `POST`                                  | `/api/notebooks/{notebook_id}/concepts/reset-progress` | Wipe spaced repetition history, resetting all cards to Box 1           |
| **Text-to-Speech**                      |                                                        |                                                                        |
| `GET`                                   | `/api/tts/health`                                      | Check Local Orpheus TTS health connection status                       |
| `POST`                                  | `/api/tts`                                             | Convert text to speech audio waveforms                                 |
| **Static Viewport Router**              |                                                        |                                                                        |
| `GET`                                   | `/`                                                    | Serves the responsive SPA web frontend                                 |

---

## Data Privacy

CarnetLM keeps uploaded material and generated data local. The databases (`memory.db` and `carnetlm.db`), raw source files, and RAG embeddings cache all live inside the `data/` folder, which is listed in `.gitignore`. Nothing is synced to GitHub, and pulling updates won't touch your local data.

---

## License

This project is licensed under the [MIT License](LICENSE).
