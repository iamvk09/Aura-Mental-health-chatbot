# AURA - Mental Health Chatbot

**AURA** (Artificial Understanding & Resilience Agent) is a fully offline Python chatbot built on **Retrieval-Augmented Generation (RAG)** and **Natural Language Processing (NLP)**. Uses sentence transformers for semantic embeddings and FAISS for fast vector similarity search. Features user authentication, encrypted conversations, mental health assessment, and can be deployed to various platforms.

## Architecture: RAG + NLP Pipeline

### RAG (Retrieval-Augmented Generation) Components:

1. **Indexing Phase** (`ingest.py`):
   - Document chunking and preprocessing
   - NLP-based text embedding using SentenceTransformers
   - FAISS vector index construction
   - Metadata storage for retrieval

2. **Retrieval Phase** (`chat.py`):
   - Query embedding and semantic search
   - Top-K similarity retrieval
   - Relevance ranking using NLP techniques

3. **Generation Phase** (`chat.py`):
   - Context synthesis from retrieved chunks
   - NLP-based response generation
   - Empathetic, context-aware responses

### NLP Features:

- **Text Preprocessing**: Cleaning, normalization, tokenization
- **Semantic Embeddings**: Sentence-level embeddings for meaning understanding
- **Key Phrase Extraction**: Identifies important concepts
- **Relevance Scoring**: Combines semantic similarity with NLP features
- **Context Synthesis**: Intelligently combines multiple retrieved chunks
- **Response Generation**: Creates coherent, empathetic responses from context

## Features

- ✅ **RAG Pipeline** - Full Retrieval-Augmented Generation implementation
- ✅ **NLP Processing** - Advanced text understanding and processing
- ✅ **100% Offline** - No API calls, no internet required (can be deployed)
- ✅ **Fast Semantic Search** - FAISS vector similarity search
- ✅ **Context-Aware Responses** - Synthesizes information from multiple sources
- ✅ **Empathetic Responses** - Mental health-focused, supportive tone
- ✅ **Crisis Detection** - Automatic detection and appropriate referrals
- ✅ **User Authentication** - Secure login/registration system
- ✅ **Encrypted Conversations** - All chats encrypted for privacy
- ✅ **Mental Health Assessment** - 5-question assessment on login
- ✅ **Conversation History** - View chat history in profile
- ✅ **SOS Button** - Emergency support button
- ✅ **Web Interface** - Modern, responsive chat UI
- ✅ **Profile Page** - User profile with chat history

## Quick Start (TL;DR)

```bash
# 1. Setup
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 2. Build index (one-time)
python ingest.py

# 3. Run chatbot
python app.py  # Web interface (recommended)
# OR
python chat.py  # Terminal interface
```

**That's it!** See detailed instructions below.

---

## Installation

### 1. Create Virtual Environment

```bash
python -m venv venv
```

### 2. Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `sentence-transformers` - For generating embeddings
- `faiss-cpu` - For fast vector similarity search
- `numpy` - For numerical operations
- `tqdm` - For progress bars
- `flask` - For web interface (optional, for web UI)

## Setup

### Prepare Your Dataset

**Option 1: Use existing topics.json (Recommended for quick start)**

If you have `data/topics.json`, convert it to the required format:

```bash
python convert_topics_to_chunks.py
```

This will create `kb_chunks.json` from your topics data.

**Option 2: Create your own kb_chunks.json**

Your dataset should be in `kb_chunks.json` with the following format:

```json
[
  {
    "text": "Your text content here...",
    "source": "Source name or title",
    "metadata": {
      "additional": "metadata fields"
    }
  }
]
```

### Run Data Ingestion

Generate embeddings and build the FAISS index:

```bash
python ingest.py
```

This will:
- Load `kb_chunks.json`
- Generate embeddings using `all-MiniLM-L6-v2`
- Build a FAISS index
- Save to `data/chatbot_index/`:
  - `faiss_index.bin` - FAISS index file
  - `embeddings.npy` - Embeddings array
  - `metadata.json` - Chunk metadata and sources

## Usage

### Option 1: Web Interface (Recommended)

Start the web-based chatbot:

```bash
python app.py
```

Then open your browser and go to: **http://localhost:5000**

The web interface provides:
- Modern, user-friendly chat interface
- Real-time messaging
- Better visual experience
- Mobile-responsive design

### Option 2: Terminal Interface

Start the terminal-based chatbot:

```bash
python chat.py
```

### Chat Commands

- Type your questions or messages
- Type `quit`, `exit`, or `bye` to end the conversation (terminal only)
- Press `Ctrl+C` to exit (terminal only)

### Important: Treatment Advice

The chatbot only provides treatment/cure advice when explicitly requested. Use phrases like:
- "How can I manage anxiety?"
- "What can I do to cope with stress?"
- "How do I improve my mental health?"
- "Ways to treat depression"

For general questions, the bot provides informational responses without treatment advice.

### Example Conversation

```
You: I'm feeling really anxious about my exams

Bot:
Relevant sources:
1. Understanding and Coping with Anxiety
2. Mental Health: Self-care and Exercise

Here is a supportive response:

I understand this can be really difficult, and I want you to know your feelings are valid.

Anxiety is a common mental health condition that affects millions of people worldwide. It can be caused by a variety of factors, including genetics, environmental stressors, and life experiences. If you're struggling with anxiety, it's important to seek help from a mental health professional.

Here's something that might help:
There are also several self-help techniques that can be effective in managing anxiety, such as practicing relaxation techniques like deep breathing or mindfulness.

Remember, you're not alone in this, and seeking help is a sign of strength.
```

## RAG Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG PIPELINE FLOW                         │
└─────────────────────────────────────────────────────────────┘

INDEXING PHASE (ingest.py):
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  kb_chunks   │ --> │   NLP Text   │ --> │  Embeddings  │
│    .json     │     │ Preprocessing│     │  Generation  │
└──────────────┘     └──────────────┘     └──────────────┘
                                                    │
                                                    v
                                          ┌──────────────┐
                                          │ FAISS Index  │
                                          │  + Metadata  │
                                          └──────────────┘

RETRIEVAL PHASE (chat.py):
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ User Query   │ --> │   Query     │ --> │  Semantic    │
│              │     │ Embedding   │     │   Search     │
└──────────────┘     └──────────────┘     └──────────────┘
                                                    │
                                                    v
                                          ┌──────────────┐
                                          │ Top-K Chunks │
                                          │  Retrieved   │
                                          └──────────────┘

GENERATION PHASE (chat.py):
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Retrieved    │ --> │   Context   │ --> │  Empathetic  │
│   Chunks     │     │  Synthesis  │     │  Response    │
│              │     │   (NLP)      │     │  Generation  │
└──────────────┘     └──────────────┘     └──────────────┘
```

## Crisis Detection

The chatbot automatically detects crisis keywords (suicide, self-harm, etc.) and provides appropriate crisis resources:

```
You: I want to kill myself

Bot:
I'm really sorry you're feeling this way.

Your safety matters. I cannot provide crisis help.

Please contact your local emergency number or a suicide prevention hotline immediately.

In India: Call Aasra at 9152987821 (24x7).
```

## File Structure

See "Project Structure" section below for complete file listing.

## Troubleshooting

### "Dataset file not found: kb_chunks.json"

Make sure `kb_chunks.json` exists in the current directory. The file should contain a JSON array of chunks with `text`, `source`, and `metadata` fields.

### "Index not found"

Run `python ingest.py` first to generate the index files.

### Model Download

The first time you run the scripts, `sentence-transformers` will download the `all-MiniLM-L6-v2` model. This is a one-time download that happens automatically.

### "Assessment not completing"

Make sure you answer all 5 assessment questions. The chat will be disabled until assessment is complete.

### "User data not persisting"

Ensure the `data/` directory has write permissions. On deployed platforms, use persistent storage volumes.

## RAG Pipeline Details

### How It Works:

1. **Query Processing (NLP)**:
   - User query is preprocessed and normalized
   - Converted to semantic embedding using SentenceTransformer

2. **Retrieval (RAG)**:
   - Semantic similarity search in FAISS index
   - Top-K chunks retrieved based on cosine similarity
   - Relevance ranking using NLP features (key phrases, semantic matching)

3. **Context Synthesis (NLP)**:
   - Multiple retrieved chunks are intelligently combined
   - Redundancy removal and sentence selection
   - Coherent context construction

4. **Response Generation (RAG)**:
   - Context-aware response generation
   - Empathetic language with mental health focus
   - Actionable advice extraction from context

## Technical Notes

- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **Similarity Metric**: Cosine similarity (normalized embeddings)
- **Retrieval**: Top 5 most similar chunks per query
- **NLP Processing**: Text preprocessing, key phrase extraction, relevance scoring
- **Context Synthesis**: Combines top 3 ranked chunks intelligently
- **All processing is local** - no data sent to external services (unless deployed)
- **Encryption**: Conversations encrypted using Fernet (password-derived keys)
- **Storage**: JSON-based file storage (can be migrated to database)
- **Session Management**: Flask sessions with encrypted cookies
- **Production Server**: Gunicorn with multiple workers

## Deployment

This application can be deployed to various platforms. See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

### Quick Deploy Options:

**Railway (Recommended):**
```bash
railway init
railway up
```

**Render:**
- Connect GitHub repo
- Set build command: `pip install -r requirements.txt && python ingest.py`
- Set start command: `gunicorn app:app --bind 0.0.0.0:$PORT`

**Docker:**
```bash
docker build -t aura-chatbot .
docker run -p 5000:5000 aura-chatbot
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete deployment guide with recommended platforms (Railway, Render, Docker, etc.).

## Project Structure

```
.
├── app.py                  # Flask web application
├── chat.py                 # Chatbot core logic (RAG pipeline)
├── ingest.py               # Data ingestion and indexing
├── auth.py                 # User authentication and encryption
├── assessment.py           # Mental health assessment
├── topic_detector.py        # Topic detection for mental health focus
├── convert_topics_to_chunks.py  # Dataset converter
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
├── gunicorn_config.py     # Production server config
├── templates/             # HTML templates
│   ├── login.html         # Login/Register page
│   ├── chat.html          # Main chat interface
│   └── profile.html       # User profile page
└── data/                  # Data storage
    ├── users.json         # User accounts (gitignored)
    ├── conversations/     # Encrypted conversations (gitignored)
    └── chatbot_index/     # FAISS index and embeddings
```

## Security Best Practices

- **Change SECRET_KEY**: Always set a strong `SECRET_KEY` environment variable in production
- **HTTPS**: Use HTTPS in production (most platforms provide this automatically)
- **Data Backup**: Regularly backup the `data/` directory
- **Access Control**: Consider adding rate limiting for production
- **Monitoring**: Set up logging and monitoring for production deployments

## Contributing

This is a personal project. Feel free to fork and modify for your needs.

## License

This is a local, offline chatbot. All processing happens on your machine or deployed server. Use responsibly and ensure compliance with local data protection regulations.

