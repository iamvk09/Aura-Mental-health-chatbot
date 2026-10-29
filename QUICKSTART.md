# Quick Start Guide - How to Run the Bot

## Step-by-Step Instructions

### Step 1: Install Dependencies

Open your terminal/PowerShell in the project directory and run:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

**Note:** The first time you install `sentence-transformers`, it will download the `all-MiniLM-L6-v2` model (~90MB). This is automatic and only happens once.

### Step 2: Prepare Dataset (if not already done)

If `kb_chunks.json` doesn't exist, create it from your topics:

```bash
python convert_topics_to_chunks.py
```

✅ **You already have `kb_chunks.json`** - you can skip this step!

### Step 3: Build the Index (One-time setup)

Generate embeddings and build the FAISS index:

```bash
python ingest.py
```

This will:
- Load your `kb_chunks.json` file
- Generate embeddings for all chunks
- Build the FAISS search index
- Save everything to `data/chatbot_index/`

**Expected output:**
```
============================================================
RAG Pipeline - Data Ingestion
Building knowledge base with NLP embeddings
============================================================
Loading dataset from kb_chunks.json...
Loaded 40 chunks from dataset

Loading SentenceTransformer model: all-MiniLM-L6-v2...
Model loaded successfully
Generating embeddings using all-MiniLM-L6-v2...
Embedding chunks: 100%|████████████| 2/2 [00:02<00:00,  1.23s/it]
Generated embeddings shape: (40, 384)
Building FAISS index...
FAISS index built with 40 vectors
Saving index and metadata...
✅ Ingestion complete!
```

### Step 4: Start the Chatbot

**Option A: Web Interface (Recommended)**

```bash
python app.py
```

Then open your browser and go to: **http://localhost:5000**

**Option B: Terminal Interface**

```bash
python chat.py
```

**Expected output (Terminal):**
```
============================================================
Offline RAG-based Mental Health Chatbot
Powered by NLP & Retrieval-Augmented Generation
============================================================
Loading model and index...
Model all-MiniLM-L6-v2 loaded
Loading chatbot index...
Loaded index with 40 vectors

============================================================
Chatbot ready! Type 'quit', 'exit', or 'bye' to end the conversation.
============================================================

You: 
```

**Expected output (Web):**
```
============================================================
Starting Web-based Mental Health Chatbot
============================================================
Initializing chatbot...
Loading model...
Model all-MiniLM-L6-v2 loaded
Loading chatbot index...
Loaded index with 40 vectors
Index loaded successfully

============================================================
Chatbot ready!
Open your browser and go to: http://localhost:5000
============================================================
```

### Step 5: Chat!

Type your questions or messages. Examples:

```
You: I'm feeling anxious about my exams

You: How can I manage stress?

You: What are some self-care tips?

You: quit  (to exit)
```

## Troubleshooting

### "Dataset file not found: kb_chunks.json"
- Run `python convert_topics_to_chunks.py` first
- Or create your own `kb_chunks.json` file

### "Index not found"
- Run `python ingest.py` first to build the index

### "Module not found" errors
- Make sure virtual environment is activated: `venv\Scripts\activate`
- Install dependencies: `pip install -r requirements.txt`

### Model download is slow
- This is normal on first run. The model (~90MB) downloads automatically
- It's cached for future use

## File Structure After Setup

```
.
├── venv/                    # Virtual environment
├── kb_chunks.json          # Your dataset
├── ingest.py               # Indexing script
├── chat.py                 # Chatbot script
├── requirements.txt        # Dependencies
└── data/
    └── chatbot_index/      # Generated index files
        ├── faiss_index.bin
        ├── embeddings.npy
        └── metadata.json
```

## Next Steps

- Customize `kb_chunks.json` with your own mental health content
- Adjust `TOP_K` in `chat.py` to retrieve more/fewer chunks
- Modify crisis keywords in `chat.py` if needed

Happy chatting! 🌟

