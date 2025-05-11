# lamben-search

> A lightweight semantic search engine for Elvish glossaries and beyond.

**lamben-search** is a proof-of-concept project that explores AI-powered semantic search for multilingual or fantasy-themed glossaries, using sentence embeddings and vector similarity. Inspired by the linguistic depth of Tolkien’s languages, it's designed as a side project to demonstrate practical applications of NLP in glossary-based platforms.

---

## Features

-  Semantic search using sentence-transformer embeddings
-  Fast vector indexing with FAISS
-  Glossary-based dataset (English ↔ Elvish)
-  RESTful API with FastAPI
-  Docker support (optional)
-  Easy to extend for any language pair or domain

---

## Tech Stack

- Python 3.10+
- [SentenceTransformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [FastAPI](https://fastapi.tiangolo.com/)
- Pandas (for dataset handling)
- Uvicorn (for serving the API)

---

## Dataset

This demo uses a handcrafted dataset of 100 English terms and their Elvish (Quenya/Sindarin) equivalents, each with contextual definitions.

You can find the CSV file here:  
`lotr_elvish_glossary.csv`

---

## Setup

```bash
git clone https://github.com/yourusername/lamben-search.git
cd lamben-search
pip install -r requirements.txt
uvicorn app:app --reload


## API Usage

**POST** `/semantic-search`
**Request JSON:**
```json
{
    "query" : "heat"
}
```

**Response JSON:**
```json
{
  "results": [
    {
      "english": "fire",
      "elvish": "naur",
      "definition": "Element associated with warmth and destruction"
    },
  ]
}
```

