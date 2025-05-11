from typing import List

import faiss
import numpy
import pandas
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Load the glossary dataset
data_set = pandas.read_csv("lotr_elvish_glossary.csv")
corpus = data_set["definition"].tolist()

# Load model and generate sentance embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
corpus_embeddings = model.encode(corpus, convert_to_numpy=True)

# Create index
dimension = corpus_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(corpus_embeddings)

app = FastAPI()


class QueryRequest(BaseModel):
    query: str


class TermResult(BaseModel):
    english: str
    elvish: str
    definition: str


class QueryResponse(BaseModel):
    results: List[TermResult]


@app.post("/semantic-search", response_model=QueryResponse)
def semantic_search(request: QueryRequest):
    query_embedding = model.encode([request.query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k=5)
    results = [
        {
            "english": data_set.iloc[i]["english"],
            "elvish": data_set.iloc[i]["elvish"],
            "definition": data_set.iloc[i]["definition"],
        }
        for i in indices[0]
    ]
    return {"results": results}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/exact-match", response_model=List[TermResult])
def exact_match(term: str):
    matches = data_set[data_set["english"].str.lower() == term.lower()]
    if matches.empty:
        raise HTTPException(status_code=404, detail="Term not found")
    return matches.to_dict(orient="records")


@app.get("/glossary", response_model=List[TermResult])
def get_glossary():
    return data_set.to_dict(orient="records")


@app.get("/term/{english}", response_model=TermResult)
def get_term_by_english(english: str):
    match = data_set[data_set["english"].str.lower() == english.lower()]
    if match.empty:
        raise HTTPException(status_code=404, detail="Term not found")
    row = match.iloc[0]
    return {
        "english": row["english"],
        "elvish": row["elvish"],
        "definition": row["definition"],
    }


@app.get("/random", response_model=TermResult)
def get_random_term():
    row = data_set.sample(1).iloc[0]
    return {
        "english": row["english"],
        "elvish": row["elvish"],
        "definition": row["definition"],
    }


@app.get("/languages")
def get_languages():
    return {
        "source": "English",
        "target": "Elvish",
        "note": "Future version could support isiZulu, Sesotho, etc.",
    }
