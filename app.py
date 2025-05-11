from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas
import numpy
import faiss
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
    results:List[TermResult]

@app.post("/semantic-search",response_model=QueryResponse)
def semantic_search(request: QueryRequest):
    query_embedding = model.encode([request.query],convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k=5)
    results = [
        {
            "english": data_set.iloc[i]["english"],
            "elvish": data_set.iloc[i]["elvish"],
            "definition": data_set.iloc[i]["definition"]
        }
        for i in indices[0]
    ]
    return {"results": results}
