import os
import json

import faiss
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Simple RAG Chatbot")

# Load FAISS vector database
index = faiss.read_index("vector_store/faiss_index.bin")

# Load text chunks
with open("vector_store/chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)


@app.get("/")
def home():
    return {"message": "RAG API is running!"}


@app.post("/ask")
def ask_question(question: str):

    # 1. Convert question into embedding
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )

    query_embedding = np.array(
        [response.data[0].embedding],
        dtype="float32"
    )

    # 2. Normalize embedding
    faiss.normalize_L2(query_embedding)

    # 3. Search FAISS
    scores, indices = index.search(query_embedding, 3)

    # 4. Retrieve relevant chunks
    retrieved_chunks = []

    for idx in indices[0]:
        if idx >= 0:
            retrieved_chunks.append(chunks[idx])

    # 5. Build context
    context = "\n\n".join(
        chunk["text"] for chunk in retrieved_chunks
    )

    # 6. Generate answer using OpenAI
    answer_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the user's question using only the "
                    "provided context. If the answer is not "
                    "available in the context, say you don't "
                    "know based on the document."
                )
            },
            {
                "role": "user",
                "content": f"""
Context:
{context}

Question:
{question}
"""
            }
        ],
        temperature=0
    )

    answer = answer_response.choices[0].message.content

    # 7. Return answer and sources
    sources = [
        {
            "page": chunk.get("page"),
            "source": chunk.get("source"),
            "text": chunk["text"]
        }
        for chunk in retrieved_chunks
    ]

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }