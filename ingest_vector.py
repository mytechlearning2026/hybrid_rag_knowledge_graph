import os
import json
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------------------------------------------------
# 2. File paths
# --------------------------------------------------

PDF_PATH = "data/spotify_web_app_architecture.pdf"

VECTOR_DIR = "vector_store"

os.makedirs(VECTOR_DIR, exist_ok=True)

# --------------------------------------------------
# 3. Extract text from PDF
# --------------------------------------------------

print("Reading PDF...")

reader = PdfReader(PDF_PATH)

documents = []

for page_number, page in enumerate(reader.pages, start=1):

    text = page.extract_text()

    if text and text.strip():

        documents.append({
            "text": text,
            "page": page_number,
            "source": PDF_PATH
        })

print(f"Total pages extracted: {len(documents)}")

# --------------------------------------------------
# 4. Split text into chunks
# --------------------------------------------------

print("Splitting text into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = []

for document in documents:

    split_texts = text_splitter.split_text(document["text"])

    for chunk_text in split_texts:

        chunks.append({
            "text": chunk_text,
            "page": document["page"],
            "source": document["source"]
        })

print(f"Total chunks created: {len(chunks)}")

# --------------------------------------------------
# 5. Generate OpenAI embeddings
# --------------------------------------------------

print("Generating embeddings...")

texts = [chunk["text"] for chunk in chunks]

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)

embeddings = [
    item.embedding
    for item in response.data
]

# Convert embeddings to NumPy array

embedding_array = np.array(
    embeddings,
    dtype="float32"
)

# --------------------------------------------------
# 6. Create FAISS vector index
# --------------------------------------------------

print("Creating FAISS index...")

# Normalize vectors so L2 search approximates cosine similarity
faiss.normalize_L2(embedding_array)

dimension = embedding_array.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embedding_array)

# --------------------------------------------------
# 7. Save FAISS index
# --------------------------------------------------

index_path = os.path.join(
    VECTOR_DIR,
    "faiss_index.bin"
)

faiss.write_index(index, index_path)

# --------------------------------------------------
# 8. Save chunk metadata and text
# --------------------------------------------------

chunks_path = os.path.join(
    VECTOR_DIR,
    "chunks.json"
)

with open(chunks_path, "w", encoding="utf-8") as file:

    json.dump(
        chunks,
        file,
        indent=4,
        ensure_ascii=False
    )

# --------------------------------------------------
# 9. Completion message
# --------------------------------------------------

print("\n================================")
print("VECTOR INGESTION SUCCESSFUL!")
print("================================")

print(f"Pages processed: {len(documents)}")
print(f"Chunks created: {len(chunks)}")
print(f"Embedding dimension: {dimension}")

print(f"FAISS index saved: {index_path}")
print(f"Chunks saved: {chunks_path}")